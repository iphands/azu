#include "tracking/ICPTracker.h"
#include "sensor/KinectSensor.h"
#include "tracking/ICPShared.h"
#include "utils/CoordinateMath.h"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kfusion {
namespace tracking {

// tracking:CPU-8: the level-scaled intrinsic helpers that used to sit here had
// zero call sites. Projection is intentionally full-resolution at every pyramid
// level (the model frame is built at full res); both backends do the same.
// Do not "restore" per-level intrinsics without first rescaling the model side.

ICPParams ICPTracker::sanitizeParams(const ICPParams& params) {
    ICPParams out = params;
    if (!std::isfinite(out.angle_threshold)) {
        out.angle_threshold = kAngleThresholdFallbackDeg;
    } else if (out.angle_threshold < kAngleThresholdMinDeg) {
        out.angle_threshold = kAngleThresholdMinDeg;
    } else if (out.angle_threshold > kAngleThresholdMaxDeg) {
        out.angle_threshold = kAngleThresholdMaxDeg;
    }
    return out;
}

ICPTracker::ICPTracker(const ICPParams& params)
    : params_(sanitizeParams(params))
{}

ICPResult ICPTracker::track(const sensor::FramePyramid& live,
                            const ModelFrame&           model,
                            const Eigen::Matrix4f&      pose_estimate,
                            const Eigen::Matrix4f&      ref_pose)
{
    ICPResult result;
    result.pose = pose_estimate;
    float last_final_step = std::numeric_limits<float>::infinity();

    for (int level = sensor::FramePyramid::LEVELS - 1; level >= 0; --level) {
        // A level with no iterations is skipped, so a coarse-only solve (finer
        // levels at 0, as relocalization scoring uses) keeps its result.
        if (params_.max_iterations[level] <= 0) continue;
        ICPResult level_result = trackLevel(live.levels[level], model, result.pose, ref_pose, level,
                                           params_.max_iterations[level]);
        if (std::isfinite(level_result.final_step)) {
            last_final_step = level_result.final_step;
        }
        level_result.final_step = last_final_step;
        result = level_result;
    }

    result.tracking_ok = result.pose.allFinite() &&
                         result.inliers > kMinInliersForOk &&
                         (result.converged || result.final_step <= kAcceptableFinalStep);
    return result;
}

ICPResult ICPTracker::trackLevel(const sensor::FrameData& live_level,
                                 const ModelFrame&        model,
                                 const Eigen::Matrix4f&   pose_estimate,
                                 const Eigen::Matrix4f&   ref_pose,
                                 int                      level,
                                 int                      max_iter)
{
    ICPResult result;
    result.pose = pose_estimate;

    for (int iter = 0; iter < max_iter; ++iter) {
        Eigen::Matrix<float, 6, 6> A = Eigen::Matrix<float, 6, 6>::Zero();
        Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
        float residual    = 0.0f;
        int   inlier_count = 0;

        if (!buildLinearSystem(live_level, model, result.pose, ref_pose, A, b, residual, inlier_count,
                               result.valid_live_points, result.valid_model_points, result.projected_points,
                               result.dist_filtered, result.angle_filtered))
            break;

        if (inlier_count < kMinInliersForIteration) break;

        result.information = A;
        A += Eigen::Matrix<float, 6, 6>::Identity() * dampingForHessian(A, level);
        Eigen::Matrix<float, 6, 1> x = A.ldlt().solve(b);

        if (!x.allFinite() || x.head<3>().norm() > kTranslationCapStep ||
            x.tail<3>().norm() > kRotationCapStep) {
            break;
        }

        const float step_norm = x.norm();
        float tx = x(0), ty = x(1), tz = x(2);
        float rx = x(3), ry = x(4), rz = x(5);

        Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
        float angle = std::sqrt(rx*rx + ry*ry + rz*rz);
        if (angle > 1e-4f) {
            Eigen::Vector3f axis(rx, ry, rz);
            axis.normalize();
            R = Eigen::AngleAxisf(angle, axis).toRotationMatrix();
        } else {
            R(0,1) = -rz; R(0,2) =  ry;
            R(1,0) =  rz; R(1,2) = -rx;
            R(2,0) = -ry; R(2,1) =  rx;
        }

        Eigen::Matrix4f delta = Eigen::Matrix4f::Identity();
        delta.block<3,3>(0,0) = R;
        delta(0,3) = tx; delta(1,3) = ty; delta(2,3) = tz;

        result.pose = result.pose * delta;
        result.pose.block<3, 3>(0, 0) = projectToSO3(result.pose.block<3, 3>(0, 0));

        result.final_step = step_norm;
        result.error      = residual / static_cast<float>(std::max(inlier_count, 1));
        result.inliers    = inlier_count;

        if (step_norm < kConvergenceStep) {
            result.converged = true;
            break;
        }
    }

    return result;
}

bool ICPTracker::buildLinearSystem(const sensor::FrameData& live,
                                   const ModelFrame&        model,
                                   const Eigen::Matrix4f&   pose,
                                   const Eigen::Matrix4f&   ref_pose,
                                   Eigen::Matrix<float,6,6>& A,
                                   Eigen::Matrix<float,6,1>& b,
                                   float& residual,
                                   int&   inlier_count,
                                   int&   valid_live,
                                   int&   valid_model,
                                   int&   projected,
                                   int&   dist_filtered,
                                   int&   angle_filtered)
{
    const int W = live.width;
    const int H = live.height;
    const float angle_thresh_cos = std::cos(params_.angle_threshold * M_PI / 180.0f);

    // Live Camera to World
    const Eigen::Matrix3f R_cw = pose.block<3,3>(0,0);
    const Eigen::Vector3f t_cw = pose.block<3,1>(0,3);

    // World to Reference Camera (the pose the model image was raycast at)
    const Eigen::Matrix4f ref_inv = ref_pose.inverse();
    const Eigen::Matrix3f R_rc = ref_inv.block<3,3>(0,0);
    const Eigen::Vector3f t_rc = ref_inv.block<3,1>(0,3);

    // Relative transform: Live Cam -> World -> Ref Cam
    const Eigen::Matrix3f R_rel = R_rc * R_cw;
    const Eigen::Vector3f t_rel = R_rc * t_cw + t_rc;

    int num_threads = num_threads_.load();
    if (num_threads <= 0) {
#ifdef _OPENMP
        num_threads = omp_get_max_threads();
#else
        num_threads = 1;
#endif
    }
    
    struct LocalAcc {
        float A_data[21]; 
        float b_data[6];
        float residual;
        int   count;
        int   valid_live, valid_model, projected;
        int   dist_filtered, angle_filtered;

        LocalAcc() : residual(0.0f), count(0),
                     valid_live(0), valid_model(0), projected(0),
                     dist_filtered(0), angle_filtered(0) 
        {
            for (int i = 0; i < 21; ++i) A_data[i] = 0.0f;
            for (int i = 0; i < 6; ++i) b_data[i] = 0.0f;
        }

        // Canonical Huber IRLS/MM accumulation (big-fix Todo 12, locked by
        // tests/icp_weighting_contract.cpp): with w = min(1, kHuberK/|e|),
        // curvature A += w*J*J^T, gradient b -= J*(w*e), and the reported
        // objective is the true Huber loss psi(|e|), all from one weight. The
        // pre-Todo-12 code paired an unweighted curvature and a (w*e)^2
        // "objective" with the w*e gradient - three inconsistent objectives;
        // docs/CANONICAL_SEMANTICS.md records why w^2 curvature is also wrong.
        inline void add(const float* J, float w, float weighted_err, float loss) {
            int k = 0;
            for (int i = 0; i < 6; ++i) {
                for (int j = i; j < 6; ++j) {
                    A_data[k++] += w * J[i] * J[j];
                }
                b_data[i] -= J[i] * weighted_err;
            }
            residual += loss;
            count++;
        }
    };

    std::vector<LocalAcc> local(num_threads);

    #pragma omp parallel for schedule(dynamic, 32) num_threads(num_threads)
    for (int y = 1; y < H - 1; ++y) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& acc = local[tid];

        for (int x = 1; x < W - 1; ++x) {
            int idx = y * W + x;
            const Eigen::Vector3f& live_v = live.vertices[idx];
            // Reject invalid and non-finite live vertices up front. The z bound is
            // unchanged for finite vertices; the allFinite test additionally drops
            // NaN/Inf, which the bare `<= 0.001f` predicate let through (a NaN z
            // compares false), so such a vertex never reaches the projection below.
            if (live_v.z() <= 0.001f || !live_v.allFinite()) continue;

            // Project live vertex into the reference camera (model image)
            Eigen::Vector3f v_ref = R_rel * live_v + t_rel;
            if (v_ref.z() <= 0.001f) continue;
            acc.valid_live++;

            float inv_z = 1.0f / v_ref.z();
            float model_x = sensor::FX * v_ref.x() * inv_z + sensor::CX;
            float model_y = sensor::FY * v_ref.y() * inv_z + sensor::CY;

            // Round the sub-pixel projection to the nearest model pixel with the
            // shared floor primitive: floor(model + 0.5) (round-half-up), the CPU
            // canonical rounding (docs/CANONICAL_SEMANTICS.md). floorToInt rejects
            // a non-finite or out-of-int-range projection (e.g. an inv_z blow-up)
            // before any integer coordinate is produced, replacing the old
            // static_cast<int>(model + 0.5f) that truncated toward zero and cast
            // NaN/Inf as undefined behavior.
            int mx = 0;
            int my = 0;
            if (!utils::floorToInt(model_x + 0.5f, &mx) ||
                !utils::floorToInt(model_y + 0.5f, &my)) {
                continue;
            }

            if (mx < 0 || mx >= sensor::FRAME_W || my < 0 || my >= sensor::FRAME_H) continue;
            acc.projected++;
            
            int midx = my * sensor::FRAME_W + mx;
            const Eigen::Vector3f& model_v_world = model.vertices[midx];
            const Eigen::Vector3f& model_n_world = model.normals[midx];
            if (!modelVertexIsValid(model_v_world) || !normalIsValid(model_n_world)) continue;
            acc.valid_model++;

            // Transform live vertex to world space for distance check
            Eigen::Vector3f v_live_world = R_cw * live_v + t_cw;
            float dist_sq = (v_live_world - model_v_world).squaredNorm();
            if (dist_sq > params_.dist_threshold * params_.dist_threshold) {
                acc.dist_filtered++;
                continue;
            }

            const Eigen::Vector3f& live_n = live.normals[idx];
            if (!normalIsValid(live_n)) continue;
            
            // Map live normal to world space for angle check
            Eigen::Vector3f n_live_world = R_cw * live_n;
            float dot = std::abs(n_live_world.dot(model_n_world));
            if (dot < angle_thresh_cos) {
                acc.angle_filtered++;
                continue;
            }

            // Point-to-plane error in world space
            float err = model_n_world.dot(v_live_world - model_v_world);

            // Reject a non-finite residual BEFORE the weight is evaluated: a
            // NaN/Inf err otherwise flows w = k/|err| into (err*w) and poisons
            // b and the objective. The Jacobian guard below cannot catch it:
            // J depends only on the model normal and the (finite-guarded) live
            // vertex, so a NaN model depth yields a NaN err with all six J
            // entries finite.
            if (!std::isfinite(err)) continue;

            // Jacobian J = [n_live_cam; cross(live_v, n_live_cam)]
            // where n_live_cam = R_cw^T * model_n_world
            Eigen::Vector3f n_live_cam = R_cw.transpose() * model_n_world;
            Eigen::Vector3f cross = live_v.cross(n_live_cam);

            float J[6] = { n_live_cam.x(), n_live_cam.y(), n_live_cam.z(), cross.x(), cross.y(), cross.z() };

            // Reject a non-finite Jacobian entry on ALL six components (the
            // pre-Todo-12 guard sampled only J[0] and J[3]), so one corrupted
            // correspondence can never enter A, b or the objective.
            if (!std::isfinite(J[0]) || !std::isfinite(J[1]) || !std::isfinite(J[2]) ||
                !std::isfinite(J[3]) || !std::isfinite(J[4]) || !std::isfinite(J[5])) {
                continue;
            }

            // Huber weight for robustness; kHuberK and psi come from
            // tracking/ICPShared.h (single source of truth).
            float abs_err = std::abs(err);
            float w = (abs_err <= kHuberK) ? 1.0f : kHuberK / abs_err;

            acc.add(J, w, err * w, huberLossFromAbs(abs_err));
        }
    }

    valid_live = 0; valid_model = 0; projected = 0; dist_filtered = 0; angle_filtered = 0;
    residual = 0; inlier_count = 0;

    for (auto& lacc : local) {
        int k = 0;
        for (int i = 0; i < 6; ++i) {
            for (int j = i; j < 6; ++j) {
                float val = lacc.A_data[k++];
                A(i, j) += val;
                if (i != j) A(j, i) += val;
            }
            b(i) += lacc.b_data[i];
        }
        residual    += lacc.residual;
        inlier_count += lacc.count;
        valid_live     += lacc.valid_live;
        valid_model    += lacc.valid_model;
        projected      += lacc.projected;
        dist_filtered  += lacc.dist_filtered;
        angle_filtered += lacc.angle_filtered;
    }

    return inlier_count > 0;
}


} // namespace tracking
} // namespace kfusion
