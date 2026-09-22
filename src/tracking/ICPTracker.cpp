#include "tracking/ICPTracker.h"
#include "sensor/KinectSensor.h"
#include "utils/CoordinateMath.h"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace kfusion {
namespace tracking {

static float getFx(int level) { return static_cast<float>(sensor::FX) / (1 << level); }
static float getFy(int level) { return static_cast<float>(sensor::FY) / (1 << level); }
static float getCx(int level) { return static_cast<float>(sensor::CX) / (1 << level); }
static float getCy(int level) { return static_cast<float>(sensor::CY) / (1 << level); }

ICPTracker::ICPTracker(const ICPParams& params)
    : params_(params)
{}

ICPResult ICPTracker::track(const sensor::FramePyramid& live,
                            const ModelFrame&           model,
                            const Eigen::Matrix4f&      pose_estimate,
                            const Eigen::Matrix4f&      ref_pose)
{
    ICPResult result;
    result.pose = pose_estimate;

    for (int level = sensor::FramePyramid::LEVELS - 1; level >= 0; --level) {
        result = trackLevel(live.levels[level], model, result.pose, ref_pose, level,
                            params_.max_iterations[level]);
    }

    result.tracking_ok = result.converged && result.inliers > 100;
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

        if (inlier_count < 10) break;

        A += Eigen::Matrix<float, 6, 6>::Identity() * 0.1f;
        Eigen::Matrix<float, 6, 1> x = A.ldlt().solve(b);

        if (!x.allFinite() || x.head<3>().norm() > 0.2f) {
            break;
        }

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

        result.pose     = result.pose * delta;
        
        // Orthonormalize rotation to prevent drift/ghosting
        Eigen::Matrix3f R_curr = result.pose.block<3,3>(0,0);
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(R_curr, Eigen::ComputeFullU | Eigen::ComputeFullV);
        result.pose.block<3,3>(0,0) = svd.matrixU() * svd.matrixV().transpose();

        result.error    = residual / static_cast<float>(std::max(inlier_count, 1));
        result.inliers  = inlier_count;

        if (x.norm() < 5e-5f) {
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

    // Reference Camera (Model) to World
    const Eigen::Matrix3f R_ref = ref_pose.block<3,3>(0,0);
    const Eigen::Vector3f t_ref = ref_pose.block<3,1>(0,3);

    // World to Reference Camera
    const Eigen::Matrix4f ref_inv = ref_pose.inverse();
    const Eigen::Matrix3f R_rc = ref_inv.block<3,3>(0,0);
    const Eigen::Vector3f t_rc = ref_inv.block<3,1>(0,3);

    // Relative transform: Live Cam -> World -> Ref Cam
    const Eigen::Matrix3f R_rel = R_rc * R_cw;
    const Eigen::Vector3f t_rel = R_rc * t_cw + t_rc;
    const Eigen::Matrix3f R_rel_T = R_rel.transpose();

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

        inline void add(const float* J, float err) {
            int k = 0;
            for (int i = 0; i < 6; ++i) {
                for (int j = i; j < 6; ++j) {
                    A_data[k++] += J[i] * J[j];
                }
                b_data[i] -= J[i] * err;
            }
            residual += err * err;
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
            if (model_v_world.norm() < 1e-6f || model_n_world.norm() < 1e-6f) continue;
            acc.valid_model++;

            // Transform live vertex to world space for distance check
            Eigen::Vector3f v_live_world = R_cw * live_v + t_cw;
            float dist_sq = (v_live_world - model_v_world).squaredNorm();
            if (dist_sq > params_.dist_threshold * params_.dist_threshold) {
                acc.dist_filtered++;
                continue;
            }

            const Eigen::Vector3f& live_n = live.normals[idx];
            if (live_n.z() == 0.0f) continue;
            
            // Map live normal to world space for angle check
            Eigen::Vector3f n_live_world = R_cw * live_n;
            float dot = std::abs(n_live_world.dot(model_n_world));
            if (dot < angle_thresh_cos) {
                acc.angle_filtered++;
                continue;
            }

            // Point-to-plane error in world space
            float err = model_n_world.dot(v_live_world - model_v_world);
            
            // Jacobian J = [n_live_cam; cross(live_v, n_live_cam)]
            // where n_live_cam = R_cw^T * model_n_world
            Eigen::Vector3f n_live_cam = R_cw.transpose() * model_n_world;
            Eigen::Vector3f cross = live_v.cross(n_live_cam);

            float J[6] = { n_live_cam.x(), n_live_cam.y(), n_live_cam.z(), cross.x(), cross.y(), cross.z() };

            if (std::isnan(J[0]) || std::isinf(J[0]) || std::isnan(J[3]) || std::isinf(J[3])) {
                continue;
            }

            // Huber weight for robustness
            float abs_err = std::abs(err);
            float huber_k = 0.02f; // Reduced from 0.05m to fix ghosting/drift
            float w = (abs_err <= huber_k) ? 1.0f : huber_k / abs_err;

            acc.add(J, err * w);
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
