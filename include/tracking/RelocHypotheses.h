#pragma once

// Candidate poses for relocalization (relocalization rework, step 4).
//
// The old relocalizer only tried poses within ~14 deg of the last good one, so a
// camera that kept turning while lost (cap_001's sweeps, a kidnap) was never
// found again. With gravity known (tracking/Gravity.h) pitch and roll come from
// the accelerometer, and a person scanning a room mostly turns in place: what is
// unknown is yaw about world up and a little translation. So the candidates are:
//   - a few fixed poses (last good, the model image's pose, keyframes, ...), each
//     "snapped" to the measured gravity: the smallest camera rotation that makes
//     the up direction the pose predicts equal the measured one;
//   - a yaw sweep about world up through two pivots: the camera centre, and a
//     point 0.12 m behind it. A handheld spin orbits the body, not the lens:
//     fitted on spin360-slow's trajectory, horizontal chord vs yaw gives
//     r = 0.11-0.12 m at 12, 88, 117, 150 and 188 deg, so a 180 deg turn also
//     moves the camera ~0.24 m.
// Near-duplicates are dropped: every candidate costs a raycast and an ICP solve.

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace kfusion {
namespace tracking {

enum class HypothesisSource : uint8_t {
    LastGood,           // the last tracked pose, gravity-snapped
    LastGoodUnsnapped,  // ...as tracked (guards against a biased accelerometer)
    ModelPose,          // the pose the current model image was raycast at
    CarryOver,          // the best candidate of the previous lost frame
    Keyframe,           // a fern-retrieved keyframe pose (tracking/FernDatabase.h)
    Sweep,              // yaw about world up
    PitchGrid,          // the old +-10 deg camera-axis grid (no gravity reading)
};

struct Hypothesis {
    Eigen::Matrix4f  pose = Eigen::Matrix4f::Identity();
    HypothesisSource source = HypothesisSource::LastGood;
};

struct PoseGap {
    float trans_m = 0.0f;
    float rot_deg = 0.0f;
};

inline PoseGap poseGap(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b) {
    const Eigen::Matrix3f dR = a.block<3,3>(0,0).transpose() * b.block<3,3>(0,0);
    const float c = std::max(-1.0f, std::min(1.0f, (dR.trace() - 1.0f) * 0.5f));
    return {(a.block<3,1>(0,3) - b.block<3,1>(0,3)).norm(), std::acos(c) * 57.2957795f};
}

// Rotate the camera by the smallest angle so that the pose's predicted up
// (R^T up_world) equals the measured up `up_cam` (both any length). The
// translation is kept. If the two are opposite, the turn is about the camera x
// axis (made orthogonal to the measured up), so the result is deterministic.
inline Eigen::Matrix4f snapToGravity(const Eigen::Matrix4f& pose, const Eigen::Vector3f& up_world,
                                     const Eigen::Vector3f& up_cam) {
    const Eigen::Matrix3f R = pose.block<3,3>(0,0);
    const Eigen::Vector3f p = (R.transpose() * up_world).normalized();
    const Eigen::Vector3f m = up_cam.normalized();
    // C with C * m = p gives (R C)^T up_world = C^T p = m.
    const Eigen::Vector3f axis = m.cross(p);
    const float s = axis.norm();
    const float c = m.dot(p);
    Eigen::Matrix3f C = Eigen::Matrix3f::Identity();
    if (s > 1e-7f) {
        C = Eigen::AngleAxisf(std::atan2(s, c), axis / s).toRotationMatrix();
    } else if (c < 0.0f) {
        Eigen::Vector3f ax = Eigen::Vector3f::UnitX() - m * m.x();
        if (ax.norm() < 1e-3f) ax = Eigen::Vector3f::UnitZ() - m * m.z();
        C = Eigen::AngleAxisf(static_cast<float>(M_PI), ax.normalized()).toRotationMatrix();
    }
    Eigen::Matrix4f out = pose;
    out.block<3,3>(0,0) = R * C;
    return out;
}

// Turn the camera by `yaw_rad` about the world-up axis through `pivot` (world
// point). Tilt against gravity is unchanged.
inline Eigen::Matrix4f yawAboutUp(const Eigen::Matrix4f& pose, float yaw_rad,
                                  const Eigen::Vector3f& up_world, const Eigen::Vector3f& pivot) {
    const Eigen::Matrix3f Ru = Eigen::AngleAxisf(yaw_rad, up_world.normalized()).toRotationMatrix();
    Eigen::Matrix4f out = pose;
    out.block<3,3>(0,0) = Ru * pose.block<3,3>(0,0);
    out.block<3,1>(0,3) = pivot + Ru * (pose.block<3,1>(0,3) - pivot);
    return out;
}

// The point `radius` behind the camera along its horizontal forward direction
// (the body a handheld spin turns about). Looking straight up or down there is
// no horizontal forward; the camera's own y axis stands in for it.
inline Eigen::Vector3f bodyPivot(const Eigen::Matrix4f& pose, const Eigen::Vector3f& up_world,
                                 float radius) {
    const Eigen::Vector3f u = up_world.normalized();
    Eigen::Vector3f f = pose.block<3,1>(0,2);
    f -= u * f.dot(u);
    if (f.norm() < 0.1f) {
        f = pose.block<3,1>(0,1);
        f -= u * f.dot(u);
    }
    return pose.block<3,1>(0,3) - radius * f.normalized();
}

struct SweepParams {
    float step_deg     = 20.0f;   // worst-case miss 10 deg: inside the coarse basin
    float body_radius  = 0.12f;   // m, measured on spin360-slow (see above)
};

// Yaw sweep around `base`: +-step, +-2 step, ... up to 180 deg, each at the
// camera pivot then the body pivot, ordered by |yaw| (small turns first).
// With the defaults: 8 magnitudes x 2 signs x 2 pivots + 180 x 2 = 34.
inline std::vector<Hypothesis> buildSweep(const Eigen::Matrix4f& base,
                                          const Eigen::Vector3f& up_world,
                                          const SweepParams& sp = SweepParams{}) {
    std::vector<Hypothesis> out;
    const Eigen::Vector3f cam = base.block<3,1>(0,3);
    const Eigen::Vector3f body = bodyPivot(base, up_world, sp.body_radius);
    const int n = static_cast<int>(std::floor(180.0f / sp.step_deg + 1e-3f));
    for (int k = 1; k <= n; ++k) {
        const float deg = std::min(180.0f, k * sp.step_deg);
        const bool half_turn = deg >= 180.0f - 1e-3f;
        for (const Eigen::Vector3f* pivot : {&cam, &body}) {
            for (float sign : {1.0f, -1.0f}) {
                out.push_back({yawAboutUp(base, sign * deg * static_cast<float>(M_PI) / 180.0f,
                                          up_world, *pivot),
                               HypothesisSource::Sweep});
                if (half_turn) break;   // +180 == -180
            }
        }
    }
    return out;
}

// The pre-rework hypothesis grid: +-10 deg yaw/pitch about the camera's own
// axes. Used when there is no usable gravity reading to sweep about.
inline std::vector<Hypothesis> pitchGrid(const Eigen::Matrix4f& base, float step_deg = 10.0f) {
    std::vector<Hypothesis> out;
    const float s = step_deg * static_cast<float>(M_PI) / 180.0f;
    for (int yi = -1; yi <= 1; ++yi) {
        for (int pi = -1; pi <= 1; ++pi) {
            if (yi == 0 && pi == 0) continue;
            Eigen::Matrix4f d = Eigen::Matrix4f::Identity();
            d.block<3,3>(0,0) = (Eigen::AngleAxisf(yi * s, Eigen::Vector3f::UnitY()) *
                                 Eigen::AngleAxisf(pi * s, Eigen::Vector3f::UnitX()))
                                    .toRotationMatrix();
            out.push_back({base * d, HypothesisSource::PitchGrid});
        }
    }
    return out;
}

// Keep the first of any candidates closer than trans_m AND rot_deg; order kept.
inline std::vector<Hypothesis> dedupeHypotheses(const std::vector<Hypothesis>& in,
                                                float trans_m = 0.03f, float rot_deg = 4.0f) {
    std::vector<Hypothesis> out;
    out.reserve(in.size());
    for (const Hypothesis& h : in) {
        if (!h.pose.allFinite()) continue;
        const bool dup = std::any_of(out.begin(), out.end(), [&](const Hypothesis& k) {
            const PoseGap g = poseGap(k.pose, h.pose);
            return g.trans_m < trans_m && g.rot_deg < rot_deg;
        });
        if (!dup) out.push_back(h);
    }
    return out;
}

} // namespace tracking
} // namespace kfusion
