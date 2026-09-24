#pragma once

// Render-and-compare verification of a relocalization candidate (relocalization
// rework, step 5).
//
// A good ICP fit does not make a pose right: on cap_001, 89 of 91
// relocalization solves that fitted Good were > 15 deg off gravity (wrong
// basins), and a flat wall fits a floor. The old relocalizer leaned on the
// per-frame motion gate (0.15 m / 30 deg from the stale pose) to reject them,
// which also refused every correct far recovery (spin360-slow after its loop
// closure). Instead, render the model at the candidate pose and compare depths
// pixel by pixel with the live frame:
//   consistent  |z_live - z_model| <= tau(z)      the same surface
//   violation   z_live > z_model + tau(z)         the live ray passes THROUGH a
//               (model surface facing the camera)   surface the model has: the
//                                                    pose puts that surface in
//                                                    the wrong place
//   in front    z_live < z_model - tau(z)         something new or an occluder:
//                                                    neutral
// tau(z) = max(3 cm, 3 sigma(z)), sigma(z) = 0.0012 + 0.0019 (z - 0.4)^2 m (the
// Kinect v1 axial noise model tests/support/SyntheticScene.h also uses): 3 cm up
// to ~2.5 m, 7.7 cm at 4 m.
// A candidate passes with enough overlap (coverage), mostly consistent, and few
// violations. Thresholds are starting values, calibrated on real takes.

#include "tracking/ICPShared.h"

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace kfusion {
namespace tracking {

inline float kinectSigma(float z) { return 0.0012f + 0.0019f * (z - 0.4f) * (z - 0.4f); }

inline float consistencyTau(float z, float floor_m = 0.03f) {
    return std::max(floor_m, 3.0f * kinectSigma(z));
}

struct ConsistencyThresholds {
    float min_coverage   = 0.30f;   // both valid / live valid
    float min_consistent = 0.60f;   // consistent / both valid
    float max_violation  = 0.10f;   // violation / both valid
    float tau_floor      = 0.03f;   // m
    float min_facing_cos = 0.26f;   // a violation needs a surface within 75 deg of facing the camera
};

struct DepthConsistency {
    int live_valid = 0;    // sampled pixels with live depth
    int both_valid = 0;    // ...that also have a model surface
    int consistent = 0;
    int violation  = 0;
    int in_front   = 0;

    float coverage() const {
        return live_valid > 0 ? static_cast<float>(both_valid) / static_cast<float>(live_valid) : 0.0f;
    }
    float consistentFraction() const {
        return both_valid > 0 ? static_cast<float>(consistent) / static_cast<float>(both_valid) : 0.0f;
    }
    float violationFraction() const {
        return both_valid > 0 ? static_cast<float>(violation) / static_cast<float>(both_valid) : 0.0f;
    }
    bool passes(const ConsistencyThresholds& t = ConsistencyThresholds{}) const {
        return coverage() >= t.min_coverage && consistentFraction() >= t.min_consistent &&
               violationFraction() <= t.max_violation;
    }
};

// `model_v` / `model_n`: world-space vertices / normals of a model image of
// mw x mh raycast at `pose` (world-from-camera). `live_depth`: the live frame's
// depth in meters (0 = none), lw x lh. Each model pixel samples the live pixel
// at the same place in the image (nearest), so the model image may be smaller.
inline DepthConsistency depthConsistency(const Eigen::Vector3f* model_v, const Eigen::Vector3f* model_n,
                                         int mw, int mh, const Eigen::Matrix4f& pose,
                                         const float* live_depth, int lw, int lh,
                                         const ConsistencyThresholds& t = ConsistencyThresholds{}) {
    DepthConsistency out;
    const Eigen::Matrix4f inv = pose.inverse();
    const Eigen::Matrix3f R = inv.block<3,3>(0,0);
    const Eigen::Vector3f tr = inv.block<3,1>(0,3);
    for (int v = 0; v < mh; ++v) {
        const int lv = std::min(lh - 1, static_cast<int>((v + 0.5f) * lh / mh));
        for (int u = 0; u < mw; ++u) {
            const int lu = std::min(lw - 1, static_cast<int>((u + 0.5f) * lw / mw));
            const float z_live = live_depth[static_cast<size_t>(lv) * lw + lu];
            if (!(z_live > 0.0f) || !std::isfinite(z_live)) continue;
            ++out.live_valid;
            const size_t i = static_cast<size_t>(v) * mw + u;
            if (!modelVertexIsValid(model_v[i]) || !normalIsValid(model_n[i])) continue;
            const Eigen::Vector3f p = R * model_v[i] + tr;
            if (!(p.z() > 0.0f)) continue;
            ++out.both_valid;
            const float tau = consistencyTau(p.z(), t.tau_floor);
            const float dz = z_live - p.z();
            if (std::fabs(dz) <= tau) {
                ++out.consistent;
            } else if (dz > 0.0f) {
                const Eigen::Vector3f n = R * model_n[i];
                if (std::fabs(n.dot(p.normalized())) >= t.min_facing_cos) ++out.violation;
            } else {
                ++out.in_front;
            }
        }
    }
    return out;
}

} // namespace tracking
} // namespace kfusion
