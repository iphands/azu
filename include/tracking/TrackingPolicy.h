#pragma once

// Frame-level tracking decision (big-fix-two T0.12). ICPResult::tracking_ok is
// the SOLVER's verdict and requires convergence (or a tiny final step); a solve
// that ran out of iterations with a near-perfect fit then counted as a tracking
// failure, integration stopped, and one bad frame put the pipeline into
// relocalization. The pipeline instead grades each result:
//
//   Good   : finite, enough inliers, inlier ratio and RMS within bounds, motion
//            inside the per-frame gate -> update the pose AND integrate
//   Poor   : finite and inside the motion gate, but the fit is weak -> update
//            the pose, do not integrate (the model stays clean)
//   Failed : anything else -> keep the previous pose; only
//            `failures_before_lost` consecutive failures enter relocalization
//
// Convergence is informational only.

#include "tracking/ICPTracker.h"
#include "tracking/ICPShared.h"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>

namespace kfusion {
namespace tracking {

enum class TrackQuality { Good, Poor, Failed };

struct TrackingPolicy {
    int   min_inliers            = kMinInliersForOk;
    float min_inlier_ratio       = 0.30f;   // inliers / valid live points
    float max_rms_m              = 0.015f;  // ~3 sigma of Kinect noise at ~2.5 m
    float max_frame_translation  = 0.15f;   // m between consecutive frames
    float max_frame_rotation     = 0.52f;   // rad (~30 deg)
    int   failures_before_lost   = 3;
};

// RMS point-to-plane residual implied by ICPResult::error (the mean Huber loss:
// e^2 inside the Huber band, so sqrt is the RMS for inlier-dominated fits).
inline float icpRmsMeters(const ICPResult& r) {
    return std::sqrt(std::max(0.0f, r.error));
}

inline float icpInlierRatio(const ICPResult& r) {
    return r.valid_live_points > 0
               ? static_cast<float>(r.inliers) / static_cast<float>(r.valid_live_points)
               : 0.0f;
}

inline TrackQuality classifyTracking(const ICPResult& r, const Eigen::Matrix4f& prev_pose,
                                     const TrackingPolicy& p = TrackingPolicy{}) {
    if (!r.pose.allFinite() || !std::isfinite(r.error) || r.inliers < p.min_inliers) {
        return TrackQuality::Failed;
    }
    const Eigen::Matrix4f d = prev_pose.inverse() * r.pose;
    const float trans = d.block<3,1>(0,3).norm();
    const float c = std::max(-1.0f, std::min(1.0f, (d.block<3,3>(0,0).trace() - 1.0f) * 0.5f));
    const float angle = std::acos(c);
    if (!(trans <= p.max_frame_translation) || !(angle <= p.max_frame_rotation)) {
        return TrackQuality::Failed;
    }
    if (icpInlierRatio(r) >= p.min_inlier_ratio && icpRmsMeters(r) <= p.max_rms_m) {
        return TrackQuality::Good;
    }
    return TrackQuality::Poor;
}

} // namespace tracking
} // namespace kfusion
