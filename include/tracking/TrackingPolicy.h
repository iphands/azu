#pragma once

// Frame-level tracking decision (big-fix-two T0.12). ICPResult::tracking_ok is
// the SOLVER's verdict and requires convergence (or a tiny final step); a solve
// that ran out of iterations with a near-perfect fit then counted as a tracking
// failure, integration stopped, and one bad frame put the pipeline into
// relocalization. The pipeline instead grades each result:
//
//   Good   : finite, enough inliers, a tight fit on the points that landed on
//            the model, motion inside the per-frame gate -> update the pose AND
//            integrate
//   Poor   : finite and inside the motion gate, but the fit is weak -> update
//            the pose, do not integrate (the model stays clean)
//   Failed : anything else -> keep the previous pose; only
//            `failures_before_lost` consecutive failures enter relocalization
//
// Convergence is informational only.
//
// v2: the fit ratio is inliers / valid_model -- over live points that landed on
// a valid model pixel -- not inliers / valid_live. Live points outside the old
// model image, outside the volume or on unobserved space say nothing about the
// fit; counting them made every frame that turns toward new geometry grade Poor,
// Poor frames are not integrated, so the model never grew and a slow 15 degree
// turn ended in Lost (the field report that motivated v2).

#include "tracking/ICPTracker.h"
#include "tracking/ICPShared.h"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>

namespace kfusion {
namespace tracking {

enum class TrackQuality { Good, Poor, Failed };

struct TrackingPolicy {
    int   min_inliers            = kMinInliersForOk;  // below this: Failed
    int   min_good_inliers       = 2000;    // Good needs this much support
    float min_fit_ratio          = 0.40f;   // inliers / valid model correspondences (real data: 0.57-0.8)
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

// Share of live points that landed on a valid model pixel and survived the
// distance / angle / normal gates.
inline float icpFitRatio(const ICPResult& r) {
    return r.valid_model_points > 0
               ? static_cast<float>(r.inliers) / static_cast<float>(r.valid_model_points)
               : 0.0f;
}

// The fit alone: Failed when the solve is unusable, Good when it is tight and
// well supported, Poor otherwise. No motion gate: a relocalization result may
// legitimately land far from the last tracked pose (spin360-slow after its loop
// closure: Good fits 0.44-0.47 m from the drifted pose, all refused by the
// per-frame gate), so it is verified another way (tracking/Relocalizer.h).
inline TrackQuality classifyFit(const ICPResult& r, const TrackingPolicy& p = TrackingPolicy{}) {
    if (!r.pose.allFinite() || !std::isfinite(r.error) || r.inliers < p.min_inliers) {
        return TrackQuality::Failed;
    }
    if (r.inliers >= p.min_good_inliers && icpFitRatio(r) >= p.min_fit_ratio &&
        icpRmsMeters(r) <= p.max_rms_m) {
        return TrackQuality::Good;
    }
    return TrackQuality::Poor;
}

// The per-frame motion gate: at most max_frame_translation / max_frame_rotation
// from `prev_pose`.
inline bool withinFrameMotion(const Eigen::Matrix4f& pose, const Eigen::Matrix4f& prev_pose,
                              const TrackingPolicy& p = TrackingPolicy{}) {
    const Eigen::Matrix4f d = prev_pose.inverse() * pose;
    const float trans = d.block<3,1>(0,3).norm();
    const float c = std::max(-1.0f, std::min(1.0f, (d.block<3,3>(0,0).trace() - 1.0f) * 0.5f));
    const float angle = std::acos(c);
    return trans <= p.max_frame_translation && angle <= p.max_frame_rotation;
}

// Frame-to-frame tracking: the fit, and the per-frame motion gate.
inline TrackQuality classifyTracking(const ICPResult& r, const Eigen::Matrix4f& prev_pose,
                                     const TrackingPolicy& p = TrackingPolicy{}) {
    const TrackQuality fit = classifyFit(r, p);
    if (fit == TrackQuality::Failed) return fit;
    if (!withinFrameMotion(r.pose, prev_pose, p)) return TrackQuality::Failed;
    return fit;
}

// Smallest over largest eigenvalue of the ICP information matrix: how well the
// weakest direction is constrained (0 when a direction is not observed at all,
// e.g. sliding along a single wall). 0 for an empty or non-finite matrix.
inline float weakestDirectionRatio(const Eigen::Matrix<float, 6, 6>& information) {
    if (!information.allFinite()) return 0.0f;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> es(information);
    if (es.info() != Eigen::Success) return 0.0f;
    const float max_ev = es.eigenvalues().maxCoeff();
    if (!(max_ev > 0.0f)) return 0.0f;
    return std::max(0.0f, es.eigenvalues().minCoeff()) / max_ev;
}

// Degeneracy-aware motion (big-fix-two T3.7). Facing a single wall, ICP cannot
// observe sliding along it or rolling about its normal; whatever the solve
// reports there is the initial guess plus noise, and the constant-velocity
// prediction then carries that noise forward frame after frame (measured: a
// synthetic 360 degree spin drifted 0.26 m / 4.8 deg). Keep only the part of the
// frame-to-frame motion that the information matrix observes: directions whose
// eigenvalue is below `rel_threshold` times the largest keep the previous pose.
struct ObservableMotion {
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    int degenerate_dofs = 0;
};

inline ObservableMotion keepObservableMotion(const Eigen::Matrix4f& prev,
                                             const Eigen::Matrix4f& est,
                                             const Eigen::Matrix<float, 6, 6>& information,
                                             float rel_threshold = 5e-3f) {
    ObservableMotion out;
    out.pose = est;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> es(information);
    if (es.info() != Eigen::Success) return out;
    const Eigen::Matrix<float, 6, 1> ev = es.eigenvalues();
    const float max_ev = ev.maxCoeff();
    if (!(max_ev > 0.0f) || !std::isfinite(max_ev)) return out;

    // Frame-to-frame increment as a twist [t; omega] in the camera frame.
    const Eigen::Matrix4f d = prev.inverse() * est;
    const Eigen::AngleAxisf aa(Eigen::Matrix3f(d.block<3,3>(0,0)));
    Eigen::Matrix<float, 6, 1> xi;
    xi.head<3>() = d.block<3,1>(0,3);
    xi.tail<3>() = aa.axis() * aa.angle();

    Eigen::Matrix<float, 6, 1> kept = Eigen::Matrix<float, 6, 1>::Zero();
    for (int i = 0; i < 6; ++i) {
        const Eigen::Matrix<float, 6, 1> v = es.eigenvectors().col(i);
        if (ev[i] >= rel_threshold * max_ev) {
            kept += v * v.dot(xi);
        } else {
            ++out.degenerate_dofs;
        }
    }
    if (out.degenerate_dofs == 0) return out;

    Eigen::Matrix4f step = Eigen::Matrix4f::Identity();
    const float angle = kept.tail<3>().norm();
    if (angle > 1e-9f) {
        step.block<3,3>(0,0) = Eigen::AngleAxisf(angle, kept.tail<3>() / angle).toRotationMatrix();
    }
    step.block<3,1>(0,3) = kept.head<3>();
    out.pose = prev * step;
    return out;
}

} // namespace tracking
} // namespace kfusion
