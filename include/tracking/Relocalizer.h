#pragma once

// Relocalization: find the camera again after tracking is lost (relocalization
// rework, step 7; big-fix-two T4.5).
//
// The old relocalizer ran ICP from 11 poses within ~14 deg of the last good one,
// all against ONE model image raycast before the loss, and then required the
// result to lie within 0.15 m / 30 deg of that stale pose. So it only came back
// if the camera returned to roughly where it was lost AND looked at what that
// image showed; and on spin360-slow, after the loop-closure loss, it refused
// every correct Good fit (0.44-0.47 m from the drifted pose).
//
// This one, per lost frame:
//   Stage A  a few fixed candidates (tracking/RelocHypotheses.h): the last good
//            pose, the previous lost frame's best, the model image's pose, and
//            the nearest fern keyframes (tracking/FernDatabase.h), each snapped to
//            the measured gravity.
//   Stage B  (only if A found nothing) the next batch of a yaw sweep about world
//            up through the camera and a point 0.12 m behind it. A cursor walks
//            the sweep across frames, so a CPU can spread it out.
// Every candidate gets its OWN model image: the TSDF is raycast at that pose
// (160x120) and a coarse ICP runs from it. The best few are refined (320x240, all
// pyramid levels) and verified:
//   - the fit grades Good (tracking/TrackingPolicy.h classifyFit: no motion gate);
//   - gravity agrees (tracking/Gravity.h);
//   - every direction is constrained (the weakest eigenvalue of the ICP
//     information matrix is not ~0: a single wall is a guess);
//   - render-and-compare: live depth agrees with the model rendered at the pose
//     (tracking/DepthConsistency.h);
//   - not a basin whose probation just failed;
//   - no second, distant pose explains the frame as well (a symmetric room:
//     wait instead of guessing). A pose near the last good one (within 0.15 m /
//     30 deg) is taken at once; a far one only after the WHOLE sweep has been
//     tried on the same frame. On the nearly square synthetic room a 180 deg
//     kidnap onto a bare corner verified 90 deg off at 99.4% consistency, and
//     the true pose was still unevaluated in a later CPU sweep batch.
// The backend (CPU or GPU raycast and ICP) comes in through two hooks, so this
// file has no idea which one runs.

#include "tracking/DepthConsistency.h"
#include "tracking/Gravity.h"
#include "tracking/ICPTracker.h"
#include "tracking/RelocHypotheses.h"

#include <Eigen/Core>

#include <cstdint>
#include <functional>
#include <vector>

namespace kfusion {
namespace tracking {

enum class RenderSize : uint8_t { Coarse, Refine, Full };   // 160x120, 320x240, 640x480

inline int renderWidth(RenderSize s) { return s == RenderSize::Coarse ? 160 : s == RenderSize::Refine ? 320 : 640; }
inline int renderHeight(RenderSize s) { return s == RenderSize::Coarse ? 120 : s == RenderSize::Refine ? 240 : 480; }

enum class RelocReject : uint8_t {
    None,
    NoDepth,       // too little live depth to try
    Unsteady,      // a reading exists but the hand is accelerating: gravity cannot vouch for any pose
    EmptyModel,    // nothing fused yet
    NoCandidate,   // no candidate fitted at all
    Fit,           // best refined fit not Good
    Gravity,
    Constraint,    // a direction is unobserved (e.g. one wall)
    Consistency,   // render-and-compare failed
    Ambiguous,     // two distant poses explain the frame equally well
    Blacklisted,   // a basin whose probation failed recently
};

const char* relocRejectName(RelocReject r);
const char* hypothesisSourceName(HypothesisSource s);

struct RelocParams {
    // Coarse scoring (pyramid level 2 only, against a 160x120 render).
    float coarse_dist_scale     = 2.5f;
    float coarse_angle_deg      = 45.0f;
    int   coarse_min_iterations = 8;
    int   coarse_max_iterations = 30;
    float coarse_min_fit        = 0.30f;   // inliers / valid model correspondences
    float merge_trans_m         = 0.03f;   // coarse results this close are one basin
    float merge_rot_deg         = 3.0f;
    // Budget per lost frame, in candidates (counted, not timed: deterministic).
    int   sweep_budget          = 34;      // the whole sweep (GPU); 6 on CPU
    int   refine_top            = 3;       // 2 on CPU
    bool  use_sweep             = true;
    SweepParams sweep;
    float keep_unsnapped_deg    = 4.0f;    // also try last_good as tracked when snapping moved it more
    // Verification.
    float min_eig_ratio         = 1e-3f;
    ConsistencyThresholds consistency;
    float ambiguous_trans_m     = 0.2f;
    float ambiguous_rot_deg     = 15.0f;
    float ambiguous_gap         = 0.05f;   // consistent fractions this close are a tie
    float local_trans_m         = 0.15f;   // accepted without the full-sweep ambiguity check
    float local_rot_deg         = 30.0f;
    float min_live_valid_share  = 0.10f;
    int   blacklist_frames      = 30;
    float blacklist_trans_m     = 0.05f;
    float blacklist_rot_deg     = 5.0f;
};

struct RelocBackend {
    // The model raycast at `pose` into a buffer of that size (host arrays filled
    // when `need_host`, or always on the CPU). The reference stays valid until
    // the next render of the same size.
    std::function<const ModelFrame&(const Eigen::Matrix4f& pose, RenderSize size, bool need_host)> render;
    // ICP of the live frame against `model` from `estimate`, with `params`.
    std::function<ICPResult(const ModelFrame& model, const Eigen::Matrix4f& estimate, const ICPParams& params)> solve;
};

struct RelocRequest {
    Eigen::Matrix4f last_good  = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f model_pose = Eigen::Matrix4f::Identity();
    std::vector<Eigen::Matrix4f> keyframes;   // nearest first
    GravityContext  gravity;
    const float*    live_depth = nullptr;     // meters, 0 = none
    int             live_w = 0, live_h = 0;
    ICPParams       icp;                      // the normal tracking parameters
    bool            model_empty = false;
};

struct RelocOutcome {
    bool             accepted = false;
    ICPResult        result;                  // the accepted solve (else the best refined, or last_good)
    DepthConsistency consistency;
    HypothesisSource source = HypothesisSource::LastGood;
    RelocReject      reject = RelocReject::None;
    int              candidates = 0, coarse_solves = 0, refines = 0;
};

class Relocalizer {
public:
    explicit Relocalizer(const RelocParams& p = RelocParams{}) : p_(p) {}
    const RelocParams& params() const { return p_; }
    void setParams(const RelocParams& p) { p_ = p; }

    // A new loss episode: the sweep starts over and nothing carries over.
    void beginLoss();
    // Probation failed at `pose`: refuse that basin for blacklist_frames runs.
    void rejectBasin(const Eigen::Matrix4f& pose);

    RelocOutcome run(const RelocRequest& req, const RelocBackend& backend);

private:
    struct Scored {
        ICPResult        r;
        HypothesisSource source;
        bool             gradable;
    };
    struct Verified {
        ICPResult        r;
        DepthConsistency c;
        HypothesisSource source;
    };
    struct Basin {
        Eigen::Matrix4f pose;
        int             runs_left;
    };

    bool blacklisted(const Eigen::Matrix4f& pose) const;
    bool isLocal(const Eigen::Matrix4f& pose, const Eigen::Matrix4f& last_good) const;
    // Coarse-score `cands`, refine the best basins, verify them. Returns the
    // verified ones; tracks the best basin's reject reason and the carry-over.
    std::vector<Verified> evaluate(const std::vector<Hypothesis>& cands, const RelocRequest& req,
                                   const RelocBackend& be, RelocOutcome& out);
    bool decide(std::vector<Verified>& all, RelocOutcome& out) const;

    RelocParams        p_;
    size_t             cursor_ = 0;
    bool               have_carry_ = false;
    Eigen::Matrix4f    carry_ = Eigen::Matrix4f::Identity();
    std::vector<Basin> blacklist_;
    // Per run(): the reject reason of the best-supported basin seen.
    int                best_basin_inliers_ = -1;
    RelocReject        best_basin_reject_ = RelocReject::None;
    bool               have_best_refined_ = false;
    ICPResult          best_refined_;
};

} // namespace tracking
} // namespace kfusion
