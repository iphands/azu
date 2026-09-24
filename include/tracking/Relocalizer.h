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
    Unvisited,     // farther than max_visited_distance_m from every position tracked before
    TooFast,       // turned farther from the last good pose than a hand can in the time lost
    Unconverged,   // the refine was still moving the pose when its rounds ran out
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
    // The sweep steps 20 deg, so a candidate can miss by 10 deg: 0.44-0.52 m of
    // point motion at 2.5-3 m. With the pipeline's default 0.1 m ICP distance
    // (x2.5 = 0.25 m) a 12 deg miss gave 0 inliers; 0.5 m converged to 0.2 mm
    // (synthetic room, 160x120 and 640x480 renders alike). The coarse solve only
    // has to find the basin; verification rejects wrong ones.
    float coarse_min_dist_m     = 0.5f;
    float coarse_angle_deg      = 45.0f;
    int   coarse_min_iterations = 8;
    int   coarse_max_iterations = 30;
    float coarse_min_fit        = 0.30f;   // inliers / valid model correspondences
    float merge_trans_m         = 0.03f;   // coarse results this close are one basin
    float merge_rot_deg         = 3.0f;
    // Budget per lost frame, in candidates (counted, not timed: deterministic).
    int   sweep_budget          = 34;      // the whole sweep (GPU); 6 on CPU
    int   refine_top            = 3;       // 2 on CPU
    // A refine that fits Good re-renders at its result and solves again until
    // a round moves the pose less than refine_converged_* (at most
    // refine_rounds solves).
    // One solve against an image rendered at the coarse pose converges only
    // partway: a cap_001 kidnap was accepted 141 mm / 9.4 deg off, and the
    // next ~10 tracked frames pulled it in while integrating.
    int   refine_rounds         = 4;
    float refine_converged_m    = 0.005f;
    float refine_converged_deg  = 0.25f;
    // If the last round still moved it more than this, the pose is not taken
    // on this frame; the carry-over candidate resumes from it on the next.
    // It fires on 3-29 relocalizing frames per cap_001 run. It does NOT catch
    // the cap_001 kidnap offsets it was aimed at (165-188 mm / 11-12 deg in 2
    // of 10 runs): those refines had converged onto the offset, along a
    // weakly constrained direction, and later views pulled them in.
    float refine_settled_m      = 0.02f;
    float refine_settled_deg    = 1.0f;
    bool  use_sweep             = true;
    SweepParams sweep;
    float keep_unsnapped_deg    = 4.0f;    // also try last_good as tracked when snapping moved it more
    // Verification.
    // Weakest/strongest information eigenvalue of the refined solve: refuses
    // truly degenerate views (a single bare wall reads 8e-6..4e-4). It does NOT
    // separate right from wrong re-acquisitions on real data: on cap_001 wrong
    // ones (closet doors, bare ceiling corners) read 1.3e-3..2.3e-3, and so did
    // spin360-slow's correct loop-closure recovery (2.1e-3..2.5e-3). See
    // max_visited_distance_m for what does.
    float min_eig_ratio         = 1e-3f;
    // A re-acquired pose must be one the camera has effectively been at before:
    // within this distance AND this angle (between the viewing directions) of
    // some pose tracked Good earlier. A view is only matched reliably from near
    // where that part of the room was seen. cap_001, 22 re-acquisitions over
    // three runs, each labelled against the RGB of the most similar earlier
    // well-tracked pose: every right one within 4-15 cm / 3-27 deg of it; every
    // wrong one 26-142 cm away, except one chained off an earlier wrong accept.
    // A position-only 0.3 m radius let wrong ones through: in an in-place spin
    // every position is near some visited one. spin360-slow's loop-closure
    // recovery lands back on its start. 0 turns the check off; so does an empty
    // list.
    float max_visited_distance_m = 0.2f;
    float max_visited_angle_deg  = 45.0f;
    // A re-acquired pose may be at most max_turn_rate_deg_s * (time since the
    // last good pose) + turn_slack_deg of rotation from it: a hand does not
    // turn a camera 104 deg in 0.13 s, which a cap_001 keyframe candidate
    // claimed (Good fit, consistency 0.94, and integrated 3 frames). Over the
    // real re-acquisitions judged right (cap_001, spin360-slow) the fastest
    // was 95 deg after 26 lost frames. Rotation only: spin360-slow's correct
    // loop-closure recovery moves 0.4 m (drift) with 2.5-4.6 deg. Unknown
    // time (RelocRequest::seconds_since_good < 0) or 0 turns it off.
    float max_turn_rate_deg_s   = 180.0f;
    float turn_slack_deg        = 30.0f;
    ConsistencyThresholds consistency;
    float ambiguous_trans_m     = 0.2f;
    float ambiguous_rot_deg     = 15.0f;
    float ambiguous_gap         = 0.05f;   // consistent fractions this close are a tie
    // Coarse-stage ambiguity (off by default): a distant basin that fitted
    // nearly as well in the coarse solve (fit ratio within the gap, at least the
    // inlier share) makes the frame ambiguous even if never refined. It caught
    // some wrong cap_001 re-acquisitions, but it also refused spin360-slow's
    // correct loop-closure recovery on 127 frames in a row: the drifted model
    // holds two copies of the start, 0.44 m apart. max_visited_distance_m
    // catches those wrong ones without that cost.
    bool  coarse_ambiguity      = false;
    float coarse_ambiguous_fit_gap    = 0.05f;
    float coarse_ambiguous_inlier_share = 0.5f;
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
    const std::vector<Eigen::Matrix4f>* visited = nullptr;   // poses tracked Good (world-from-camera)
    float           seconds_since_good = -1.0f;   // since last_good was tracked; < 0 = unknown
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
    int              candidates = 0, coarse_solves = 0, refines = 0;   // refines: refine solves
    float            eig_ratio = 0.0f;       // weakestDirectionRatio of `result`
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
        float            coarse_fit = 0.0f;   // the coarse basin it was refined from
        int              coarse_inliers = 0;
    };
    struct CoarseBasin {
        Eigen::Matrix4f pose;
        float           fit;
        int             inliers;
    };
    struct Basin {
        Eigen::Matrix4f pose;
        int             runs_left;
    };

    bool blacklisted(const Eigen::Matrix4f& pose) const;
    bool nearVisited(const Eigen::Matrix4f& pose, const RelocRequest& req) const;
    bool plausibleTurn(const Eigen::Matrix4f& pose, const RelocRequest& req) const;
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
    std::vector<CoarseBasin> run_basins_;   // every gradable coarse basin of this run
};

} // namespace tracking
} // namespace kfusion
