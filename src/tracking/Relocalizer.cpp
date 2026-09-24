#include "tracking/Relocalizer.h"

#include "tracking/TrackingPolicy.h"

#include <algorithm>
#include <cmath>

namespace kfusion {
namespace tracking {

const char* relocRejectName(RelocReject r) {
    switch (r) {
        case RelocReject::None:        return "none";
        case RelocReject::NoDepth:     return "no_depth";
        case RelocReject::Unsteady:    return "unsteady";
        case RelocReject::Unvisited:   return "unvisited";
        case RelocReject::TooFast:     return "too_fast";
        case RelocReject::Unconverged: return "unconverged";
        case RelocReject::EmptyModel:  return "empty_model";
        case RelocReject::NoCandidate: return "no_candidate";
        case RelocReject::Fit:         return "fit";
        case RelocReject::Gravity:     return "gravity";
        case RelocReject::Constraint:  return "constraint";
        case RelocReject::Consistency: return "consistency";
        case RelocReject::Ambiguous:   return "ambiguous";
        case RelocReject::Blacklisted: return "blacklisted";
    }
    return "?";
}

const char* hypothesisSourceName(HypothesisSource s) {
    switch (s) {
        case HypothesisSource::LastGood:          return "last_good";
        case HypothesisSource::LastGoodUnsnapped: return "last_good_unsnapped";
        case HypothesisSource::ModelPose:         return "model_pose";
        case HypothesisSource::CarryOver:         return "carry_over";
        case HypothesisSource::Keyframe:          return "keyframe";
        case HypothesisSource::Sweep:             return "sweep";
        case HypothesisSource::PitchGrid:         return "pitch_grid";
    }
    return "?";
}

void Relocalizer::beginLoss() {
    cursor_ = 0;
    have_carry_ = false;
}

void Relocalizer::rejectBasin(const Eigen::Matrix4f& pose) {
    blacklist_.push_back({pose, p_.blacklist_frames});
}

bool Relocalizer::blacklisted(const Eigen::Matrix4f& pose) const {
    return std::any_of(blacklist_.begin(), blacklist_.end(), [&](const Basin& b) {
        const PoseGap g = poseGap(b.pose, pose);
        return g.trans_m < p_.blacklist_trans_m && g.rot_deg < p_.blacklist_rot_deg;
    });
}

bool Relocalizer::nearVisited(const Eigen::Matrix4f& pose, const RelocRequest& req) const {
    if (p_.max_visited_distance_m <= 0.0f || !req.visited || req.visited->empty()) return true;
    const Eigen::Vector3f c = pose.block<3,1>(0,3);
    const Eigen::Vector3f f = pose.block<3,1>(0,2);
    const float r2 = p_.max_visited_distance_m * p_.max_visited_distance_m;
    const float min_cos = std::cos(p_.max_visited_angle_deg * 3.14159265f / 180.0f);
    for (const Eigen::Matrix4f& v : *req.visited) {
        if ((v.block<3,1>(0,3) - c).squaredNorm() <= r2 && v.block<3,1>(0,2).dot(f) >= min_cos) return true;
    }
    return false;
}

bool Relocalizer::plausibleTurn(const Eigen::Matrix4f& pose, const RelocRequest& req) const {
    if (p_.max_turn_rate_deg_s <= 0.0f || req.seconds_since_good < 0.0f) return true;
    return poseGap(pose, req.last_good).rot_deg <= p_.max_turn_rate_deg_s * req.seconds_since_good + p_.turn_slack_deg;
}

bool Relocalizer::isLocal(const Eigen::Matrix4f& pose, const Eigen::Matrix4f& last_good) const {
    const PoseGap g = poseGap(pose, last_good);
    return g.trans_m <= p_.local_trans_m && g.rot_deg <= p_.local_rot_deg;
}

RelocOutcome Relocalizer::run(const RelocRequest& req, const RelocBackend& be) {
    RelocOutcome out;
    out.result.pose = req.last_good;
    best_basin_inliers_ = -1;
    best_basin_reject_ = RelocReject::None;
    have_best_refined_ = false;
    run_basins_.clear();

    for (Basin& b : blacklist_) --b.runs_left;
    blacklist_.erase(std::remove_if(blacklist_.begin(), blacklist_.end(),
                                    [](const Basin& b) { return b.runs_left <= 0; }),
                     blacklist_.end());

    // Nothing to try against: not a solve wasted.
    if (req.model_empty) { out.reject = RelocReject::EmptyModel; return out; }
    int live_valid = 0;
    const int live_n = req.live_w * req.live_h;
    if (req.live_depth) {
        for (int i = 0; i < live_n; i += 4) live_valid += req.live_depth[i] > 0.0f;
    }
    if (live_n <= 0 || live_valid * 4 < p_.min_live_valid_share * live_n) {
        out.reject = RelocReject::NoDepth;
        return out;
    }
    // An Unsteady reading can never let a re-acquisition through
    // (GravityContext::allowsReacquire), so every solve would be wasted.
    if (req.gravity.blocksEveryPose()) { out.reject = RelocReject::Unsteady; return out; }

    const GravityContext& g = req.gravity;
    const Eigen::Vector3f up = g.have_ref ? g.up_world : Eigen::Vector3f(0.0f, -1.0f, 0.0f);
    auto snap = [&](const Eigen::Matrix4f& pose) {
        return g.canSnap() ? snapToGravity(pose, g.up_world, g.up_cam) : pose;
    };

    // Stage A: the fixed candidates.
    std::vector<Hypothesis> fixed;
    const Eigen::Matrix4f base = snap(req.last_good);
    fixed.push_back({base, HypothesisSource::LastGood});
    if (poseGap(base, req.last_good).rot_deg > p_.keep_unsnapped_deg) {
        fixed.push_back({req.last_good, HypothesisSource::LastGoodUnsnapped});
    }
    if (have_carry_) fixed.push_back({snap(carry_), HypothesisSource::CarryOver});
    fixed.push_back({snap(req.model_pose), HypothesisSource::ModelPose});
    for (const Eigen::Matrix4f& kf : req.keyframes) fixed.push_back({snap(kf), HypothesisSource::Keyframe});
    fixed = dedupeHypotheses(fixed);
    std::vector<Verified> verified = evaluate(fixed, req, be, out);

    auto bestIsLocal = [&]() {
        const auto it = std::max_element(verified.begin(), verified.end(), [](const Verified& a, const Verified& b) {
            return a.c.consistentFraction() < b.c.consistentFraction();
        });
        return it != verified.end() && isLocal(it->r.pose, req.last_good);
    };

    // Stage B: the next batch of the sweep (plus the old camera-axis grid when
    // there is no gravity to sweep about), unless Stage A found the camera near
    // where it was lost.
    std::vector<Hypothesis> sweep;
    if (p_.use_sweep && p_.sweep_budget > 0) {
        if (!g.canSnap()) sweep = pitchGrid(base);
        const std::vector<Hypothesis> yaw = buildSweep(base, up, p_.sweep);
        sweep.insert(sweep.end(), yaw.begin(), yaw.end());
    }
    auto notIn = [](const std::vector<Hypothesis>& cands, const std::vector<Hypothesis>& tried) {
        std::vector<Hypothesis> all = tried;
        all.insert(all.end(), cands.begin(), cands.end());
        all = dedupeHypotheses(all);
        const size_t kept_tried = dedupeHypotheses(tried).size();
        return std::vector<Hypothesis>(all.begin() + static_cast<long>(std::min(all.size(), kept_tried)), all.end());
    };
    std::vector<Hypothesis> tried = fixed;
    if (!sweep.empty() && !bestIsLocal()) {
        std::vector<Hypothesis> batch;
        const size_t n = std::min(sweep.size(), static_cast<size_t>(p_.sweep_budget));
        for (size_t i = 0; i < n; ++i) batch.push_back(sweep[(cursor_ + i) % sweep.size()]);
        cursor_ = (cursor_ + n) % sweep.size();
        batch = notIn(batch, tried);
        tried.insert(tried.end(), batch.begin(), batch.end());
        const std::vector<Verified> vb = evaluate(batch, req, be, out);
        verified.insert(verified.end(), vb.begin(), vb.end());
    }
    // A far pose is only trusted once the whole sweep has been tried on THIS
    // frame: a symmetric room can hold a second, equally good basin.
    if (!sweep.empty() && !verified.empty() && !bestIsLocal()) {
        const std::vector<Hypothesis> rest = notIn(sweep, tried);
        const std::vector<Verified> vc = evaluate(rest, req, be, out);
        verified.insert(verified.end(), vc.begin(), vc.end());
    }

    if (have_best_refined_) {
        carry_ = best_refined_.pose;
        have_carry_ = true;
        out.result = best_refined_;
        out.eig_ratio = weakestDirectionRatio(best_refined_.information);
    }
    if (verified.empty()) {
        out.reject = best_basin_inliers_ >= 0 ? best_basin_reject_ : RelocReject::NoCandidate;
        return out;
    }
    decide(verified, out);
    return out;
}

bool Relocalizer::decide(std::vector<Verified>& all, RelocOutcome& out) const {
    std::stable_sort(all.begin(), all.end(), [](const Verified& a, const Verified& b) {
        return a.c.consistentFraction() > b.c.consistentFraction();
    });
    auto distant = [&](const Eigen::Matrix4f& a, const Eigen::Matrix4f& b) {
        const PoseGap gap = poseGap(a, b);
        return gap.trans_m >= p_.ambiguous_trans_m || gap.rot_deg >= p_.ambiguous_rot_deg;
    };
    bool ambiguous = false;
    for (size_t i = 1; i < all.size() && !ambiguous; ++i) {
        ambiguous = distant(all[0].r.pose, all[i].r.pose) &&
                    all[0].c.consistentFraction() - all[i].c.consistentFraction() <= p_.ambiguous_gap;
    }
    // A distant basin that fitted nearly as well in the coarse stage, refined
    // or not: on cap_001 bare ceiling corners and closet doors verified at
    // 0.76-0.96 consistency against other corners and walls while such twins
    // sat unrefined further down the coarse list.
    for (const CoarseBasin& b : run_basins_) {
        if (ambiguous || !p_.coarse_ambiguity) break;
        ambiguous = distant(all[0].r.pose, b.pose) && b.fit >= all[0].coarse_fit - p_.coarse_ambiguous_fit_gap &&
                    b.inliers >= p_.coarse_ambiguous_inlier_share * all[0].coarse_inliers;
    }
    if (ambiguous) {
        out.reject = RelocReject::Ambiguous;
        out.result = all[0].r;
        out.consistency = all[0].c;
        out.eig_ratio = weakestDirectionRatio(all[0].r.information);
        return false;
    }
    out.accepted = true;
    out.reject = RelocReject::None;
    out.eig_ratio = weakestDirectionRatio(all[0].r.information);
    out.result = all[0].r;
    out.consistency = all[0].c;
    out.source = all[0].source;
    return true;
}

std::vector<Relocalizer::Verified> Relocalizer::evaluate(const std::vector<Hypothesis>& cands,
                                                         const RelocRequest& req, const RelocBackend& be,
                                                         RelocOutcome& out) {
    std::vector<Verified> verified;
    if (cands.empty()) return verified;
    const GravityContext& g = req.gravity;

    ICPParams coarse = req.icp;
    coarse.dist_threshold = std::max(coarse.dist_threshold * p_.coarse_dist_scale, p_.coarse_min_dist_m);
    coarse.angle_threshold = p_.coarse_angle_deg;
    constexpr int kLevels = sensor::FramePyramid::LEVELS;
    for (int l = 0; l < kLevels - 1; ++l) coarse.max_iterations[l] = 0;
    coarse.max_iterations[kLevels - 1] =
        std::max(p_.coarse_min_iterations,
                 std::min(p_.coarse_max_iterations, 2 * req.icp.max_iterations[kLevels - 1]));

    // Coarse: each candidate against a model image raycast at that candidate.
    std::vector<Scored> scored;
    for (const Hypothesis& h : cands) {
        const ModelFrame& m = be.render(h.pose, RenderSize::Coarse, false);
        const ICPResult r = be.solve(m, h.pose, coarse);
        ++out.coarse_solves;
        ++out.candidates;
        const bool gradable = classifyFit(r) != TrackQuality::Failed && icpFitRatio(r) >= p_.coarse_min_fit &&
                              GravityContext::allowsReacquire(g.judge(r.pose));
        scored.push_back({r, h.source, gradable});
    }
    std::stable_sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) {
        if (a.gradable != b.gradable) return a.gradable;
        return a.r.inliers > b.r.inliers;
    });
    // One entry per basin.
    std::vector<Scored> basins;
    for (const Scored& s : scored) {
        if (!s.gradable) break;
        const bool dup = std::any_of(basins.begin(), basins.end(), [&](const Scored& b) {
            const PoseGap gap = poseGap(b.r.pose, s.r.pose);
            return gap.trans_m < p_.merge_trans_m && gap.rot_deg < p_.merge_rot_deg;
        });
        if (!dup) basins.push_back(s);
    }
    for (const Scored& b : basins) run_basins_.push_back({b.r.pose, icpFitRatio(b.r), b.r.inliers});

    // Refine the best few, then verify. The reject reason reported is the one
    // of the best-supported basin seen in this run.
    const int top = std::min(static_cast<int>(basins.size()), std::max(1, p_.refine_top));
    for (int i = 0; i < top; ++i) {
        ICPResult rr = be.solve(be.render(basins[i].r.pose, RenderSize::Refine, false), basins[i].r.pose, req.icp);
        ++out.refines;
        // Only a basin that already fits Good can be accepted, so only that
        // one is worth converging (the others mostly keep sliding).
        const bool converge = rr.pose.allFinite() && classifyFit(rr) == TrackQuality::Good;
        bool settled = !converge || p_.refine_rounds <= 1;
        for (int round = 1; converge && round < p_.refine_rounds; ++round) {
            const ICPResult next = be.solve(be.render(rr.pose, RenderSize::Refine, false), rr.pose, req.icp);
            ++out.refines;
            if (!next.pose.allFinite() || classifyFit(next) == TrackQuality::Failed) break;
            const PoseGap step = poseGap(rr.pose, next.pose);
            rr = next;
            settled = step.trans_m < p_.refine_settled_m && step.rot_deg < p_.refine_settled_deg;
            if (step.trans_m < p_.refine_converged_m && step.rot_deg < p_.refine_converged_deg) break;
        }
        if (rr.pose.allFinite() && (!have_best_refined_ || rr.inliers > best_refined_.inliers)) {
            best_refined_ = rr;
            have_best_refined_ = true;
        }
        RelocReject why = RelocReject::None;
        DepthConsistency c;
        if (classifyFit(rr) != TrackQuality::Good) {
            why = RelocReject::Fit;
        } else if (!settled) {
            why = RelocReject::Unconverged;
        } else if (!GravityContext::allowsReacquire(g.judge(rr.pose))) {
            why = RelocReject::Gravity;
        } else if (weakestDirectionRatio(rr.information) < p_.min_eig_ratio) {
            why = RelocReject::Constraint;
        } else if (blacklisted(rr.pose)) {
            why = RelocReject::Blacklisted;
        } else if (!nearVisited(rr.pose, req)) {
            why = RelocReject::Unvisited;
        } else if (!plausibleTurn(rr.pose, req)) {
            why = RelocReject::TooFast;
        } else {
            const ModelFrame& mv = be.render(rr.pose, RenderSize::Refine, true);
            c = depthConsistency(mv.vertices.data(), mv.normals.data(), mv.width, mv.height, rr.pose,
                                 req.live_depth, req.live_w, req.live_h, p_.consistency);
            out.consistency = c;
            if (!c.passes(p_.consistency)) why = RelocReject::Consistency;
        }
        if (why == RelocReject::None) {
            verified.push_back({rr, c, basins[i].source, icpFitRatio(basins[i].r), basins[i].r.inliers});
        } else if (rr.inliers > best_basin_inliers_) {
            best_basin_inliers_ = rr.inliers;
            best_basin_reject_ = why;
        }
    }
    return verified;
}

} // namespace tracking
} // namespace kfusion
