#pragma once

#include <algorithm>
#include <cmath>

namespace kfusion {
namespace tracking {

/**
 * Shared CPU tracking constants (big-fix Todo 12), header-only and CPU-only.
 *
 * This is the single source of truth for the two robustness constants the CPU
 * tracking path duplicates: the ICP Huber threshold and the depth-discontinuity
 * (jump) threshold used by the level-0 normal kernel and by pyramid downsampling
 * (the pyramid exists to feed ICP, so its guard constant lives here too; see
 * docs/CANONICAL_SEMANTICS.md and dossier rows tracking:A13 / tracking:A34).
 * The CUDA/HIP backends still carry their own inline literals; porting them
 * onto these constants is deferred backend work, recorded in
 * docs/CUDA_HIP_DEFERRED_CHANGES.md, not done in this run.
 */

// Huber transition point for the point-to-plane residual, in meters. The
// canonical IRLS/MM triple that consumes it (locked by
// tests/icp_weighting_contract.cpp): with w = min(1, kHuberK / |e|),
//   curvature:  A += w * J * J^T          (weight w, NOT 1 and NOT w^2)
//   gradient:   b -= J * (w * e)          (already w-weighted; kept as-is)
//   objective:  psi(|e|) = e^2            (|e| <= k)
//              psi(|e|) = 2*k*|e| - k^2   (|e| >  k)
// The w^2 curvature claimed by the pre-Todo-12 docs is false: it pairs with a
// w^2*e gradient, not with the w*e gradient this code computes, and 1/2*(w*e)^2
// is bounded by 2*k^2 so it cannot majorize the unbounded Huber loss it reports.
inline constexpr float kHuberK = 0.02f;

// True Huber loss psi of the absolute residual (the reported per-correspondence
// objective; continuous at t == k, where both branches give k^2). Callers must
// pass a finite, non-negative abs_err; a non-finite residual is rejected before
// the weight/loss is evaluated (src/tracking/ICPTracker.cpp).
inline constexpr float huberLossFromAbs(float abs_err, float k = kHuberK) {
    return (abs_err <= k) ? abs_err * abs_err : 2.0f * k * abs_err - k * k;
}

// Depth-discontinuity guard: two depths are treated as the same surface only
// while they agree within depthJumpThreshold(d) = max(kDepthJumpBaseMeters,
// kDepthJumpRelFrac * d) — an absolute floor for near ranges plus a 5%
// proportional term for far ones. Values are the pre-Todo-12 level-0 normal
// kernel literals, moved here unchanged so the normal kernel and the pyramid
// downsample (Todo 12) cannot drift apart.
inline constexpr float kDepthJumpBaseMeters = 0.03f;
inline constexpr float kDepthJumpRelFrac    = 0.05f;

inline float depthJumpThreshold(float depth_m) {
    return std::max(kDepthJumpBaseMeters, depth_m * kDepthJumpRelFrac);
}

} // namespace tracking
} // namespace kfusion
