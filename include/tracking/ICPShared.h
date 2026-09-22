#pragma once

#include <Eigen/Dense>

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

// Canonical CPU ICP numeric-policy constants (big-fix Todo 13). These values and
// the predicates below are the single CPU source of truth for adaptive damping,
// step caps, validity gates, SO(3) projection, and the accepted-step success
// rule. CUDA/HIP ports remain deferred backend work.
inline constexpr float kDampingDefault       = 0.1f;
inline constexpr float kDampingLevel0        = 0.01f;
inline constexpr float kDampingEscalated     = 1.0f;
inline constexpr float kCondEscalate         = 1e7f;
inline constexpr float kMinEigEscalate       = 1e-4f;
inline constexpr float kTranslationCapStep   = 0.2f;
inline constexpr float kRotationCapStep      = 0.5f;
inline constexpr float kModelVertexNormSqMin = 1e-12f;
inline constexpr float kNormalNormSqMin      = 0.9f;
inline constexpr float kConvergenceStep      = 5e-5f;
inline constexpr float kAcceptableFinalStep  = 1e-3f;
inline constexpr float kAngleThresholdFallbackDeg = 30.0f;
inline constexpr float kAngleThresholdMinDeg      = 0.0f;
inline constexpr float kAngleThresholdMaxDeg      = 85.0f;
inline constexpr int   kMinInliersForIteration = 10;
inline constexpr int   kMinInliersForOk        = 100;

inline bool modelVertexIsValid(const Eigen::Vector3f& v) {
    return v.allFinite() && v.squaredNorm() > kModelVertexNormSqMin;
}

inline bool normalIsValid(const Eigen::Vector3f& n) {
    return n.allFinite() && n.squaredNorm() > kNormalNormSqMin;
}

inline float dampingForHessian(const Eigen::Matrix<float, 6, 6>& A, int level) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<float, 6, 6>> solver(A);
    if (solver.info() != Eigen::Success) return kDampingEscalated;

    const Eigen::Matrix<float, 6, 1> eig = solver.eigenvalues();
    const float min_eig = eig[0];
    const float max_eig = eig[eig.size() - 1];
    if (!std::isfinite(min_eig) || !std::isfinite(max_eig) || min_eig < kMinEigEscalate) {
        return kDampingEscalated;
    }
    if (max_eig > kCondEscalate * min_eig) {
        return kDampingEscalated;
    }
    return level == 0 ? kDampingLevel0 : kDampingDefault;
}

inline Eigen::Matrix3f projectToSO3(const Eigen::Matrix3f& M) {
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f U = svd.matrixU();
    const Eigen::Matrix3f V = svd.matrixV();
    if ((U * V.transpose()).determinant() < 0.0f) {
        U.col(2) = -U.col(2);
    }
    return U * V.transpose();
}

} // namespace tracking
} // namespace kfusion
