// icp_weighting_contract (big-fix todo 12): CPU-only contract for the canonical
// Huber IRLS/MM triple, non-finite rejection, and the CPU pyramid depth-jump guard.
//
//   * A += w*J*J^T (curvature weight w), b -= J*(w*e) (gradient weight w*e),
//     reported objective = sum of the true Huber loss psi(|e|), k = 0.02.
//   * A non-finite residual or Jacobian entry rejects the correspondence without
//     poisoning A/b/the objective (remaining data must reproduce the clean run
//     bit-for-bit).
//   * kfusion::sensor::buildFramePyramid 2x downsampling must not average depth
//     across a discontinuity beyond the shared max(0.03, 0.05*d_min) threshold;
//     it keeps the nearest (smallest-depth) valid sample instead.
//
// The oracle is hand-derived in double from the fixture definition (literal
// expected numerators/denominators), never by calling the product code or a
// helper. Todo 13 makes the undamped Hessian's zero eigenvalues escalate the
// level-0 damping to 1.0, so the canonical triple is locked against
// unweighted curvature (-0.12/12), w^2 curvature with w^2*e gradient
// (-0.105/11.0625), and the (w*e)^2 objective (0.0014/11).
// Public API only (ICPTracker::track, buildFramePyramid). No device, display,
// GPU, sensor, thread timing or filesystem.

#include "sensor/FrameData.h"
#include "tracking/ICPTracker.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "icp_weighting_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

using kfusion::sensor::CX;
using kfusion::sensor::CY;
using kfusion::sensor::FX;
using kfusion::sensor::FY;
using kfusion::tracking::ICPParams;
using kfusion::tracking::ICPResult;
using kfusion::tracking::ICPTracker;
using kfusion::tracking::ModelFrame;

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, what)                                                        \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            std::printf("FAIL: %s  [%s:%d]\n", std::string(what).c_str(),       \
                        __FILE__, __LINE__);                                     \
            ++g_failures;                                                        \
        }                                                                        \
    } while (false)

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();

// ---------------------------------------------------------------------------
// ICP fixture: identity pose, level-0-only iteration, 10 inliers + 1 outlier.
//
// 11 live pixels at z = 1.0 m; model normals (0,0,1); model plane offset so the
// point-to-plane residual is exactly the depth gap (err = v.z - model_v.z with
// identity poses). 10 inliers carry err = 0.01 (w = 1); one outlier near the
// optical center carries err = 0.08 (w = 0.25). The 10 inliers are five pixel
// pairs point-symmetric about the optical center (319.5, 239.5), so their
// cross-coupling into rows 3..5 cancels exactly and t_z decouples:
//   A(2,2) = sum(w_i * Jz^2) + 1.0 escalated damping = 10*1 + 0.25 + 1 = 11.25
//   b(2)   = -sum(w_i * e_i)                        = -(10*0.01 + 0.25*0.08) = -0.12
//   t_z    = -0.12 / 11.25                          (canonical, w curvature)
//   t_z    = -0.12 / (11 + 1) = -0.12/12            (unweighted curvature)
//   t_z    = -0.105 / (10 + 0.25^2 + 1)             (w^2 curvature + w^2 e: option B)
// Objective per correspondence is psi(t) = t^2 (t <= k) or 2*k*t - k^2 (t > k):
//   error = (10*0.01^2 + (2*0.02*0.08 - 0.02^2)) / 11 = 0.0038/11
//   the rejected (w*e)^2 objective would give (10*0.0001 + 0.0004)/11 = 0.0014/11.
// ---------------------------------------------------------------------------

// Hand-derived canonical references (double, from the fixture definition only).
constexpr double kTzCanonical = -0.12 / 11.25;          // -0.01066667 (escalated damping)
constexpr double kTzUnweighted = -0.12 / 12.0;          // -0.01000000 (unweighted curvature)
constexpr double kTzWSquared = -0.105 / 11.0625;        // -0.00949153 (option B)
constexpr double kErrCanonical = 0.0038 / 11.0;         //  3.4545455e-4
constexpr double kErrWeightedSq = 0.0014 / 11.0;        //  1.2727273e-4 (pre-fix)

// Discrimination gaps: |canonical-unweighted| ~ 6.7e-4, |canonical-w2| ~ 1.2e-3,
// |errCanonical-errWeightedSq| ~ 2.2e-4. Float evaluation noise is ~1e-8, so a
// 5e-6 match tolerance and 2e-4 gap floors keep every assertion decisive.
constexpr double kTolMatch = 5e-6;
constexpr double kGapFloor = 2e-4;

enum Corruption {
    kClean = 0,
    kInfModelNormalExtra,  // extra correspondence with a model normal (0,0,+Inf)
    kNaNModelDepthExtra    // extra correspondence with a NaN model depth (NaN residual,
                           // all six Jacobian entries finite: the pre-fix poison path)
};

struct RunPair {
    ICPResult first;
    ICPResult second;
};

// Places one live/model correspondence pair. Live pixel (px,py) back-projects at
// depth z; the model vertex shares the live x/y so the residual is exactly the
// z gap. Pixel projections land exactly on (px,py) because the tracker computes
// FX * ((px-CX)/FX) + CX which round-trips to px+/-0.5 -> floor(px+0.5) = px.
void putCorrespondence(kfusion::sensor::FramePyramid& live, ModelFrame& model,
                       int px, int py, float err, bool inf_model_normal,
                       bool nan_model_depth) {
    const float z = 1.0f;
    const float vx = (static_cast<float>(px) - CX) / FX * z;
    const float vy = (static_cast<float>(py) - CY) / FY * z;
    const size_t lidx = static_cast<size_t>(py) * kfusion::sensor::FRAME_W + px;
    live.levels[0].vertices[lidx] = Eigen::Vector3f(vx, vy, z);
    live.levels[0].normals[lidx]  = Eigen::Vector3f(0.0f, 0.0f, 1.0f);

    Eigen::Vector3f model_v(vx, vy, z - err);
    if (nan_model_depth) model_v.z() = kNaN;
    const size_t midx = static_cast<size_t>(py) * kfusion::sensor::FRAME_W + px;
    model.vertices[midx] = model_v;
    model.normals[midx] =
        inf_model_normal ? Eigen::Vector3f(0.0f, 0.0f, kInf) : Eigen::Vector3f(0.0f, 0.0f, 1.0f);
}

RunPair runFixture(Corruption corruption) {
    ICPParams params;
    params.max_iterations[0] = 1;  // exactly one level-0 iteration
    params.max_iterations[1] = 0;
    params.max_iterations[2] = 0;
    ICPTracker tracker(params);
    tracker.setNumThreads(1);  // single-threaded: no reduction-order variance

    kfusion::sensor::FramePyramid live;
    ModelFrame model;

    for (int k = 0; k < 5; ++k) {
        putCorrespondence(live, model, 319 - k, 239 - k, 0.01f, false, false);
        putCorrespondence(live, model, 320 + k, 240 + k, 0.01f, false, false);
    }
    putCorrespondence(live, model, 320, 239, 0.08f, false, false);  // outlier near center

    switch (corruption) {
        case kInfModelNormalExtra: putCorrespondence(live, model, 340, 260, 0.01f, true, false); break;
        case kNaNModelDepthExtra:  putCorrespondence(live, model, 300, 220, 0.01f, false, true); break;
        case kClean: break;
    }

    const Eigen::Matrix4f id = Eigen::Matrix4f::Identity();
    RunPair pair{tracker.track(live, model, id, id), tracker.track(live, model, id, id)};
    return pair;
}

bool poseBitEq(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b) {
    return std::memcmp(a.data(), b.data(), sizeof(float) * 16) == 0;
}

void checkCounters(const ICPResult& r, int expect_valid_live, int expect_valid_model,
                   const std::string& who) {
    CHECK(r.valid_live_points == expect_valid_live,
          who + ": valid_live == " + std::to_string(expect_valid_live) +
              ", got " + std::to_string(r.valid_live_points));
    CHECK(r.valid_model_points == expect_valid_model,
          who + ": valid_model == " + std::to_string(expect_valid_model) +
              ", got " + std::to_string(r.valid_model_points));
    CHECK(r.dist_filtered == 0, who + ": no distance filtering, got " + std::to_string(r.dist_filtered));
    CHECK(r.angle_filtered == 0, who + ": no angle filtering, got " + std::to_string(r.angle_filtered));
}

void testCanonicalWeighting() {
    const RunPair rp = runFixture(kClean);
    const ICPResult& r = rp.first;

    // Self-validation of the fixture BEFORE locking any pose value: exactly 11
    // correspondences, no distance/angle filtering, no invalid model/live pixels.
    checkCounters(r, 11, 11, "clean fixture");
    CHECK(r.projected_points == 11,
          "clean fixture: projected == 11, got " + std::to_string(r.projected_points));
    CHECK(r.inliers == 11, "clean fixture: inliers == 11, got " + std::to_string(r.inliers));

    CHECK(r.pose.allFinite() && r.pose.col(3).allFinite(), "clean: pose is finite");
    CHECK(std::isfinite(r.error), "clean: reported error is finite");

    const double tz = static_cast<double>(r.pose(2, 3));
    CHECK(std::abs(tz - kTzCanonical) <= kTolMatch,
          "t_z == canonical -0.12/11.25 = " + std::to_string(kTzCanonical) +
              ", got " + std::to_string(tz));
    CHECK(std::abs(tz - kTzUnweighted) >= kGapFloor,
          "t_z is NOT the unweighted-curvature value " + std::to_string(kTzUnweighted) +
              " (|diff| >= 2e-4), got " + std::to_string(tz));
    CHECK(std::abs(tz - kTzWSquared) >= kGapFloor,
          "t_z is NOT the w^2-curvature/w^2*e value " + std::to_string(kTzWSquared) +
              " (|diff| >= 2e-4), got " + std::to_string(tz));

    const double err = static_cast<double>(r.error);
    CHECK(std::abs(err - kErrCanonical) <= kTolMatch,
          "error == canonical Huber psi sum 0.0038/11 = " + std::to_string(kErrCanonical) +
              ", got " + std::to_string(err));
    CHECK(std::abs(err - kErrWeightedSq) >= kGapFloor,
          "error is NOT the (w*e)^2 objective " + std::to_string(kErrWeightedSq) +
              " (|diff| >= 2e-4), got " + std::to_string(err));

    // Jx = Jy = 0 for every correspondence (model normals are +z), so the
    // translation x/y components are exactly zero at this fixture.
    CHECK(std::abs(static_cast<double>(r.pose(0, 3))) <= kTolMatch, "t_x == 0 (tol 5e-6)");
    CHECK(std::abs(static_cast<double>(r.pose(1, 3))) <= kTolMatch, "t_y == 0 (tol 5e-6)");

    // Determinism: two runs over identical input must be bit-identical.
    CHECK(poseBitEq(r.pose, rp.second.pose), "clean: pose is bit-identical across two runs");
    CHECK(r.error == rp.second.error, "clean: error is bit-identical across two runs");
    CHECK(r.inliers == rp.second.inliers, "clean: inlier count is identical across two runs");

    std::printf("  canonical weighting: t_z=%.9f (ref %.9f) error=%.9f (ref %.9f) inliers=%d\n",
                tz, kTzCanonical, err, kErrCanonical, r.inliers);
}

void testRejectsInfModelNormal() {
    const RunPair clean = runFixture(kClean);
    const RunPair rp = runFixture(kInfModelNormalExtra);
    const ICPResult& r = rp.first;

    // The (0,0,+Inf) model normal is rejected by the squared-norm validity gate
    // before valid_model is incremented. The sample is still projected, but only
    // the 11 clean model points count; the clean correspondences reproduce the
    // clean result bit-for-bit.
    CHECK(r.projected_points == 12,
          "inf normal: corrupt sample is projected (12), got " +
              std::to_string(r.projected_points));
    CHECK(r.inliers == 11, "inf normal: inliers stays 11, got " + std::to_string(r.inliers));
    CHECK(r.valid_model_points == 11,
          "inf normal: invalid model normal is not counted by valid_model (11), got " +
              std::to_string(r.valid_model_points));
    CHECK(r.dist_filtered == 0 && r.angle_filtered == 0, "inf normal: corrupt sample not dist/angle filtered");
    CHECK(r.pose.allFinite() && std::isfinite(r.error), "inf normal: pose and error remain finite");
    CHECK(poseBitEq(r.pose, clean.first.pose), "inf normal: pose bit-identical to clean run");
    CHECK(r.error == clean.first.error, "inf normal: error bit-identical to clean run");
    CHECK(poseBitEq(r.pose, rp.second.pose) && r.error == rp.second.error,
          "inf normal: repeat run bit-identical");
    std::printf("  inf model normal: rejected, inliers=%d error=%.9f (clean %.9f)\n", r.inliers,
                static_cast<double>(r.error), static_cast<double>(clean.first.error));
}

void testRejectsNaNResidual() {
    const RunPair clean = runFixture(kClean);
    const RunPair rp = runFixture(kNaNModelDepthExtra);
    const ICPResult& r = rp.first;

    // A NaN model depth is rejected by the finite model-vertex validity gate
    // before valid_model is incremented. The sample is still projected, but the
    // remaining 11 correspondences reproduce the clean result bit-for-bit.
    CHECK(r.projected_points == 12,
          "nan residual: corrupt sample is projected (12), got " +
              std::to_string(r.projected_points));
    CHECK(r.inliers == 11, "nan residual: inliers stays 11, got " + std::to_string(r.inliers));
    CHECK(r.valid_model_points == 11,
          "nan residual: invalid model vertex is not counted by valid_model (11), got " +
              std::to_string(r.valid_model_points));
    CHECK(r.pose.allFinite() && std::isfinite(r.error), "nan residual: pose and error remain finite");
    CHECK(poseBitEq(r.pose, clean.first.pose), "nan residual: pose bit-identical to clean run");
    CHECK(r.error == clean.first.error, "nan residual: error bit-identical to clean run");
    CHECK(poseBitEq(r.pose, rp.second.pose) && r.error == rp.second.error,
          "nan residual: repeat run bit-identical");
    std::printf("  nan residual: rejected, inliers=%d t_z=%.9f (clean %.9f)\n", r.inliers,
                static_cast<double>(r.pose(2, 3)), static_cast<double>(clean.first.pose(2, 3)));
}

// ---------------------------------------------------------------------------
// Pyramid depth-jump guard (public buildFramePyramid on a small 16x16 field).
// downsample() averages 2x2 blocks of VALID samples. Guard: when the valid
// depths of a block span more than max(0.03, 0.05 * d_min), do not average
// across the jump - keep the nearest (smallest-depth) valid sample instead.
// ---------------------------------------------------------------------------

constexpr int kPW = 16;
constexpr int kPH = 16;

kfusion::sensor::FrameData makeField() {
    kfusion::sensor::FrameData f;
    f.width  = kPW;
    f.height = kPH;
    f.vertices.assign(static_cast<size_t>(kPW) * kPH, Eigen::Vector3f::Zero());
    f.normals.assign(static_cast<size_t>(kPW) * kPH, Eigen::Vector3f::Zero());
    f.depth_meters.assign(static_cast<size_t>(kPW) * kPH, 1.0f);
    f.rgb.assign(static_cast<size_t>(kPW) * kPH * 3, 0);
    for (int y = 0; y < kPH; ++y) {
        for (int x = 0; x < kPW; ++x) {
            const size_t i = static_cast<size_t>(y) * kPW + x;
            // Arbitrary deterministic planar field; the expected smooth average
            // is derived in double from these stored float corner values.
            f.vertices[i] = Eigen::Vector3f(0.001f * static_cast<float>(x),
                                           0.002f * static_cast<float>(y), 1.0f);
            f.normals[i]  = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        }
    }
    return f;
}

float& depthAt(kfusion::sensor::FrameData& f, int x, int y) {
    return f.depth_meters[static_cast<size_t>(y) * f.width + x];
}
float depthRef(const kfusion::sensor::FrameData& f, int x, int y) {
    return f.depth_meters[static_cast<size_t>(y) * f.width + x];
}
const Eigen::Vector3f& vertexAt(const kfusion::sensor::FrameData& f, int x, int y) {
    return f.vertices[static_cast<size_t>(y) * f.width + x];
}
const Eigen::Vector3f& normalAt(const kfusion::sensor::FrameData& f, int x, int y) {
    return f.normals[static_cast<size_t>(y) * f.width + x];
}

void testPyramidDepthJumpGuard() {
    kfusion::sensor::FrameData src = makeField();

    // Block coarse(0,0): src (0,0),(1,0),(0,1),(1,1). One sample at 1.1 among
    // 1.0: span 0.1 > max(0.03, 0.05*1.0) -> keep nearest sample (0,0).
    depthAt(src, 1, 0) = 1.1f;
    // Block coarse(1,0): src (2,0),(3,0),(2,1),(3,1): descending depths, the
    // minimum sits at the far corner (3,1) -> that exact sample must be kept.
    depthAt(src, 2, 0) = 1.3f;
    depthAt(src, 3, 0) = 1.2f;
    depthAt(src, 2, 1) = 1.1f;
    depthAt(src, 3, 1) = 1.0f;
    // Block coarse(0,1): src (0,2),(1,2),(0,3),(1,3): span 0.025 <= 0.05 ->
    // unchanged smoothing behavior (average of the valid samples).
    depthAt(src, 1, 2) = 1.025f;
    // Block coarse(1,1): one invalid (sentinel 0) among three valid 1.0 samples:
    // invalid handling unchanged (average over the three valid samples).
    depthAt(src, 3, 3) = 0.0f;
    // Block coarse(2,2): all four invalid -> coarse pixel stays invalid.
    depthAt(src, 4, 4) = 0.0f;
    depthAt(src, 5, 4) = 0.0f;
    depthAt(src, 4, 5) = 0.0f;
    depthAt(src, 5, 5) = 0.0f;

    kfusion::sensor::FramePyramid pyr;
    kfusion::sensor::buildFramePyramid(src, pyr);
    const kfusion::sensor::FrameData& l1 = pyr.levels[1];

    CHECK(l1.width == kPW / 2 && l1.height == kPH / 2, "pyramid: level 1 is 8x8");

    // Jump block -> nearest (smallest-depth) sample kept, NOT the average.
    // Pre-fix the coarse depth averaged across the jump (1.025 and 1.15).
    CHECK(vertexAt(l1, 0, 0).z() == 1.0f && depthRef(l1, 0, 0) == 1.0f,
          "pyramid guard: coarse(0,0) depth == nearest 1.0 (not avg 1.025), got " +
              std::to_string(depthRef(l1, 0, 0)));
    CHECK(vertexAt(l1, 0, 0) == vertexAt(src, 0, 0),
          "pyramid guard: coarse(0,0) vertex == src(0,0) sample exactly");
    CHECK(depthRef(l1, 1, 0) == 1.0f,
          "pyramid guard: coarse(1,0) depth == nearest 1.0 (not avg 1.15), got " +
              std::to_string(depthRef(l1, 1, 0)));
    CHECK(vertexAt(l1, 1, 0) == vertexAt(src, 3, 1),
          "pyramid guard: coarse(1,0) vertex == src(3,1) sample exactly (min at far corner)");
    CHECK(normalAt(l1, 0, 0) == Eigen::Vector3f(0.0f, 0.0f, 1.0f) &&
              normalAt(l1, 1, 0) == Eigen::Vector3f(0.0f, 0.0f, 1.0f),
          "pyramid guard: coarse normals kept the nearest sample's normal");

    // Smooth block within threshold: unchanged 2x2 averaging behavior.
    const double smooth_d = (1.0 + 1.025 + 1.0 + 1.0) / 4.0;
    CHECK(std::abs(static_cast<double>(depthRef(l1, 0, 1)) - smooth_d) <= 1e-6,
          "pyramid smooth: coarse(0,1) depth == mean 1.00625, got " +
              std::to_string(depthRef(l1, 0, 1)));
    const Eigen::Vector3f smooth_v = (vertexAt(src, 0, 2) + vertexAt(src, 1, 2) +
                                      vertexAt(src, 0, 3) + vertexAt(src, 1, 3)) * 0.25f;
    CHECK((vertexAt(l1, 0, 1) - smooth_v).norm() <= 1e-6f,
          "pyramid smooth: coarse(0,1) vertex == 2x2 mean (behavior preserved)");

    // Invalid handling unchanged: average over the three valid samples.
    CHECK(std::abs(static_cast<double>(depthRef(l1, 1, 1)) - 1.0) <= 1e-6,
          "pyramid invalid: coarse(1,1) depth == mean of three valid samples");
    // Fully invalid block stays invalid (zeros).
    CHECK(depthRef(l1, 2, 2) == 0.0f && vertexAt(l1, 2, 2) == Eigen::Vector3f::Zero() &&
              normalAt(l1, 2, 2) == Eigen::Vector3f::Zero(),
          "pyramid invalid: all-sentinel block yields a zero coarse pixel");

    // Flat interior blocks still average.
    CHECK(std::abs(static_cast<double>(depthRef(l1, 3, 3)) - 1.0) <= 1e-6 &&
              vertexAt(l1, 3, 3).z() == 1.0f,
          "pyramid smooth: flat block depth == 1.0");

    // Determinism: byte-identical pyramid across a repeated build.
    kfusion::sensor::FramePyramid pyr2;
    kfusion::sensor::buildFramePyramid(src, pyr2);
    size_t diff = 0;
    for (int l = 0; l < kfusion::sensor::FramePyramid::LEVELS; ++l) {
        const kfusion::sensor::FrameData& a = pyr.levels[l];
        const kfusion::sensor::FrameData& b = pyr2.levels[l];
        if (a.vertices != b.vertices || a.normals != b.normals || a.depth_meters != b.depth_meters) {
            ++diff;
        }
    }
    CHECK(diff == 0, "pyramid: byte-identical across repeated builds");

    std::printf("  pyramid guard: jump blocks keep nearest sample; smooth/invalid behavior preserved; deterministic\n");
}

} // namespace

int main() {
    testCanonicalWeighting();
    testRejectsInfModelNormal();
    testRejectsNaNResidual();
    testPyramidDepthJumpGuard();

    if (g_failures == 0) {
        std::printf("icp_weighting_contract: PASS (%d checks: canonical Huber triple + inf-normal "
                    "rejection + NaN-residual rejection + pyramid depth-jump guard)\n", g_checks);
        return 0;
    }
    std::printf("icp_weighting_contract: FAIL (%d failed checks of %d)\n", g_failures, g_checks);
    return 1;
}
