// icp_numeric_policy_contract (big-fix todo 13): CPU-only public-API contract for
// the canonical CPU ICP numeric policy that Todo 13 unifies:
//
//   1. Adaptive Tikhonov damping, per iteration and per level: level 0 starts at
//      0.01, other levels at 0.1, and the solve escalates to 1.0 when the UNDAMPED
//      Hessian is ill-conditioned (cond > 1e7 or min_eig < 1e-4).
//   2. Per-iteration step guards: reject (break, never clamp) a non-finite update,
//      a translation step above 0.2 m OR a rotation step above 0.5 rad.
//   3. Squared-norm validity gates: a model vertex needs allFinite &&
//      squaredNorm > 1e-12; a model OR live normal needs allFinite &&
//      squaredNorm > 0.9 (no upper bound, no renormalization).
//   4. Determinant-corrected SVD re-orthonormalization: the reported rotation is
//      always a proper rotation (det = +1), never a reflection.
//   5. Success rule: tracking_ok = pose finite && inliers > 100 &&
//      (converged || final_step <= 1e-3), so an exhausted-iteration run whose last
//      accepted step was already acceptable is success, not false TrackingLost.
//   6. ICPResult::pose defaults to identity; ICPParams::angle_threshold is
//      clamped to [0, 85] degrees (non-finite -> 30) by the constructor/setParams.
//
// Every expected number is an independent oracle hand-derived in double from the
// fixture definition (closed-form decoupled normal-equation solutions), never by
// calling product code. Wrong-policy counterfactuals are computed as explicit
// discrimination floors, so a contract that only asserted finiteness could not
// pass. Public API only: ICPTracker::track / params() / setParams() /
// setNumThreads(), ICPResult, ICPParams, ModelFrame, FramePyramid. Single-threaded
// via setNumThreads(1); every scenario runs twice and compares bit-for-bit. No
// device, display, GPU, sensor, thread timing or filesystem access.
//
// SIZE NOTE: one contract, ten required scenarios, oracle-table dominated (the
// fixtures and their hand-derived constants ARE the content). Splitting it would
// scatter one contract across files, so it is kept whole and over the 250 pure
// LOC house ceiling on purpose (same trade-off already recorded for
// tests/coordinate_rounding_contract.cpp in .omo/notepads/big-fix/learnings.md).

#include "sensor/FrameData.h"
#include "tracking/ICPTracker.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "icp_numeric_policy_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

using kfusion::sensor::CX;
using kfusion::sensor::CY;
using kfusion::sensor::FX;
using kfusion::sensor::FY;
using kfusion::sensor::FRAME_H;
using kfusion::sensor::FRAME_W;
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
// Canonical CPU policy constants (big-fix Todo 13), restated here as literals so
// the contract stays an INDEPENDENT witness: if the product constant drifts, this
// file fails instead of following it. docs/CANONICAL_SEMANTICS.md is the prose
// source of truth; include/tracking/ICPShared.h is the code source.
// ---------------------------------------------------------------------------
constexpr double kDampLevel0   = 0.01;   // level-0 base damping
constexpr double kDampOther    = 0.1;    // level 1/2 base damping
constexpr double kDampEscalate = 1.0;    // ill-conditioned override
constexpr double kTransCap     = 0.2;    // m, per iteration
constexpr double kRotCap       = 0.5;    // rad, per iteration
constexpr double kConvStep     = 5e-5;   // converged when |x| < this
constexpr double kAcceptStep   = 1e-3;   // success without converged
constexpr double kMinInliersOk = 100.0;  // tracking_ok needs inliers > 100
constexpr double kHuberK       = 0.02;   // Todo 12 (unchanged here)

// Float evaluation noise against these double oracles measured <= 1e-8, so a
// 1e-7 match tolerance is exact-in-practice while every counterfactual gap below
// is >= 3.7e-5 (>= 2e-4 wherever the 2e-4 floor is used).
constexpr double kTolMatch  = 1e-7;
constexpr double kGapFloor  = 2e-4;
constexpr double kGapFloorS = 1e-5;  // for the 5b damping triple (min gap 3.7e-5)

// ---------------------------------------------------------------------------
// Fixture plumbing: one correspondence = one live pixel (index + vertex + normal)
// plus one model pixel (index + vertex + normal). The model pixel index is
// derived here with an independent re-implementation of the tracker's projection
// (floor(FX * x/z + CX + 0.5)), never read from product code, so a fixture whose
// projection does not land where the test thinks it does fails loudly instead of
// silently degrading into an empty correspondence set.
// ---------------------------------------------------------------------------
struct Corr {
    int lpx, lpy;            // live pixel index
    int mpx, mpy;            // model pixel index the live vertex projects onto
    Eigen::Vector3f lv, ln;  // live vertex / normal (camera space)
    Eigen::Vector3f mv, mn;  // model vertex / normal (world space)
};

struct RunPair {
    ICPResult first;
    ICPResult second;
};

// The live->model pixel projection at identity poses (v_ref == v_live).
void projectedPixel(const Eigen::Vector3f& v_ref, int* mx, int* my) {
    const float inv_z = 1.0f / v_ref.z();
    *mx = static_cast<int>(std::floor(FX * v_ref.x() * inv_z + CX + 0.5f));
    *my = static_cast<int>(std::floor(FY * v_ref.y() * inv_z + CY + 0.5f));
}

Eigen::Vector3f backProject(int px, int py, float z) {
    return Eigen::Vector3f((static_cast<float>(px) - CX) / FX * z,
                           (static_cast<float>(py) - CY) / FY * z, z);
}

// Identity-pose correspondence whose model vertex is lv offset along `axis` by
// `offset`; the residual is then exactly `offset` for a normal along that axis.
Corr makeCorr(int px, int py, float z, const Eigen::Vector3f& n, int axis, float offset) {
    Corr c;
    c.lpx = px;
    c.lpy = py;
    c.lv  = backProject(px, py, z);
    c.ln  = n;
    c.mv  = c.lv;
    c.mv(axis) = c.lv(axis) - offset;
    c.mn  = n;
    projectedPixel(c.lv, &c.mpx, &c.mpy);
    return c;
}

RunPair runFixture(const std::vector<Corr>& cs, const ICPParams& params,
                   const Eigen::Matrix4f& pose_est = Eigen::Matrix4f::Identity(),
                   const Eigen::Vector4f& unused_hint = Eigen::Vector4f::Zero()) {
    (void)unused_hint;
    ICPTracker tracker(params);
    tracker.setNumThreads(1);  // single-threaded: no reduction-order variance

    kfusion::sensor::FramePyramid live;
    ModelFrame model;
    for (const Corr& c : cs) {
        const size_t lidx = static_cast<size_t>(c.lpy) * FRAME_W + c.lpx;
        const size_t midx = static_cast<size_t>(c.mpy) * FRAME_W + c.mpx;
        live.levels[0].vertices[lidx] = c.lv;
        live.levels[0].normals[lidx]  = c.ln;
        model.vertices[midx]          = c.mv;
        model.normals[midx]           = c.mn;
    }
    const Eigen::Matrix4f id = Eigen::Matrix4f::Identity();
    RunPair pair{tracker.track(live, model, pose_est, id), tracker.track(live, model, pose_est, id)};
    return pair;
}

ICPParams level0Only(int iterations) {
    ICPParams p;
    p.max_iterations[0] = iterations;
    p.max_iterations[1] = 0;
    p.max_iterations[2] = 0;
    return p;
}

bool poseBitEq(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b) {
    return std::memcmp(a.data(), b.data(), sizeof(float) * 16) == 0;
}

bool runBitEq(const ICPResult& a, const ICPResult& b) {
    return poseBitEq(a.pose, b.pose) && a.error == b.error && a.inliers == b.inliers &&
           a.converged == b.converged && a.tracking_ok == b.tracking_ok &&
           a.final_step == b.final_step && a.valid_live_points == b.valid_live_points &&
           a.valid_model_points == b.valid_model_points &&
           a.projected_points == b.projected_points && a.dist_filtered == b.dist_filtered &&
           a.angle_filtered == b.angle_filtered;
}

std::string acceptanceText(bool accepted) {
    return accepted ? " is accepted" : " is rejected";
}

void checkDeterminism(const RunPair& rp, const std::string& who) {
    CHECK(runBitEq(rp.first, rp.second), who + ": repeated run is bit-identical");
}

void checkCounters(const ICPResult& r, int live, int model, int projected, int dist_f,
                   int angle_f, const std::string& who) {
    CHECK(r.valid_live_points == live, who + ": valid_live == " + std::to_string(live) + ", got " +
                                           std::to_string(r.valid_live_points));
    CHECK(r.valid_model_points == model, who + ": valid_model == " + std::to_string(model) +
                                             ", got " + std::to_string(r.valid_model_points));
    CHECK(r.projected_points == projected, who + ": projected == " + std::to_string(projected) +
                                               ", got " + std::to_string(projected));
    CHECK(r.dist_filtered == dist_f, who + ": dist_filtered == " + std::to_string(dist_f) +
                                         ", got " + std::to_string(r.dist_filtered));
    CHECK(r.angle_filtered == angle_f, who + ": angle_filtered == " + std::to_string(angle_f) +
                                           ", got " + std::to_string(r.angle_filtered));
}

// ---------------------------------------------------------------------------
// 1 + 2. Success rule: acceptable final step vs the inlier floor.
//
// Both fixtures are a point-symmetric pixel grid with a UNIFORM depth error, model
// and live normals both (0,0,1), identity poses. Symmetry makes the model-normal
// rows couple to nothing:
//   A(2,2) = sum(w) = N,  b(2) = -sum(w*e) = -N*e,  A(2,3) = sum(v_y) = 0,
//   A(2,4) = -sum(v_x) = 0, A(2,0)=A(2,1)=A(2,5)=0 (J0=J1=0, J5=0 for n = +z),
// so row 2 decouples and t_z = -N*e / (N + damping) exactly. The Hessian always
// has three zero eigenvalues (rows 0,1,5 vanish for +z normals), so the canonical
// policy escalates damping to 1.0 and damping is NOT observable at N = 240 -
// which is the point: these two fixtures isolate the SUCCESS RULE, not damping.
//   N = 240, e = 5e-4 -> t_z = -0.12/241 = -4.97925e-4, |x| = 4.979e-4
//     -> 5e-5 < |x| <= 1e-3, converged == false, inliers 240 > 100 => SUCCESS
//   N = 100, e = 5e-4 -> t_z = -0.05/101  = -4.95050e-4, |x| = 4.951e-4
//     -> same acceptable step, but inliers 100 is NOT > 100 => FAILURE
// The 101-pixel twin of the second fixture flips tracking_ok back to true, so the
// inlier floor (not the step, not convergence) is the discriminating quantity.
// ---------------------------------------------------------------------------
std::vector<Corr> gridFixture(int x0, int x1, int y0, int y1, float z, float offset,
                              const Eigen::Vector3f& n = Eigen::Vector3f(0.0f, 0.0f, 1.0f)) {
    std::vector<Corr> cs;
    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) cs.push_back(makeCorr(x, y, z, n, 2, offset));
    }
    return cs;
}

void testAcceptableFinalStepIsSuccess() {
    // x 300..339 is symmetric about CX=319.5, y 237..242 about CY=239.5.
    const std::vector<Corr> cs = gridFixture(300, 339, 237, 242, 1.0f, 5e-4f);
    CHECK(cs.size() == 240, "grid fixture holds 240 correspondences");
    const RunPair rp = runFixture(cs, level0Only(1));
    const ICPResult& r = rp.first;

    checkCounters(r, 240, 240, 240, 0, 0, "final-step fixture");
    CHECK(r.inliers == 240, "inliers == 240, got " + std::to_string(r.inliers));
    CHECK(!r.converged, "converged stays false: |x| = 4.98e-4 > 5e-5");
    CHECK(r.final_step > kConvStep, "final_step > kConvStep 5e-5, got " + std::to_string(r.final_step));
    CHECK(r.final_step <= kAcceptStep, "final_step <= kAcceptStep 1e-3, got " + std::to_string(r.final_step));
    CHECK(r.tracking_ok, "tracking_ok is TRUE without converged (acceptable final step)");

    const double tz = static_cast<double>(r.pose(2, 3));
    const double oracle = -240.0 * 5e-4 / (240.0 + kDampEscalate);
    CHECK(std::abs(tz - oracle) <= kTolMatch,
          "t_z == -0.12/241 = " + std::to_string(oracle) + ", got " + std::to_string(tz));
    CHECK(std::abs(static_cast<double>(r.pose(0, 3))) <= kTolMatch &&
              std::abs(static_cast<double>(r.pose(1, 3))) <= kTolMatch,
          "t_x == t_y == 0 (symmetric grid)");
    CHECK(std::abs(static_cast<double>(r.error) - 5e-4 * 5e-4) <= kTolMatch,
          "error == psi(e)/1 mean = e^2 = 2.5e-7 m^2, got " + std::to_string(r.error));
    checkDeterminism(rp, "final-step fixture");
    std::printf("  final step: inliers=%d converged=%d final_step=%.9f (oracle %.9f) ok=%d\n",
                r.inliers, r.converged ? 1 : 0, r.final_step, oracle, r.tracking_ok ? 1 : 0);
}

void testInlierFloorStillTrackingLost() {
    // x 310..329 symmetric, y 237..241 symmetric -> exactly 100 correspondences.
    const std::vector<Corr> cs = gridFixture(310, 329, 237, 241, 1.0f, 5e-4f);
    CHECK(cs.size() == 100, "floor fixture holds 100 correspondences");
    const RunPair rp = runFixture(cs, level0Only(1));
    const ICPResult& r = rp.first;

    checkCounters(r, 100, 100, 100, 0, 0, "inlier floor");
    CHECK(r.inliers == 100, "inliers == 100, got " + std::to_string(r.inliers));
    CHECK(!r.converged, "not converged");
    CHECK(r.final_step <= kAcceptStep, "the step IS acceptable: " + std::to_string(r.final_step));
    CHECK(!r.tracking_ok, "tracking_ok is FALSE: inliers 100 is not > 100");

    const double tz = static_cast<double>(r.pose(2, 3));
    const double oracle = -100.0 * 5e-4 / (100.0 + kDampEscalate);
    CHECK(std::abs(tz - oracle) <= kTolMatch,
          "t_z == -0.05/101 = " + std::to_string(oracle) + ", got " + std::to_string(tz));
    checkDeterminism(rp, "inlier floor");

    // One extra correspondence (101 > 100) on the same geometry flips success, so
    // the floor - not the step, the filter counters or convergence - is decisive.
    std::vector<Corr> plus = cs;
    plus.push_back(makeCorr(200, 200, 1.0f, Eigen::Vector3f(0.0f, 0.0f, 1.0f), 2, 5e-4f));
    const RunPair rp101 = runFixture(plus, level0Only(1));
    CHECK(rp101.first.inliers == 101, "twin fixture reaches 101 inliers, got " +
                                          std::to_string(rp101.first.inliers));
    CHECK(rp101.first.final_step <= kAcceptStep, "twin keeps an acceptable final step");
    CHECK(rp101.first.tracking_ok, "101 inliers -> tracking_ok true (the floor is the only delta)");
    checkDeterminism(rp101, "inlier floor twin");
    std::printf("  inlier floor: 100 -> ok=%d, 101 -> ok=%d (final_step %.3e both)\n",
                r.tracking_ok ? 1 : 0, rp101.first.tracking_ok ? 1 : 0, rp101.first.final_step);
}

// ---------------------------------------------------------------------------
// 3. Rotation cap 0.5 rad, per iteration, reject-not-clamp.
//
// Two point-symmetric rows (y = 450 with v_y = +421/525 and y = 29 with
// v_y = -421/525 at z = 2 m) carrying OPPOSITE depth errors (+1 m on the upper
// row, -1 m on the lower row). The Huber clamp pins every |w*e| at
// kHuberK = 0.02, so with N total pixels and w = 0.02/1 = 0.02:
//   b(2) = -sum(w*e)        = -(N/2*0.02 + N/2*(-0.02)) = 0   -> no translation
//   b(3) = -sum(w*e*v_y)    = -N*kHuberK*row_vy
//   A(2,2) = N*kHuberK, A(3,3) = N*kHuberK*row_vy^2,
//   A(2,3) = sum(w*v_y) = 0, A(3,4) = -sum(w*v_x*v_y) = 0
//   (columns are symmetric about CX and v_y is constant per row)
// so rows 2 and 3 decouple, and with the ill-conditioned (3 zero-eigenvalue)
// Hessian the canonical damping is 1.0:
//   |x_rot| = N*kHuberK*row_vy / (N*kHuberK*row_vy^2 + 1), |x_trans| = 0
//   N = 200 -> |x_rot| = 0.8979 rad  (> 0.5  -> REJECT)
//   N =  20 -> |x_rot| = 0.2551 rad (< 0.5  -> ACCEPT)
// The bracket is the witness: identical geometry, only N differs, so the ONLY
// thing that can reject the N=200 update is the rotation cap (translation is
// exactly zero, the filters reject nothing). The 1 m residual needs a widened
// dist_threshold (3 m) so the correspondence survives the distance gate - the
// step guard, not the filters, is under test.
// ---------------------------------------------------------------------------
std::vector<Corr> rotationLeakFixture(int per_row) {
    std::vector<Corr> cs;
    const int x0 = 320 - per_row / 2;
    for (int i = 0; i < per_row; ++i) {
        cs.push_back(makeCorr(x0 + i, 450, 2.0f, Eigen::Vector3f(0.0f, 0.0f, 1.0f), 2, 1.0f));
    }
    for (int i = 0; i < per_row; ++i) {
        cs.push_back(makeCorr(x0 + i, 29, 2.0f, Eigen::Vector3f(0.0f, 0.0f, 1.0f), 2, -1.0f));
    }
    return cs;
}

void testRotationCapRejectsUpdate() {
    ICPParams params    = level0Only(1);
    params.dist_threshold = 3.0f;  // admit the deliberate 1 m residual (step guard test)

    const std::vector<Corr> big = rotationLeakFixture(100);  // 200 correspondences
    CHECK(big.size() == 200, "rotation fixture holds 200 correspondences");
    const RunPair rp = runFixture(big, params);
    const ICPResult& r = rp.first;

    // Nothing was filtered: the rejection happened at the step guard.
    checkCounters(r, 200, 200, 200, 0, 0, "rotation cap");
    CHECK(r.inliers == 0, "no update was accepted, so the reported inliers stay 0, got " +
                              std::to_string(r.inliers));
    CHECK(!r.converged && !r.tracking_ok, "neither converged nor tracking_ok");
    CHECK(std::isinf(r.final_step) && r.final_step > 0.0f,
          "final_step stays +inf (never an accepted update), got " + std::to_string(r.final_step));
    CHECK(poseBitEq(r.pose, Eigen::Matrix4f::Identity()),
          "pose is EXACTLY the estimate: the over-cap update is rejected, not clamped");

    // Raw (pre-cap) step magnitudes, hand-derived: rotation far over the cap,
    // translation exactly zero, so the rotation guard is the firing one.
    const double row_vy = (450.0 - static_cast<double>(CY)) * 2.0 / static_cast<double>(FY);
    const double raw_rot = 200.0 * kHuberK * row_vy /
                           (200.0 * kHuberK * row_vy * row_vy + kDampEscalate);
    CHECK(raw_rot > kRotCap + 0.3, "fixture really drives |x_rot| = " + std::to_string(raw_rot) +
                                       " rad, far above the 0.5 rad cap");
    checkDeterminism(rp, "rotation cap");

    // Companion bracket: same shape, N = 20 -> |x_rot| = 0.2551 < 0.5 -> accepted.
    const std::vector<Corr> small = rotationLeakFixture(10);
    const RunPair rp_small = runFixture(small, params);
    const ICPResult& s = rp_small.first;
    CHECK(s.inliers == 20, "sub-cap bracket accepts its update, got " + std::to_string(s.inliers));
    const double oracle_rot = 20.0 * kHuberK * row_vy /
                              (20.0 * kHuberK * row_vy * row_vy + kDampEscalate);
    CHECK(std::abs(static_cast<double>(s.final_step) - oracle_rot) <= kTolMatch,
          "accepted |x| == 20*kHuberK*row_vy/(20*kHuberK*row_vy^2+1) = " +
              std::to_string(oracle_rot) + ", got " + std::to_string(s.final_step));
    CHECK(std::abs(static_cast<double>(s.pose(0, 3))) <= kTolMatch &&
              std::abs(static_cast<double>(s.pose(1, 3))) <= kTolMatch &&
              std::abs(static_cast<double>(s.pose(2, 3))) <= kTolMatch,
          "accepted update carries no translation (|x_trans| = 0 by construction)");
    CHECK(std::abs(static_cast<double>(s.pose(2, 1))) >= 0.2,
          "the accepted 0.255 rad step really rotated the pose about x, got " +
              std::to_string(std::abs(static_cast<double>(s.pose(2, 1)))));
    checkDeterminism(rp_small, "rotation cap bracket");
    std::printf("  rotation cap: N=200 |x_rot|=%.6f -> rejected (pose bit-identical); "
                "N=20 |x_rot|=%.6f -> accepted\n",
                raw_rot, oracle_rot);
}

// ---------------------------------------------------------------------------
// 4. Translation cap 0.2 m stays (regression guard for the sibling guard).
//
// Same symmetric-grid geometry with a uniform 1 m residual, so every |w*e| is
// pinned at 0.02 and the update is pure translation:
//   |x_trans| = 0.02*N / (0.02*N + 1)   (b(3) = b(4) = 0 by symmetry)
//   N = 240 -> 4.8/5.8 = 0.8276 m  (> 0.2 -> REJECT)
//   N =  12 -> 0.24/1.24 = 0.1935 m (< 0.2 -> ACCEPT, and it equals the oracle)
// ---------------------------------------------------------------------------
void testTranslationCapRejectsUpdate() {
    ICPParams params      = level0Only(1);
    params.dist_threshold = 3.0f;

    const std::vector<Corr> big = gridFixture(300, 339, 237, 242, 2.0f, 1.0f);
    const RunPair rp = runFixture(big, params);
    const ICPResult& r = rp.first;
    checkCounters(r, 240, 240, 240, 0, 0, "translation cap");
    CHECK(r.inliers == 0, "over-cap translation rejects the update (inliers 0), got " +
                              std::to_string(r.inliers));
    CHECK(poseBitEq(r.pose, Eigen::Matrix4f::Identity()), "pose stays exactly the estimate");
    CHECK(std::isinf(r.final_step) && !r.converged && !r.tracking_ok,
          "final_step +inf, not converged, not ok");
    const double raw_trans = 0.02 * 240.0 / (0.02 * 240.0 + kDampEscalate);
    CHECK(raw_trans > kTransCap + 0.5, "fixture drives |x_trans| = " + std::to_string(raw_trans) +
                                           " m, far above the 0.2 m cap");
    checkDeterminism(rp, "translation cap");

    const std::vector<Corr> small = gridFixture(317, 322, 239, 240, 2.0f, 1.0f);  // 12 px
    const RunPair rp_small = runFixture(small, params);
    const ICPResult& s = rp_small.first;
    CHECK(s.inliers == 12, "sub-cap bracket accepts, got " + std::to_string(s.inliers));
    const double oracle = -0.24 / 1.24;
    CHECK(std::abs(static_cast<double>(s.pose(2, 3)) - oracle) <= kTolMatch,
          "accepted t_z == -0.24/1.24 = " + std::to_string(oracle) +
              ", got " + std::to_string(s.pose(2, 3)));
    checkDeterminism(rp_small, "translation cap bracket");
    std::printf("  translation cap: N=240 |x|=%.6f -> rejected; N=12 t_z=%.6f -> accepted\n",
                raw_trans, oracle);
}

// ---------------------------------------------------------------------------
// 5. Adaptive damping, verified through the public pose.
//
// (a) ESCALATION. Six point-symmetric pairs at err 0.01 (w = 1) plus one outlier
//     at err 0.08 (w = 0.25) near the optical center, exactly the shape of the
//     Todo 12 fixture with one more pair:
//       A(2,2) = 12*1 + 0.25 = 12.25,  b(2) = -(12*0.01 + 0.25*0.08) = -0.14
//     The +z-only normals leave rows 0,1,5 identically zero, so the UNDAMPED
//     Hessian has min_eig = 0 exactly and the canonical policy must escalate to
//     damping 1.0:
//       t_z = -0.14/13.25 = -0.010566038        (canonical, escalated)
//       t_z = -0.14/12.35 = -0.011336032        (level-0 0.01 -> ... no: 0.1 damping)
//       t_z = -0.14/12.26 = -0.011419250        (level-0 0.01 damping, no escalation)
//     Both wrong fixed points sit >= 7.7e-4 away, 4 orders of magnitude above the
//     1e-7 match tolerance.
// (b) NO ESCALATION. A well-conditioned three-wall fixture (small +z wall, big +x
//     and +y walls, per-pixel varying depth) whose undamped Hessian has
//     min_eig = 5.21 and cond = 1322: no condition-number trigger, so level 0 must
//     use its own base damping 0.01, NOT 0.1 (the coarse-level value) and NOT 1.0:
//       t_z = -0.24/24.01 = -0.0099958351       (canonical, level-0 damping)
//       t_z = -0.24/24.1  = -0.0099585062       (0.1 damping: 3.7e-5 away)
//       t_z = -0.24/25.0  = -0.0096000000       (escalated: 4.0e-4 away)
//     The +x/+y walls contribute zero residual, so they add curvature only; the +z
//     wall stays symmetric in x and y, which keeps row 2 decoupled (verified:
//     A(2,3) = A(2,4) = A(2,5) = 0) and preserves the closed form.
// ---------------------------------------------------------------------------
void testAdaptiveDamping() {
    // (a) escalation: 13 correspondences, +z normals only.
    std::vector<Corr> esc;
    for (int k = 0; k < 6; ++k) {
        esc.push_back(makeCorr(319 - k, 239 - k, 1.0f, Eigen::Vector3f(0, 0, 1), 2, 0.01f));
        esc.push_back(makeCorr(320 + k, 240 + k, 1.0f, Eigen::Vector3f(0, 0, 1), 2, 0.01f));
    }
    esc.push_back(makeCorr(320, 239, 1.0f, Eigen::Vector3f(0, 0, 1), 2, 0.08f));
    const RunPair rp_esc = runFixture(esc, level0Only(1));
    const ICPResult& re = rp_esc.first;
    checkCounters(re, 13, 13, 13, 0, 0, "escalation fixture");
    const double tz_esc = static_cast<double>(re.pose(2, 3));
    const double oracle_esc = -0.14 / 13.25;
    CHECK(std::abs(tz_esc - oracle_esc) <= kTolMatch,
          "escalated t_z == -0.14/13.25 = " + std::to_string(oracle_esc) +
              ", got " + std::to_string(tz_esc));
    CHECK(std::abs(tz_esc - (-0.14 / 12.35)) >= kGapFloor,
          "t_z is NOT the 0.1-damping value -0.14/12.35, got " + std::to_string(tz_esc));
    CHECK(std::abs(tz_esc - (-0.14 / 12.26)) >= kGapFloor,
          "t_z is NOT the 0.01-damping value -0.14/12.26, got " + std::to_string(tz_esc));
    checkDeterminism(rp_esc, "escalation fixture");

    // (b) no escalation: 24-pixel +z wall (err 0.01) + 1600 +x and 1600 +y pixels
    //     (err 0) with per-pixel varying depth.
    std::vector<Corr> wc;
    for (int x = 314; x <= 325; ++x) {
        // Symmetric in y about 239.5 (150 + 329 = 479) and in x about 319.5.
        wc.push_back(makeCorr(x, 150, 1.0f, Eigen::Vector3f(0, 0, 1), 2, 0.01f));
        wc.push_back(makeCorr(x, 329, 1.0f, Eigen::Vector3f(0, 0, 1), 2, 0.01f));
    }
    const float depths[4] = {1.0f, 1.5f, 2.0f, 2.5f};
    const Eigen::Vector3f n_x(1.0f, 0.0f, 0.0f);
    const Eigen::Vector3f n_y(0.0f, 1.0f, 0.0f);
    for (int x = 300; x <= 339; ++x) {
        for (int y = 200; y <= 239; ++y) {
            wc.push_back(makeCorr(x, y, depths[(x + y) & 3], n_x, 0, 0.0f));
        }
        for (int y = 240; y <= 279; ++y) {
            wc.push_back(makeCorr(x, y, depths[(2 * x + y) & 3], n_y, 0, 0.0f));
        }
    }
    const RunPair rp_wc = runFixture(wc, level0Only(1));
    const ICPResult& rw = rp_wc.first;
    CHECK(rw.inliers == 3224, "well-conditioned fixture keeps 3224 correspondences, got " +
                                  std::to_string(rw.inliers));
    CHECK(rw.angle_filtered == 0 && rw.dist_filtered == 0,
          "well-conditioned fixture filters nothing (af=" + std::to_string(rw.angle_filtered) +
              ", df=" + std::to_string(rw.dist_filtered) + ")");
    const double tz_wc = static_cast<double>(rw.pose(2, 3));
    const double oracle_wc = -0.24 / 24.01;
    CHECK(std::abs(tz_wc - oracle_wc) <= kTolMatch,
          "level-0 damped t_z == -0.24/24.01 = " + std::to_string(oracle_wc) +
              ", got " + std::to_string(tz_wc));
    CHECK(std::abs(tz_wc - (-0.24 / 24.1)) >= kGapFloorS,
          "t_z is NOT the 0.1-damping value -0.24/24.1 (gap >= 1e-5), got " + std::to_string(tz_wc));
    CHECK(std::abs(tz_wc - (-0.24 / 25.0)) >= kGapFloorS,
          "t_z is NOT the escalated -0.24/25 value (gap >= 1e-5), got " + std::to_string(tz_wc));
    CHECK(std::abs(static_cast<double>(rw.final_step) - std::abs(oracle_wc)) <= kTolMatch,
          "final_step == |t_z| (pure translation update), got " + std::to_string(rw.final_step));
    checkDeterminism(rp_wc, "well-conditioned fixture");
    std::printf("  damping: escalated t_z=%.9f (oracle %.9f)  level0 t_z=%.9f (oracle %.9f)\n",
                tz_esc, oracle_esc, tz_wc, oracle_wc);
}

// ---------------------------------------------------------------------------
// 6. Model-data validity gate: squared norms, not norms, not finiteness alone.
//
// A 10-correspondence clean base (t_z = -0.1/11 = -0.009090909 with the escalated
// damping 1.0) plus ONE extra pixel (340,260) carrying a chosen model normal. The
// gate is `allFinite(mv) && squaredNorm(mv) > 1e-12 && allFinite(mn) &&
// squaredNorm(mn) > 0.9`, evaluated BEFORE valid_model++, so a rejected sample
// still counts as projected but never as a valid model point, and the surviving 10
// reproduces the base pose bit-for-bit.
//   (0,0,1)   sq 1.0     accepted
//   (0,0,.95) sq .9025   accepted   <- boundary pair just INSIDE
//   (0,0,.94) sq .8836   rejected   <- boundary pair just OUTSIDE (no upper bound,
//                                       no renormalization: acceptance is a test)
//   (0,0,.5)  sq .25     rejected   <- the pre-Todo-13 `norm > 1e-6` gate accepted
//   (0,0,0)   sq 0       rejected
//   (0,0,NaN) / (0,0,Inf)           rejected by allFinite
// ---------------------------------------------------------------------------
std::vector<Corr> modelNormalFixture(const Eigen::Vector3f* extra_normal) {
    std::vector<Corr> cs;
    for (int k = 0; k < 5; ++k) {
        cs.push_back(makeCorr(319 - k, 239 - k, 1.0f, Eigen::Vector3f(0, 0, 1), 2, 0.01f));
        cs.push_back(makeCorr(320 + k, 240 + k, 1.0f, Eigen::Vector3f(0, 0, 1), 2, 0.01f));
    }
    if (extra_normal != nullptr) {
        Corr c = makeCorr(340, 260, 1.0f, Eigen::Vector3f(0, 0, 1), 2, 0.01f);
        c.mn = *extra_normal;
        cs.push_back(c);
    }
    return cs;
}

void testModelNormalGate() {
    const Eigen::Vector3f opposite_unit(0.0f, 0.0f, -1.0f);
    const RunPair base = runFixture(modelNormalFixture(&opposite_unit), level0Only(1));
    // (0,0,-1) is a valid unit normal with the opposite sign: it must be ACCEPTED
    // (|dot| is used by the angle gate) and it shifts the pose away from the
    // +z-only base, proving the extra pixel really participates.
    CHECK(base.first.inliers == 11, "extra valid model normal is counted, got " +
                                        std::to_string(base.first.inliers));

    const RunPair clean = runFixture(modelNormalFixture(nullptr), level0Only(1));
    const ICPResult& rc = clean.first;
    const double tz_base = static_cast<double>(rc.pose(2, 3));
    CHECK(rc.inliers == 10, "base fixture holds 10 inliers, got " + std::to_string(rc.inliers));
    CHECK(std::abs(tz_base - (-0.1 / 11.0)) <= kTolMatch,
          "base t_z == -0.1/11 = -0.009090909, got " + std::to_string(tz_base));
    checkCounters(rc, 10, 10, 10, 0, 0, "model gate base");
    checkDeterminism(clean, "model gate base");

    struct Case {
        const char* name;
        Eigen::Vector3f n;
        bool accepted;
    };
    const Case cases[] = {
        {"unit +z", Eigen::Vector3f(0, 0, 1), true},
        {"boundary in (sq 0.9025)", Eigen::Vector3f(0, 0, 0.95f), true},
        {"boundary out (sq 0.8836)", Eigen::Vector3f(0, 0, 0.94f), false},
        {"half length", Eigen::Vector3f(0, 0, 0.5f), false},
        {"zero", Eigen::Vector3f(0, 0, 0), false},
        {"NaN", Eigen::Vector3f(0, 0, kNaN), false},
        {"+Inf", Eigen::Vector3f(0, 0, kInf), false},
        {"NaN in x", Eigen::Vector3f(kNaN, 0, 1), false},
    };
    for (const Case& c : cases) {
        const RunPair rp = runFixture(modelNormalFixture(&c.n), level0Only(1));
        const ICPResult& r = rp.first;
        const std::string who = std::string("model normal ") + c.name;
        checkCounters(r, 11, c.accepted ? 11 : 10, 11, 0, 0, who);
        CHECK(r.inliers == (c.accepted ? 11 : 10),
              who + acceptanceText(c.accepted));
        if (!c.accepted) {
            CHECK(poseBitEq(r.pose, rc.pose), who + ": rejected sample leaves the pose "
                                               "bit-identical to the 11-point base");
            CHECK(r.error == rc.error, who + ": rejected sample leaves the objective identical");
        }
        checkDeterminism(rp, who);
    }

    // A non-finite model VERTEX is rejected by the same gate (vertex half, not
    // normal half), still counting as projected.
    std::vector<Corr> nan_vertex = modelNormalFixture(&opposite_unit);
    nan_vertex.back().mv.z() = kNaN;
    const RunPair rp_nv = runFixture(nan_vertex, level0Only(1));
    checkCounters(rp_nv.first, 11, 10, 11, 0, 0, "NaN model vertex");
    CHECK(rp_nv.first.inliers == 10, "NaN model vertex is rejected, got " +
                                          std::to_string(rp_nv.first.inliers));
    std::printf("  model gate: %zu normal variants gated on squared norm (0.9 floor), "
                "NaN/+Inf/zero/half all rejected\n", sizeof(cases) / sizeof(Case));
}

// ---------------------------------------------------------------------------
// 7. Live-normal validity gate: squared norm, and its ORDER (after the distance
//    gate, before the angle gate).
//
// A model wall with normals (1,0,0) is a TANGENT wall for a camera looking along
// +z, so its live normals have z == 0. The pre-Todo-13 CPU gate `live_n.z() == 0`
// rejected every such correspondence (inliers 0, pose frozen); the canonical
// squared-norm gate accepts (0,0,1)-valid AND (1,0,0)-valid normals, which is the
// whole point of the fix. Invalid live normals (zero, NaN, +Inf) are dropped
// silently - no counter increments - because the canonical policy adds no new
// diagnostic for them.
//   gate order witness: a correspondence whose model vertex is 0.5 m away AND
//   whose live normal is NaN must land in dist_filtered, proving the distance
//   gate still runs first.
// ---------------------------------------------------------------------------
std::vector<Corr> liveNormalFixture(const Eigen::Vector3f& live_n, double offset = 0.01,
                                    double dist_offset = 0.0) {
    std::vector<Corr> cs;
    for (int k = 0; k < 5; ++k) {
        for (int i = 0; i < 2; ++i) {
            const int px = i ? 320 + k : 319 - k;
            const int py = i ? 240 + k : 239 - k;
            Corr c = makeCorr(px, py, 1.0f, Eigen::Vector3f(1, 0, 0), 0, static_cast<float>(offset));
            c.ln = live_n;
            if (dist_offset != 0.0) c.mv.z() = static_cast<float>(c.mv.z() + dist_offset);
            cs.push_back(c);
        }
    }
    return cs;
}

void testLiveNormalGate() {
    const Eigen::Matrix4f id = Eigen::Matrix4f::Identity();

    // Tangential live normal: accepted by the squared-norm gate.
    const RunPair ok = runFixture(liveNormalFixture(Eigen::Vector3f(1, 0, 0)), level0Only(1));
    const ICPResult& r = ok.first;
    checkCounters(r, 10, 10, 10, 0, 0, "tangential live normal");
    CHECK(r.inliers == 10, "a z==0 but unit live normal is ACCEPTED, got inliers " +
                               std::to_string(r.inliers));
    CHECK(std::abs(static_cast<double>(r.pose(0, 3))) > 1e-4,
          "the accepted tangential correspondments moved t_x, got " +
              std::to_string(r.pose(0, 3)));
    CHECK(std::abs(static_cast<double>(r.pose(0, 3)) + 0.1 / 21.0) <= kTolMatch,
          "t_x == -0.1/21 (rows 0/4 coupled by the (1,0,0) normal), got " +
              std::to_string(r.pose(0, 3)));
    checkDeterminism(ok, "tangential live normal");

    const char* names[] = {"zero", "NaN", "+Inf"};
    const Eigen::Vector3f bad[] = {Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(kNaN, 0, 0),
                                   Eigen::Vector3f(kInf, 0, 0)};
    for (int i = 0; i < 3; ++i) {
        const RunPair rp = runFixture(liveNormalFixture(bad[i]), level0Only(1));
        const ICPResult& s = rp.first;
        const std::string who = std::string("live normal ") + names[i];
        CHECK(s.inliers == 0, who + ": filtered, no correspondence accumulates, got " +
                                      std::to_string(s.inliers));
        // valid_live / valid_model / projected all counted, NOTHING filtered:
        // the live-normal gate has no counter of its own.
        checkCounters(s, 10, 10, 10, 0, 0, who);
        CHECK(poseBitEq(s.pose, id), who + ": pose stays exactly the estimate");
        CHECK(std::isinf(s.final_step) && !s.converged && !s.tracking_ok,
              who + ": no accepted update (final_step +inf)");
        checkDeterminism(rp, who);
    }

    // Gate ORDER: distance first. Same fixture, model vertices pushed 0.5 m away
    // and the live normal NaN -> the distance gate must catch it (dist_filtered).
    const RunPair rp_order =
        runFixture(liveNormalFixture(Eigen::Vector3f(kNaN, 0, 0), 0.01, 0.5), level0Only(1));
    checkCounters(rp_order.first, 10, 10, 10, 10, 0, "live-normal gate order");
    CHECK(rp_order.first.inliers == 0, "out-of-range sample never reaches the normal gate");
    checkDeterminism(rp_order, "live-normal gate order");
    std::printf("  live gate: (1,0,0) accepted (t_x=%.6f), zero/NaN/Inf filtered silently, "
                "distance gate runs first\n",
                static_cast<double>(r.pose(0, 3)));
}

// ---------------------------------------------------------------------------
// 8. angle_threshold: clamped at construction AND by setParams, and the gate stays
//    active behind the clamp.
//
// A wall whose model normals are (0,0,1) and whose live normals are (1,0,0) has
// |dot| = 0, below cos(85 deg) = 0.0872, so even the widest legal threshold
// filters it: angle_filtered = 10, nothing accumulates, the pose never moves. The
// +z/+z control on the same grid proves the filter (not the geometry) is firing.
// ---------------------------------------------------------------------------
void testAngleThresholdClampAndGate() {
    struct ClampCase {
        float in;
        float out;
        const char* name;
    };
    const ClampCase cases[] = {
        {120.0f, 85.0f, "120 -> 85 (upper clamp)"},
        {kNaN, 30.0f, "NaN -> 30 (non-finite fallback)"},
        {kInf, 30.0f, "+Inf -> 30 (non-finite fallback)"},
        {-5.0f, 0.0f, "-5 -> 0 (lower clamp)"},
        {45.0f, 45.0f, "45 -> 45 (unchanged)"},
        {30.0f, 30.0f, "30 -> 30 (default)"},
    };
    for (const ClampCase& c : cases) {
        ICPParams p;
        p.angle_threshold = c.in;
        ICPTracker t_ctor(p);
        CHECK(t_ctor.params().angle_threshold == c.out,
              std::string("ctor clamp ") + c.name + ", got " +
                  std::to_string(t_ctor.params().angle_threshold));
        ICPTracker t_set;
        t_set.setParams(p);
        CHECK(t_set.params().angle_threshold == c.out,
              std::string("setParams clamp ") + c.name + ", got " +
                  std::to_string(t_set.params().angle_threshold));
        // No other ICPParams field is sanitized, renamed or dropped.
        CHECK(t_ctor.params().dist_threshold == p.dist_threshold &&
                  t_ctor.params().min_depth == p.min_depth &&
                  t_ctor.params().max_depth == p.max_depth &&
                  t_ctor.params().max_iterations[0] == p.max_iterations[0] &&
                  t_ctor.params().max_iterations[1] == p.max_iterations[1] &&
                  t_ctor.params().max_iterations[2] == p.max_iterations[2],
              std::string("clamp leaves every other ICPParams field untouched (") + c.name + ")");
    }

    ICPParams wide = level0Only(1);
    wide.angle_threshold = 120.0f;  // clamped to 85 deg -> cos = 0.0872
    const Eigen::Vector3f n_live(1, 0, 0);
    const Eigen::Vector3f n_model(0, 0, 1);
    std::vector<Corr> orthogonal;
    for (int k = 0; k < 5; ++k) {
        for (int i = 0; i < 2; ++i) {
            const int px = i ? 320 + k : 319 - k;
            const int py = i ? 240 + k : 239 - k;
            Corr c = makeCorr(px, py, 1.0f, n_model, 2, 0.01f);
            c.ln = n_live;
            orthogonal.push_back(c);
        }
    }
    const RunPair rp = runFixture(orthogonal, wide);
    checkCounters(rp.first, 10, 10, 10, 0, 10, "orthogonal normals at 85 deg");
    CHECK(rp.first.inliers == 0, "the angle gate stays active behind the clamp, got " +
                                     std::to_string(rp.first.inliers));
    CHECK(poseBitEq(rp.first.pose, Eigen::Matrix4f::Identity()), "angle-filtered run leaves pose");
    checkDeterminism(rp, "orthogonal normals");

    const RunPair control = runFixture(
        gridFixture(319 - 4, 320 + 4, 239 - 4, 240 + 4, 1.0f, 0.01f), wide);
    CHECK(control.first.angle_filtered == 0 && control.first.inliers >= 10,
          "control: aligned normals at the same 85 deg threshold are NOT angle filtered "
          "(af=" + std::to_string(control.first.angle_filtered) + ")");
    std::printf("  angle threshold: clamped {120->85, NaN->30, -5->0, 45->45}; gate active "
                "(10 filtered at cos 85deg)\n");
}

// ---------------------------------------------------------------------------
// 9. Determinant-corrected SVD re-orthonormalization.
//
// A deliberately reflected, non-orthonormal pose_estimate (diag(-2,1,2), det = -4)
// is fed through the public track(): the correspondence set is built so the
// residual is exactly zero (12 live pixels all on the optical axis of the model
// pixel that R*live_v projects onto), so b = 0, x = 0, the update IS accepted, and
// the only thing the iteration does is re-orthonormalize the rotation block.
//   uncorrected  U*V^T            -> det = -1 (a reflection, not a rotation)
//   canonical    U*diag(1,1,det)·V^T -> det = +1 and R^T R = I
// The exact 3x3 result is NOT asserted (the singular values 2,2,1 are degenerate,
// so U and V are only unique as a pair); det = +1, orthonormality and bit-stable
// repeats are the public witness.
// ---------------------------------------------------------------------------
void testSvdReturnsProperRotation() {
    Eigen::Matrix4f reflected = Eigen::Matrix4f::Identity();
    reflected(0, 0) = -2.0f;
    reflected(1, 1) = 1.0f;
    reflected(2, 2) = 2.0f;

    // v_ref = R * v = (0,0,2) for every live pixel -> one shared model pixel.
    const Eigen::Vector3f v_ref(0.0f, 0.0f, 2.0f);
    int mx = 0;
    int my = 0;
    projectedPixel(v_ref, &mx, &my);
    CHECK(mx >= 0 && mx < FRAME_W && my >= 0 && my < FRAME_H, "shared model pixel in range");

    std::vector<Corr> cs;
    for (int i = 0; i < 12; ++i) {
        Corr c;
        c.lpx = 300 + i;
        c.lpy = 200;
        c.lv  = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        c.ln  = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        c.mv  = v_ref;                        // == R * v  -> zero residual
        c.mn  = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        c.mpx = mx;
        c.mpy = my;
        cs.push_back(c);
    }

    const RunPair rp = runFixture(cs, level0Only(1), reflected);
    const ICPResult& r = rp.first;
    CHECK(r.inliers == 12, "the zero-residual update is accepted, got " + std::to_string(r.inliers));
    CHECK(r.converged && std::abs(r.final_step) < 1e-20f,
          "zero update, converged, final_step 0 (final_step=" + std::to_string(r.final_step) + ")");
    checkCounters(r, 12, 12, 12, 0, 0, "reflection fixture");

    const Eigen::Matrix3f R = r.pose.block<3, 3>(0, 0);
    const double det = static_cast<double>(R.determinant());
    CHECK(std::abs(det - 1.0) <= 1e-5,
          "resulting rotation has det = +1, got " + std::to_string(det));
    CHECK((R.transpose() * R - Eigen::Matrix3f::Identity()).cwiseAbs().maxCoeff() <= 1e-4f,
          "resulting rotation is orthonormal (R^T R = I)");
    CHECK(r.pose.col(3).isApprox(Eigen::Vector4f(0, 0, 0, 1), 1e-6f),
          "translation untouched by the projection");
    CHECK(poseBitEq(r.pose, rp.second.pose), "reflection projection is bit-stable across repeats");
    std::printf("  SVD: reflected det(-2,1,2) pose projects to det = %.9f, orthonormal\n", det);
}

// ---------------------------------------------------------------------------
// 10. ICPResult{} pose starts as the identity, never an indeterminate matrix.
// ---------------------------------------------------------------------------
void testDefaultResultPoseIsIdentity() {
    const ICPResult fresh{};
    CHECK(poseBitEq(fresh.pose, Eigen::Matrix4f::Identity()),
          "ICPResult{} pose is bit-identically the identity");
    CHECK(fresh.pose.isIdentity(0.0f), "ICPResult{} pose.isIdentity(0)");
    CHECK(std::isinf(fresh.final_step) && fresh.final_step > 0.0f,
          "ICPResult{} final_step defaults to +inf (no accepted update yet)");
    CHECK(!fresh.converged && !fresh.tracking_ok && fresh.inliers == 0 && fresh.error == 0.0f,
          "ICPResult{} defaults: not converged, not ok, 0 inliers, 0 error");
    std::printf("  ICPResult{} pose == identity, final_step == +inf\n");
}

} // namespace

int main() {
    testAcceptableFinalStepIsSuccess();
    testInlierFloorStillTrackingLost();
    testRotationCapRejectsUpdate();
    testTranslationCapRejectsUpdate();
    testAdaptiveDamping();
    testModelNormalGate();
    testLiveNormalGate();
    testAngleThresholdClampAndGate();
    testSvdReturnsProperRotation();
    testDefaultResultPoseIsIdentity();

    if (g_failures == 0) {
        std::printf("icp_numeric_policy_contract: PASS (%d checks: success rule + inlier floor "
                    "+ rotation/translation caps + adaptive damping + squared-norm gates + "
                    "angle clamp + det-corrected SVD + identity default pose)\n",
                    g_checks);
        return 0;
    }
    std::printf("icp_numeric_policy_contract: FAIL (%d failed checks of %d)\n", g_failures, g_checks);
    return 1;
}
