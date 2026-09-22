// tsdf_integration_min_depth_contract (big-fix Todo 15): CPU-only contract that TSDF
// integration honors the CONFIGURED near depth bound. Public API only (TSDFVolume
// ctor/params/integrate/voxelData/voxelAt + the FusionHyperparams sync helper): no
// device, display, GPU, sensor, thread timing, sleep or filesystem. Expected values are
// hand-derived doubles from the fixture geometry, never product output. Locks:
//   A  the configured min_depth is the pixel gate: a frame whose only valid pixel is
//      nearer than min_depth leaves the volume in the canonical empty state, while the
//      identical frame with a smaller min_depth integrates
//   B  the configured min_depth also raises the march's near clamp, so the first march
//      step moves and the voxel it folds is a different hand-derived number
//      (the historical hard-coded 0.1f floor gives both volumes the SAME value)
//   C  max_depth still gates, and non-finite depth still gates
//   D  TSDFParams carries the band with the documented defaults, and the single
//      hyperparameter owner propagates it (no second store)
//
// Fixture: 64^3 volume, 10 mm voxels, truncation 25 mm, origin (-0.32,-0.32,0). A 4x4
// frame whose ONLY valid pixel is (2,2) with cx=cy=2, fx=fy=500, so that pixel's ray is
// exactly (0,0,1): its march sample at parameter t sits at world (0,0,t) and
// sdf = D - t. The march step is voxel_size*0.75 = 7.5 mm.
//   D = 0.50125, trunc = 0.025  ->  t_max = 0.52625
//   min_depth 0.1    -> t_min = max(0.1,    0.47625) = 0.47625 -> z/vs = 47.625
//   min_depth 0.4775 -> t_min = max(0.4775, 0.47625) = 0.47750 -> z/vs = 47.750
// Both first steps land in voxel (32,32,47) (v.x = v.y = 0.32/0.01 = 32) and each voxel
// 47 gets exactly ONE candidate, so its folded value is the single-candidate blend
//   tsdf = (EMPTY_TSDF*0 + tsdf_new*1) / (0 + 1 + 1e-6),  weight = min(0+1, 128) = 1
//   tsdf_new(A) = min(1, (0.50125-0.47625)/0.025) = min(1, 1.0)  = 1.0
//   tsdf_new(C) =        (0.50125-0.47750)/0.025  = 0.02375/0.025 = 0.95
#include "app/FusionHyperparams.h"
#include "tsdf/TSDFVolume.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "tsdf_integration_min_depth_contract must be compiled with AZU_PIPELINE_TEST_SEAM"
#endif

namespace {

using kfusion::app::FusionHyperparams;
using kfusion::app::syncTsdfDepthFromRange;
using kfusion::tsdf::EMPTY_COLOR;
using kfusion::tsdf::EMPTY_TSDF;
using kfusion::tsdf::EMPTY_WEIGHT;
using kfusion::tsdf::TSDFParams;
using kfusion::tsdf::TSDFVolume;
using kfusion::tsdf::Voxel;

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, what)                                                           \
    do {                                                                            \
        ++g_checks;                                                                 \
        if (!(cond)) {                                                              \
            std::printf("FAIL: %s  [%s:%d]\n", std::string(what).c_str(), __FILE__, \
                        __LINE__);                                                  \
            ++g_failures;                                                           \
        }                                                                           \
    } while (false)

constexpr int   kRes    = 64;
constexpr float kVs     = 0.01f;
constexpr float kTrunc  = 0.025f;
constexpr float kDepth  = 0.50125f;          // the one valid pixel's Z-depth
constexpr int   kImgW   = 4, kImgH = 4;
constexpr int   kPx     = 2, kPy   = 2;      // on-axis pixel
constexpr float kFx     = 500.0f, kFy = 500.0f;
constexpr float kCx     = 2.0f, kCy = 2.0f;
constexpr uint8_t kR = 200, kG = 100, kB = 40;

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();

TSDFParams fixtureParams() {
    TSDFParams p;
    p.resolution = kRes;
    p.voxel_size = kVs;
    p.truncation = kTrunc;
    p.max_weight = 128.0f;
    p.origin     = Eigen::Vector3f(-0.32f, -0.32f, 0.0f);
    return p;
}

// One frame, one valid on-axis pixel; every other pixel is depth 0 (invalid).
struct Frame {
    std::vector<float>   depth;
    std::vector<uint8_t> rgb;
};
Frame makeFrame(float depth_value) {
    Frame f;
    f.depth.assign(static_cast<size_t>(kImgW) * kImgH, 0.0f);
    f.depth[static_cast<size_t>(kPy) * kImgW + kPx] = depth_value;
    f.rgb.assign(static_cast<size_t>(kImgW) * kImgH * 3, 0);
    for (size_t i = 0; i < static_cast<size_t>(kImgW) * kImgH; ++i) {
        f.rgb[i * 3 + 0] = kR;
        f.rgb[i * 3 + 1] = kG;
        f.rgb[i * 3 + 2] = kB;
    }
    return f;
}

int integrateOnce(float min_depth, float max_depth, float depth_value, TSDFVolume& vol) {
    const Frame f = makeFrame(depth_value);
    vol.integrate(f.depth.data(), f.rgb.data(), Eigen::Matrix4f::Identity(), kFx, kFy, kCx, kCy,
                  kImgW, kImgH, min_depth, max_depth);
    return vol.integratedFrames();
}

const Voxel& atVoxel(const TSDFVolume& v, int x, int y, int z) {
    const size_t idx = static_cast<size_t>(z) * kRes * kRes + static_cast<size_t>(y) * kRes + x;
    return v.voxelData()[idx];
}

size_t dirtyVoxels(const TSDFVolume& v) {
    size_t n = 0;
    for (const Voxel& vox : v.voxelData()) {
        if (vox.weight > EMPTY_WEIGHT) ++n;
    }
    return n;
}

// The canonical empty state, everywhere (hand-derived: nothing was integrated).
bool allEmpty(const TSDFVolume& v) {
    for (const Voxel& vox : v.voxelData()) {
        if (vox.tsdf != EMPTY_TSDF || vox.weight != EMPTY_WEIGHT || vox.r != EMPTY_COLOR ||
            vox.g != EMPTY_COLOR || vox.b != EMPTY_COLOR) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// A. the configured min_depth is the pixel gate
// ---------------------------------------------------------------------------
void gateA_pixelGate() {
    const std::string tag = "A/pixel-gate";

    TSDFVolume near_reject(fixtureParams());
    const int frames = integrateOnce(0.60f, 3.0f, kDepth, near_reject);
    CHECK(allEmpty(near_reject),
          tag + ": a pixel at 0.50125 m with min_depth=0.60 leaves the volume canonically empty");
    CHECK(dirtyVoxels(near_reject) == 0, tag + ": zero observed voxels after the rejected frame");
    CHECK(frames == 1, tag + ": the frame still counted as integrated (the gate is per pixel)");

    TSDFVolume near_accept(fixtureParams());
    integrateOnce(0.10f, 3.0f, kDepth, near_accept);
    CHECK(dirtyVoxels(near_accept) > 0,
          tag + ": the identical frame with min_depth=0.10 integrates, dirty=" +
              std::to_string(dirtyVoxels(near_accept)));
    CHECK(!allEmpty(near_accept), tag + ": the accepted volume is not empty");
    std::printf("A reject_dirty=%zu accept_dirty=%zu\n", dirtyVoxels(near_reject),
                dirtyVoxels(near_accept));
}

// ---------------------------------------------------------------------------
// B. the configured min_depth also moves the march's near clamp
// ---------------------------------------------------------------------------
// The near clamp is a Z-depth clamp, so it is applied to the derived march interval in
// the same units the pixel gate uses. Voxel (32,32,47) takes exactly one candidate in
// each volume; the two candidates differ because the march start differs.
void gateB_marchNearClamp() {
    const std::string tag        = "B/march-near-clamp";
    const double      eps_blend  = 1e-6;      // the documented blend epsilon

    const double tsdf_new_open  = 1.0;                       // min(1, 0.025/0.025)
    const double expect_open    = (EMPTY_TSDF * 0.0 + tsdf_new_open * 1.0) / (0.0 + 1.0 + eps_blend);
    const double tsdf_new_clipped = (kDepth - 0.4775) / kTrunc;   // 0.02375/0.025 = 0.95
    const double expect_clipped   = (EMPTY_TSDF * 0.0 + tsdf_new_clipped * 1.0) / (0.0 + 1.0 + eps_blend);

    TSDFVolume open_vol(fixtureParams()), clipped_vol(fixtureParams());
    integrateOnce(0.10f, 3.0f, kDepth, open_vol);
    integrateOnce(0.4775f, 3.0f, kDepth, clipped_vol);

    const Voxel& o = atVoxel(open_vol, 32, 32, 47);
    const Voxel& c = atVoxel(clipped_vol, 32, 32, 47);

    CHECK(o.weight == 1.0f && c.weight == 1.0f,
          tag + ": each voxel 47 took exactly one candidate (w=1), got " +
              std::to_string(o.weight) + " / " + std::to_string(c.weight));
    CHECK(std::fabs(static_cast<double>(o.tsdf) - expect_open) < 1e-5,
          tag + ": min_depth=0.10 voxel tsdf == " + std::to_string(expect_open) + ", got " +
              std::to_string(o.tsdf));
    CHECK(std::fabs(static_cast<double>(c.tsdf) - expect_clipped) < 1e-5,
          tag + ": min_depth=0.4775 voxel tsdf == " + std::to_string(expect_clipped) +
              " (march started at the configured near bound), got " + std::to_string(c.tsdf));
    // The discriminator, independent of any tolerance: a hard-coded 0.1f floor makes
    // the two volumes identical.
    CHECK(std::fabs(static_cast<double>(o.tsdf) - static_cast<double>(c.tsdf)) > 0.04,
          tag + ": the configured band demonstrably changed the fold, delta=" +
              std::to_string(std::fabs(o.tsdf - c.tsdf)));
    // Color gate: both voxels are in the color band (sdf > -trunc/2), and the single
    // update from EMPTY_COLOR is the hand-derived src value.
    const uint8_t expect_r = static_cast<uint8_t>(
        std::round((EMPTY_COLOR * 0.0 + static_cast<double>(kR)) / (0.0 + 1.0 + eps_blend)));
    CHECK(o.r == expect_r && o.g == kG && o.b == kB,
          tag + ": color of the accepted voxel is the frame's RGB, got (" + std::to_string(o.r) +
              "," + std::to_string(o.g) + "," + std::to_string(o.b) + ")");
    CHECK(c.r == expect_r && c.g == kG && c.b == kB, tag + ": the clipped voxel keeps the same color");
    std::printf("B tsdf_open=%.9f want=%.9f tsdf_clipped=%.9f want=%.9f w=%.6f/%.6f\n", o.tsdf,
                expect_open, c.tsdf, expect_clipped, o.weight, c.weight);
}

// ---------------------------------------------------------------------------
// C. the far bound and the non-finite predicate still gate
// ---------------------------------------------------------------------------
void gateC_farAndNonFinite() {
    const std::string tag = "C/far-and-non-finite";
    struct Case {
        float       depth;
        float       min_d, max_d;
        const char* name;
    };
    const Case cases[] = {
        {kDepth, 0.1f, 0.40f, "max_depth=0.40 rejects 0.50125 m"},
        {kNaN, 0.1f, 3.0f, "NaN depth is rejected"},
        {kInf, 0.1f, 3.0f, "+Inf depth is rejected"},
        {-kInf, 0.1f, 3.0f, "-Inf depth is rejected"},
    };
    for (const Case& c : cases) {
        TSDFVolume vol(fixtureParams());
        integrateOnce(c.min_d, c.max_d, c.depth, vol);
        CHECK(allEmpty(vol), tag + ": " + std::string(c.name) + " -> canonically empty");
    }

    // The far bound is not a literal either: the same pixel is accepted when the band
    // covers it and rejected when it does not.
    TSDFVolume inside(fixtureParams());
    integrateOnce(0.1f, 0.51f, kDepth, inside);
    CHECK(dirtyVoxels(inside) > 0, tag + ": max_depth=0.51 accepts 0.50125 m, dirty=" +
                                      std::to_string(dirtyVoxels(inside)));
    std::printf("C far_accept_dirty=%zu\n", dirtyVoxels(inside));
}

// ---------------------------------------------------------------------------
// D. the band is first-class CPU configuration with one owner
// ---------------------------------------------------------------------------
void gateD_configuration() {
    const std::string tag = "D/configuration";

    const TSDFParams defaults{};
    CHECK(defaults.min_depth == 0.30f,
          tag + ": TSDFParams default min_depth is the documented 0.30 m, got " +
              std::to_string(defaults.min_depth));
    CHECK(defaults.max_depth == 5.00f,
          tag + ": TSDFParams default max_depth is the documented 5.00 m, got " +
              std::to_string(defaults.max_depth));

    // The hyperparameter struct is the single owner; the sync helper mirrors it into
    // the TSDF params exactly like syncIcpDepthFromRange mirrors it into ICP, and it
    // leaves ICP alone (that is the other helper's job).
    FusionHyperparams h;
    h.min_depth = 0.42f;
    h.max_depth = 1.70f;
    const float   icp_before_min = h.icp.min_depth;
    const float   icp_before_max = h.icp.max_depth;
    syncTsdfDepthFromRange(h);
    CHECK(h.tsdf.min_depth == 0.42f && h.tsdf.max_depth == 1.70f,
          tag + ": syncTsdfDepthFromRange propagates the owner's band into TSDFParams, got (" +
              std::to_string(h.tsdf.min_depth) + "," + std::to_string(h.tsdf.max_depth) + ")");
    CHECK(h.icp.min_depth == icp_before_min && h.icp.max_depth == icp_before_max,
          tag + ": syncTsdfDepthFromRange does not touch ICP");

    // A volume configured through setParams reports the band it raycasts with.
    TSDFVolume vol(fixtureParams());
    TSDFParams p = fixtureParams();
    p.min_depth = 0.55f;
    p.max_depth = 2.25f;
    vol.setParams(p);
    CHECK(vol.params().min_depth == 0.55f && vol.params().max_depth == 2.25f,
          tag + ": setParams round-trips the band, got (" + std::to_string(vol.params().min_depth) +
              "," + std::to_string(vol.params().max_depth) + ")");

    // The band is part of the parameter identity, so the canonical Todo 8 rule holds:
    // a band change is a parameter change and no fused state survives it.
    integrateOnce(0.1f, 3.0f, kDepth, vol);
    CHECK(dirtyVoxels(vol) > 0,
          tag + ": baseline integration observed voxels, dirty=" + std::to_string(dirtyVoxels(vol)));
    TSDFParams rebanded = vol.params();
    rebanded.max_depth = 2.75f; // only the band differs
    vol.setParams(rebanded);
    CHECK(allEmpty(vol), tag + ": a depth-band change clears the volume (Todo 8 rule)");
    std::printf("D defaults=(%.2f,%.2f) synced=(%.2f,%.2f) roundtrip=(%.2f,%.2f) dirty_before=%zu\n",
                defaults.min_depth, defaults.max_depth, h.tsdf.min_depth, h.tsdf.max_depth,
                vol.params().min_depth, vol.params().max_depth, dirtyVoxels(vol));
}

} // namespace

int main() {
    gateA_pixelGate();
    gateB_marchNearClamp();
    gateC_farAndNonFinite();
    gateD_configuration();

    if (g_failures == 0) {
        std::printf("tsdf_integration_min_depth_contract: PASS (%d checks: configured pixel gate, "
                    "configured march near clamp, far/non-finite gates, one config owner)\n",
                    g_checks);
        return 0;
    }
    std::printf("tsdf_integration_min_depth_contract: FAIL (%d failed checks of %d)\n", g_failures,
                g_checks);
    return 1;
}
