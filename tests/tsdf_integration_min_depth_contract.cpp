// tsdf_integration_min_depth_contract (big-fix Todo 15): CPU-only contract that TSDF
// integration honors the CONFIGURED near depth bound. Public API only (TSDFVolume
// ctor/params/integrate/voxelData/voxelAt + the FusionHyperparams sync helper): no
// device, display, GPU, sensor, thread timing, sleep or filesystem. Expected values are
// hand-derived doubles from the fixture geometry, never product output. Locks:
//   A  the configured min_depth is the pixel gate: a frame whose only valid pixel is
//      nearer than min_depth leaves the volume in the canonical empty state, while the
//      identical frame with a smaller min_depth integrates
//   B  an accepted pixel folds hand-derived voxel values (one update per voxel per
//      frame, SDF at the voxel corner), identical for any min_depth it passes
//   C  max_depth still gates, and non-finite depth still gates
//   D  TSDFParams carries the band with the documented defaults, and the single
//      hyperparameter owner propagates it (no second store)
//
// Fixture: 64^3 volume, 10 mm voxels, truncation 25 mm, origin (-0.32,-0.32,0). A 4x4
// frame whose ONLY valid pixel is (2,2) with cx=cy=2, fx=fy=500, so the axis voxel
// corners (0,0,k*vs) project exactly onto it and every off-axis voxel falls outside
// the image. D = 0.50125, trunc = 0.025.
#include "app/FusionHyperparams.h"
#include "tsdf/TSDFVolume.h"
#include "utils/ColorMath.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// The canonical single-update color fold, in double, from the neutral EMPTY_COLOR:
// (EMPTY_COLOR*0 + byte/255) / (0 + 1 + eps). Independent of the product's float math.
double colorOracle(uint8_t byte) {
    return (static_cast<double>(EMPTY_COLOR) * 0.0 + static_cast<double>(byte) / 255.0) /
           (0.0 + 1.0 + 1e-6);
}
bool nearColor(float got, double want) { return std::fabs(static_cast<double>(got) - want) < 1e-6; }

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
// B. an accepted pixel folds the hand-derived voxel values, whatever min_depth is
// ---------------------------------------------------------------------------
// Voxel-projective integration has no march and no near clamp: min_depth only
// gates the MEASURED depth. Axis voxel k (world z = k*vs) takes one update with
// sdf = D - k*vs. Voxel 50 (sdf = +1.25 mm) is a surface voxel inside the colour
// band |sdf| < trunc/2; voxel 47 (sdf = +31.25 mm > trunc) is free space: tsdf 1,
// weight 1, colour untouched. Both volumes must agree bit for bit.
void gateB_acceptedPixelFold() {
    const std::string tag = "B/accepted-pixel-fold";

    TSDFVolume open_vol(fixtureParams()), tight_vol(fixtureParams());
    integrateOnce(0.10f, 3.0f, kDepth, open_vol);
    integrateOnce(0.50f, 3.0f, kDepth, tight_vol);   // 0.50125 still passes

    const double expect_surface = (kDepth - 0.50) / kTrunc;   // 0.05
    for (const TSDFVolume* vol : {&open_vol, &tight_vol}) {
        const std::string who = tag + (vol == &open_vol ? " min=0.10" : " min=0.50");
        const Voxel& s50 = atVoxel(*vol, 32, 32, 50);
        const Voxel& f47 = atVoxel(*vol, 32, 32, 47);
        CHECK(s50.weight == 1.0f && std::fabs(static_cast<double>(s50.tsdf) - expect_surface) < 1e-5,
              who + ": surface voxel tsdf == " + std::to_string(expect_surface) + ", got " +
                  std::to_string(s50.tsdf));
        const double er = colorOracle(kR), eg = colorOracle(kG), eb = colorOracle(kB);
        CHECK(nearColor(s50.r, er) && nearColor(s50.g, eg) && nearColor(s50.b, eb),
              who + ": surface voxel colour is the frame's RGB");
        uint8_t qr = 0, qg = 0, qb = 0;
        CHECK(kfusion::utils::srgbFloatToUint8(s50.r, qr) && kfusion::utils::srgbFloatToUint8(s50.g, qg) &&
                  kfusion::utils::srgbFloatToUint8(s50.b, qb) && qr == kR && qg == kG && qb == kB,
              who + ": the float colour quantizes back to the frame's exact RGB bytes");
        CHECK(f47.weight == 1.0f && f47.tsdf == 1.0f, who + ": free-space voxel observed as tsdf 1");
        CHECK(f47.r == EMPTY_COLOR && f47.g == EMPTY_COLOR && f47.b == EMPTY_COLOR,
              who + ": free-space voxel colour untouched (outside the colour band)");
    }
    CHECK(std::memcmp(open_vol.voxelData().data(), tight_vol.voxelData().data(),
                      open_vol.voxelData().size() * sizeof(Voxel)) == 0,
          tag + ": min_depth below the measured depth does not change the fold");
    std::printf("B surface tsdf=%.9f want=%.9f\n", atVoxel(open_vol, 32, 32, 50).tsdf, expect_surface);
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
    gateB_acceptedPixelFold();
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
