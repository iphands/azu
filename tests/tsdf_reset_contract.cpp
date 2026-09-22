// tsdf_reset_contract (big-fix todo 8): CPU-only contract for the canonical
// empty-voxel sentinel and for stale host state after a TSDFParams change.
// Public API only; no device, display, GPU, sensor, thread timing or filesystem.

#include "tsdf/TSDFVolume.h"
#include "utils/ColorMath.h"

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "tsdf_reset_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

using kfusion::tsdf::EMPTY_COLOR;
using kfusion::tsdf::EMPTY_TSDF;
using kfusion::tsdf::EMPTY_WEIGHT;
using kfusion::tsdf::TSDFParams;
using kfusion::tsdf::TSDFVolume;
using kfusion::tsdf::Voxel;
using kfusion::utils::srgbUint8ToFloat;   // uint8 sRGB -> the volume's float sRGB domain

int g_failures = 0;
int g_checks   = 0;

void reportFailure(const std::string& what, const char* file, int line) {
    std::printf("FAIL: %s  [%s:%d]\n", what.c_str(), file, line);
    ++g_failures;
}

#define CHECK(cond, what)                                                       \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) reportFailure(std::string(what), __FILE__, __LINE__);       \
    } while (false)

constexpr int kRes = 8;  // 512 voxels: every voxel is scanned, never sampled

TSDFParams baseParams() {
    TSDFParams p;
    p.resolution = kRes;
    p.voxel_size = 0.05f;
    p.truncation = 0.05f;
    p.max_weight = 128.0f;
    p.origin     = Eigen::Vector3f(-0.2f, -0.2f, 0.3f);
    return p;
}

const Voxel& at(const TSDFVolume& v, int x, int y, int z) {
    const int res = v.params().resolution;
    return v.voxelData()[static_cast<size_t>((z * res + y) * res + x)];
}

bool isClean(const Voxel& v) {
    return v.tsdf == EMPTY_TSDF && v.weight == EMPTY_WEIGHT &&
           v.r == EMPTY_COLOR && v.g == EMPTY_COLOR && v.b == EMPTY_COLOR;
}

// Scans EVERY voxel of the host mirror (not a sample) and reports the count that
// is not the canonical empty state, printing the first offender.
void checkAllVoxelsEmpty(const TSDFVolume& vol, const std::string& stage) {
    const int res = vol.params().resolution;
    const size_t expected = static_cast<size_t>(res) * res * res;
    CHECK(vol.voxelData().size() == expected,
          stage + ": host storage holds exactly resolution^3 voxels");
    size_t dirty = 0;
    for (size_t i = 0; i < vol.voxelData().size(); ++i) {
        const Voxel& v = vol.voxelData()[i];
        if (!isClean(v)) {
            if (dirty == 0) {
                std::printf("  first non-empty voxel at %zu: tsdf=%.9g weight=%.9g "
                            "rgb=(%.9g,%.9g,%.9g); expected tsdf=%.9g weight=%.9g rgb=(%.9g,%.9g,%.9g)\n",
                            i, v.tsdf, v.weight, v.r, v.g, v.b,
                            EMPTY_TSDF, EMPTY_WEIGHT, EMPTY_COLOR, EMPTY_COLOR, EMPTY_COLOR);
            }
            ++dirty;
        }
    }
    CHECK(dirty == 0, stage + ": every voxel is tsdf==EMPTY_TSDF, weight==EMPTY_WEIGHT, neutral color "
                       "(found " + std::to_string(dirty) + " non-empty)");
    CHECK(vol.usageFraction() == 0.0f, stage + ": usageFraction() is exactly 0");
    CHECK(vol.integratedFrames() == 0, stage + ": integratedFrames() is 0");
}

// Four in-bounds voxels carrying state that differs from empty in all fields.
const int kDirtyPts[4][3] = {{0, 0, 0}, {1, 2, 3}, {3, 4, 5}, {kRes - 1, kRes - 1, kRes - 1}};

float dirtyTsdf(int i) { return 0.25f + 0.1f * static_cast<float>(i); }
float dirtyWeight(int i) { return 7.0f + static_cast<float>(i); }
// The color domain is float sRGB [0,1], so the dirty fill goes through the same
// byte -> float widening production integration uses; a raw 11 in this field is
// an out-of-range sRGB component, not a color.
float dirtyR(int i) { return srgbUint8ToFloat(static_cast<uint8_t>(11 + i)); }
float dirtyG(int i) { return srgbUint8ToFloat(static_cast<uint8_t>(22 + i)); }
float dirtyB(int i) { return srgbUint8ToFloat(static_cast<uint8_t>(33 + i)); }

void dirtyVolume(TSDFVolume& vol) {
    for (int i = 0; i < 4; ++i) {
        Voxel& v = vol.voxelAt(kDirtyPts[i][0], kDirtyPts[i][1], kDirtyPts[i][2]);
        v.tsdf   = dirtyTsdf(i);
        v.weight = dirtyWeight(i);
        v.r = dirtyR(i); v.g = dirtyG(i); v.b = dirtyB(i);
    }
}

size_t countDirty(const TSDFVolume& vol) {
    size_t n = 0;
    for (const Voxel& v : vol.voxelData()) {
        if (!isClean(v)) ++n;
    }
    return n;
}

void checkDirtyStateIsStuck(TSDFVolume& vol, const std::string& stage) {
    CHECK(countDirty(vol) == 4, stage + ": exactly 4 voxels carry non-empty state");
    CHECK(vol.usageFraction() > 0.0f, stage + ": dirty voxels are counted as used");
    for (int i = 0; i < 4; ++i) {
        const Voxel& v = at(vol, kDirtyPts[i][0], kDirtyPts[i][1], kDirtyPts[i][2]);
        CHECK(v.tsdf == dirtyTsdf(i) && v.weight == dirtyWeight(i) &&
              v.r == dirtyR(i) && v.g == dirtyG(i) && v.b == dirtyB(i),
              stage + ": dirty voxel " + std::to_string(i) + " holds the written bit pattern");
    }
}

// One 8x8 synthetic frame, camera at the world origin looking down +Z, so the
// integration band straddles the volume. No sensor, no GPU, no threading API.
void integrateOneFrame(TSDFVolume& vol) {
    const int W = 8, H = 8;
    std::vector<float> depth(static_cast<size_t>(W) * H, 0.5f);
    std::vector<uint8_t> rgb(static_cast<size_t>(W) * H * 3, 200);
    const Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    vol.integrate(depth.data(), rgb.data(), pose, 500.0f, 500.0f, 3.5f, 3.5f, W, H);
}

void testFreshVolumeIsCanonicalEmpty() {
    const TSDFVolume vol(baseParams());
    checkAllVoxelsEmpty(vol, "freshly constructed volume");
    // The constants themselves are the canonical pair, and a default Voxel
    // (what std::vector<Voxel>::resize() yields) agrees with them.
    CHECK(EMPTY_TSDF == 1.0f, "EMPTY_TSDF is +1.0f, not the zero isosurface");
    CHECK(EMPTY_WEIGHT == 0.0f, "EMPTY_WEIGHT is exactly 0.0f");
    const Voxel defaulted;
    CHECK(isClean(defaulted), "default-constructed Voxel equals the canonical empty state");
}

void testResetRestoresEveryVoxel() {
    TSDFVolume vol(baseParams());
    dirtyVolume(vol);
    checkDirtyStateIsStuck(vol, "after dirtying");

    vol.reset();

    checkAllVoxelsEmpty(vol, "after reset()");
    CHECK(countDirty(vol) == 0, "after reset(): no voxel retains dirty state");
}

void testResetClearsIntegrationState() {
    TSDFVolume vol(baseParams());
    integrateOneFrame(vol);
    CHECK(vol.integratedFrames() == 1, "one integrate() call counts one integrated frame");
    CHECK(vol.usageFraction() > 0.0f, "the integrated frame actually observed voxels to reset");

    vol.reset();

    checkAllVoxelsEmpty(vol, "after reset() of an integrated volume");
}

void testIdenticalParamsPreserveVoxels() {
    TSDFVolume vol(baseParams());
    dirtyVolume(vol);
    checkDirtyStateIsStuck(vol, "before identical setParams");

    vol.setParams(baseParams());  // every field exactly equal to the current params

    checkDirtyStateIsStuck(vol, "after setParams(equal params)");
    CHECK(vol.params().resolution == baseParams().resolution, "params() resolution unchanged");
}

void checkParamsRoundTrip(const TSDFVolume& vol, const TSDFParams& want, const std::string& stage) {
    const TSDFParams& got = vol.params();
    CHECK(got.resolution == want.resolution, stage + ": params().resolution is the new value");
    CHECK(got.voxel_size == want.voxel_size, stage + ": params().voxel_size is the new value");
    CHECK(got.truncation == want.truncation, stage + ": params().truncation is the new value");
    CHECK(got.max_weight == want.max_weight, stage + ": params().max_weight is the new value");
    CHECK((got.origin - want.origin).lpNorm<1>() == 0.0f, stage + ": params().origin is the new value");
}

// Independent (double-precision) recomputation of the geometry mapping, so the
// new voxel_size / origin are proven to be in effect rather than merely stored.
void checkGeometryUsesNewParams(const TSDFVolume& vol, const std::string& stage) {
    const TSDFParams& p = vol.params();
    const double vs = static_cast<double>(p.voxel_size);
    const double ox = static_cast<double>(p.origin.x());
    const Eigen::Vector3f w = vol.voxelToWorld(1, 0, 0);
    CHECK(std::abs(static_cast<double>(w.x()) - (ox + vs)) < 1e-6,
          stage + ": voxelToWorld(1,0,0).x == origin.x + voxel_size");
    const Eigen::Vector3i vi =
        vol.worldToVoxel(p.origin + Eigen::Vector3f(p.voxel_size * 4.5f, p.voxel_size * 0.5f,
                                                    p.voxel_size * 2.5f));
    CHECK(vi.x() == 4 && vi.y() == 0 && vi.z() == 2,
          stage + ": worldToVoxel maps mid-voxel points with the new voxel_size/origin");
}

void testEveryParamChangeClearsStaleState(const char* field,
                                          const std::function<void(TSDFParams&)>& mutate) {
    TSDFVolume vol(baseParams());
    dirtyVolume(vol);
    checkDirtyStateIsStuck(vol, std::string("before ") + field + " change");

    TSDFParams next = baseParams();
    mutate(next);
    vol.setParams(next);

    const std::string stage = std::string("after setParams(changed ") + field + ")";
    checkParamsRoundTrip(vol, next, stage);
    checkAllVoxelsEmpty(vol, stage + " — stale host voxels must not survive");
    CHECK(countDirty(vol) == 0, stage + ": no dirty voxel survived");
    checkGeometryUsesNewParams(vol, stage);
}

} // namespace

int main() {
    testFreshVolumeIsCanonicalEmpty();
    testResetRestoresEveryVoxel();
    testResetClearsIntegrationState();
    testIdenticalParamsPreserveVoxels();

    testEveryParamChangeClearsStaleState("voxel_size", [](TSDFParams& p) { p.voxel_size = 0.025f; });
    testEveryParamChangeClearsStaleState("origin",
                                         [](TSDFParams& p) { p.origin = Eigen::Vector3f(0.1f, -0.3f, 0.4f); });
    testEveryParamChangeClearsStaleState("truncation", [](TSDFParams& p) { p.truncation = 0.02f; });
    testEveryParamChangeClearsStaleState("max_weight", [](TSDFParams& p) { p.max_weight = 32.0f; });
    testEveryParamChangeClearsStaleState("resolution", [](TSDFParams& p) { p.resolution = 5; });

    if (g_failures == 0) {
        std::printf("tsdf_reset_contract: PASS (empty sentinel, reset, setParams stale-state "
                    "clearing; %d checks)\n", g_checks);
        return 0;
    }
    std::printf("tsdf_reset_contract: FAIL (%d failed checks of %d)\n", g_failures, g_checks);
    return 1;
}
