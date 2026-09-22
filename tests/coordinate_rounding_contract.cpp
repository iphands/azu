// coordinate_rounding_contract (big-fix todo 11): CPU-only contract for the
// shared floor/finite/range integer-coordinate conversion and the two CPU
// production surfaces that consume it.
//
//   * kfusion::utils::floorToInt()            (include/utils/CoordinateMath.h)
//   * kfusion::tsdf::TSDFVolume::worldToVoxel()  (+ its getTSDF fallback, which
//                                                delegates to worldToVoxel)
//   * kfusion::tracking::ICPTracker model-pixel projection, exercised through the
//     public ICPTracker::track() entry point via ICPResult::valid_live_points.
//
// The oracle is the mathematical definition of floor (greatest integer <= the
// value) hand-encoded as literal expected integers; it never calls the helper or
// std::floor to compute an expected value, so a truncating implementation
// (static_cast<int>) fails against these literals. Public API only. No device,
// display, GPU, sensor, thread timing or filesystem.

#include "sensor/FrameData.h"
#include "tracking/ICPTracker.h"
#include "tsdf/TSDFVolume.h"
#include "utils/CoordinateMath.h"

#include <climits>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "coordinate_rounding_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

using kfusion::tsdf::EMPTY_TSDF;
using kfusion::tsdf::TSDFParams;
using kfusion::tsdf::TSDFVolume;
using kfusion::tsdf::Voxel;
using kfusion::utils::floorToInt;

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, what)                                                        \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            std::printf("FAIL: %s  [%s:%d]\n", std::string(what).c_str(),         \
                        __FILE__, __LINE__);                                     \
            ++g_failures;                                                        \
        }                                                                        \
    } while (false)

const float kNaN   = std::numeric_limits<float>::quiet_NaN();
const float kInf   = std::numeric_limits<float>::infinity();
// Distinct from any plausible converted value so "left untouched" is observable.
const int   kSentinel = -1234567;

bool vecEq(const Eigen::Vector3i& v, int x, int y, int z) {
    return v.x() == x && v.y() == y && v.z() == z;
}

std::string vecStr(const Eigen::Vector3i& v) {
    return "(" + std::to_string(v.x()) + ", " + std::to_string(v.y()) + ", " +
           std::to_string(v.z()) + ")";
}

// --- helper floorToInt --------------------------------------------------------

// Expected integers are the floor of each input per the definition, never the
// result of calling floor in this test. Negative fractional rows are the
// truncation discriminators: floor(-0.0001) = -1 and floor(-2.7) = -3, whereas
// static_cast<int> would give 0 and -2.
struct FloorCase {
    float in;
    bool  expect_ok;
    int   expect;  // meaningful only when expect_ok
};

const FloorCase kFloorCases[] = {
    {0.0f, true, 0},
    {0.5f, true, 0},
    {0.9999f, true, 0},
    {1.0f, true, 1},
    {2.7f, true, 2},
    {2.9999f, true, 2},
    {3.0f, true, 3},
    {425.0f, true, 425},
    {-0.0001f, true, -1},   // trunc -> 0, floor -> -1
    {-0.5f, true, -1},
    {-1.0f, true, -1},
    {-1.5f, true, -2},
    {-2.7f, true, -3},      // trunc -> -2, floor -> -3
    {-3.0f, true, -3},
    {1000000.0f, true, 1000000},
    {-1000000.0f, true, -1000000},
    {kNaN, false, kSentinel},
    {kInf, false, kSentinel},
    {-kInf, false, kSentinel},
    {3e9f, false, kSentinel},   // floor > INT_MAX
    {-3e9f, false, kSentinel},  // floor < INT_MIN
    {1e10f, false, kSentinel},
};

void testHelperFloorSemantics() {
    for (const FloorCase& c : kFloorCases) {
        int out = kSentinel;
        const bool ok = floorToInt(c.in, &out);
        const std::string who = "floorToInt(" + std::to_string(c.in) + ")";
        CHECK(ok == c.expect_ok, who + (c.expect_ok ? ": accepted" : ": rejected"));

        if (c.expect_ok) {
            CHECK(out == c.expect, who + " == " + std::to_string(c.expect));
        } else {
            // Rejection must leave *out untouched (never a half-written value).
            CHECK(out == kSentinel, who + ": rejected input leaves *out untouched");
        }
    }

    // floorToInt must never write through a null out-pointer.
    CHECK(!floorToInt(1.5f, nullptr), "floorToInt(1.5f, nullptr) returns false");
    std::printf("  helper: %zu inputs (floor / finite / int-range)\n",
                sizeof(kFloorCases) / sizeof(FloorCase));
}

// The projection consumes floorToInt(model + 0.5f): round-half-up for positives.
// Expected pixel = floor(model + 0.5) per the definition.
struct RoundCase {
    float model;
    int   expect;
};
const RoundCase kRoundCases[] = {
    {0.0f, 0},    // floor(0.5)
    {0.4f, 0},    // floor(0.9)
    {0.5f, 1},    // floor(1.0)
    {1.5f, 2},    // floor(2.0)
    {2.3f, 2},    // floor(2.8)
    {424.5f, 425} // floor(425.0)
};

void testProjectionRoundingShape() {
    for (const RoundCase& c : kRoundCases) {
        int out = kSentinel;
        const bool ok = floorToInt(c.model + 0.5f, &out);
        CHECK(ok, "round(" + std::to_string(c.model) + "): convertible");
        CHECK(out == c.expect,
              "round(" + std::to_string(c.model) + ") == " + std::to_string(c.expect));
    }
    std::printf("  projection rounding shape: %zu half-up cases\n",
                sizeof(kRoundCases) / sizeof(RoundCase));
}

// --- TSDFVolume::worldToVoxel -------------------------------------------------

// A power-of-two voxel size (0.25) and an integral negative origin make every
// (world - origin) / voxel_size exact in float, so the floor boundary is tested
// without float-epsilon fragility. Origin has negative x/y/z so below-origin
// points exercise the negative-coordinate path.
constexpr int   kRes = 8;
constexpr float kVs  = 0.25f;

TSDFParams negOriginParams() {
    TSDFParams p;
    p.resolution = kRes;
    p.voxel_size = kVs;
    p.truncation = kVs;
    p.max_weight = 128.0f;
    p.origin     = Eigen::Vector3f(-1.0f, -1.0f, -1.0f);
    return p;
}

struct WorldCase {
    Eigen::Vector3f world;
    int             ex, ey, ez;
};

const WorldCase kWorldCases[] = {
    // exact voxel corner at the origin
    {{-1.0000f, -1.0000f, -1.0000f}, 0, 0, 0},
    // exact positive multiples: v = (1, 2, 3)
    {{-0.7500f, -0.5000f, -0.2500f}, 1, 2, 3},
    // mid-voxel positive: v = (0.5, 1.5, 2.5) -> floor (0, 1, 2)
    {{-0.8750f, -0.6250f, -0.3750f}, 0, 1, 2},
    // exact top-corner: v = (7, 2, 6)
    {{0.7500f, -0.5000f, 0.5000f}, 7, 2, 6},
    // BELOW origin, half a voxel: v = (-0.5,-0.5,-0.5). floor -> -1 each.
    // trunc would give (0,0,0): this row is the RED discriminator.
    {{-1.1250f, -1.1250f, -1.1250f}, -1, -1, -1},
    // BELOW origin, exact -1 voxel boundary: v = (-1,-1,-1).
    {{-1.2500f, -1.2500f, -1.2500f}, -1, -1, -1},
    // negative fractional past -1: v = (-1.75) -> floor -2 (trunc -> -1).
    {{-1.4375f, -1.0000f, -1.0000f}, -2, 0, 0},
};

void testWorldToVoxelFloors() {
    const TSDFVolume vol(negOriginParams());
    for (const WorldCase& c : kWorldCases) {
        const Eigen::Vector3i got = vol.worldToVoxel(c.world);
        const std::string who = "worldToVoxel(" + std::to_string(c.world.x()) + ", " +
                                std::to_string(c.world.y()) + ", " +
                                std::to_string(c.world.z()) + ")";
        CHECK(vecEq(got, c.ex, c.ey, c.ez),
              who + " == (" + std::to_string(c.ex) + ", " + std::to_string(c.ey) + ", " +
                  std::to_string(c.ez) + "), got " + vecStr(got));
    }
    std::printf("  worldToVoxel: %zu world points floored\n",
                sizeof(kWorldCases) / sizeof(WorldCase));
}

// The mixed floor/trunc fallback bug: getTSDF() (private) is only reachable
// through worldToVoxel() for a below-origin point, because its trilinear grid
// floor (x0 = floor(v) = -1 < 0) sends it into the inBounds(worldToVoxel(p))
// fallback. So worldToVoxel(origin - eps) == (-1,-1,-1) is exactly what makes
// getTSDF return EMPTY_TSDF there instead of reading voxel (0,0,0). Voxel (0,0,0)
// is dirtied below so the pre-fix fallback (which returned trunc(-0.5) = (0,0,0))
// would have surfaced a non-empty value; the assertion is the delegate itself.
void testGetTSDFFallbackDelegate() {
    TSDFVolume vol(negOriginParams());
    // Dirty (0,0,0) so "returned voxel (0,0,0)" would be observably non-empty.
    Voxel& v0 = vol.voxelAt(0, 0, 0);
    v0.tsdf   = 0.5f;
    v0.weight = 3.0f;

    const Eigen::Vector3f just_below_origin(-1.125f, -1.125f, -1.125f);  // v = -0.5
    const Eigen::Vector3i vi = vol.worldToVoxel(just_below_origin);
    CHECK(!vecEq(vi, 0, 0, 0),
          "worldToVoxel(just below origin) is NOT voxel (0,0,0) [getTSDF fallback], got " + vecStr(vi));
    CHECK(vecEq(vi, -1, -1, -1),
          "worldToVoxel(just below origin) == (-1,-1,-1) [out-of-bounds sentinel path]");
    // Every component is < 0, so the volume's own inBounds() rejects it and
    // getTSDF's fallback returns EMPTY_TSDF, never the dirtied (0,0,0).
    CHECK(vi.x() < 0 && vi.y() < 0 && vi.z() < 0,
          "below-origin voxel index is out-of-bounds on every axis");
    std::printf("  getTSDF fallback delegate: worldToVoxel below-origin = %s\n", vecStr(vi).c_str());
}

void testWorldToVoxelNonFinite() {
    const TSDFVolume vol(negOriginParams());

    const Eigen::Vector3i all_nan = vol.worldToVoxel(Eigen::Vector3f(kNaN, kNaN, kNaN));
    CHECK(vecEq(all_nan, INT_MIN, INT_MIN, INT_MIN),
          "worldToVoxel(NaN,NaN,NaN) == (INT_MIN,INT_MIN,INT_MIN), got " + vecStr(all_nan));

    const Eigen::Vector3i pos_inf = vol.worldToVoxel(Eigen::Vector3f(kInf, 0.0f, 0.0f));
    CHECK(pos_inf.x() == INT_MIN,
          "worldToVoxel(+Inf,.,.) x axis == INT_MIN, got " + vecStr(pos_inf));

    const Eigen::Vector3i neg_inf = vol.worldToVoxel(Eigen::Vector3f(-kInf, 0.0f, 0.0f));
    CHECK(neg_inf.x() == INT_MIN,
          "worldToVoxel(-Inf,.,.) x axis == INT_MIN, got " + vecStr(neg_inf));

    // Finite but beyond the int range on one axis: a single invalid axis makes
    // the WHOLE returned vector the (INT_MIN,INT_MIN,INT_MIN) out-of-bounds
    // sentinel (worldToVoxel short-circuits on the first failed axis), so only
    // the x axis needs asserting here.
    const Eigen::Vector3i huge = vol.worldToVoxel(Eigen::Vector3f(1e10f, 0.0f, 0.0f));
    CHECK(huge.x() == INT_MIN,
          "worldToVoxel(1e10,.,.) x axis == INT_MIN, got " + vecStr(huge));

    // The sentinel is rejected by inBounds() (INT_MIN < 0), and a non-finite x
    // never silently aliases a finite y/z into an in-bounds index.
    CHECK(pos_inf.x() < 0 && neg_inf.x() < 0 && huge.x() < 0,
          "non-finite / out-of-range axes are always out-of-bounds");
    std::printf("  worldToVoxel non-finite: NaN/+Inf/-Inf/1e10 -> INT_MIN sentinel, no crash\n");
}

// --- voxelToWorld round trip --------------------------------------------------

void testVoxelToWorldRoundTrip() {
    const TSDFVolume vol(negOriginParams());
    const Eigen::Vector3i corners[] = {
        {0, 0, 0}, {kRes - 1, kRes - 1, kRes - 1}, {3, 4, 5}, {-1, -1, -1},
    };
    for (const Eigen::Vector3i& v : corners) {
        const Eigen::Vector3f corner = vol.voxelToWorld(v);
        // Corner round trip: exact multiple of voxel_size floors back to v.
        CHECK(vecEq(vol.worldToVoxel(corner), v.x(), v.y(), v.z()),
              "worldToVoxel(voxelToWorld(v)) == v (corner), v=" + vecStr(v));
        // Center round trip: corner + half a voxel is v + 0.5, floors back to v.
        const Eigen::Vector3f center = corner + Eigen::Vector3f(kVs * 0.5f, kVs * 0.5f, kVs * 0.5f);
        CHECK(vecEq(vol.worldToVoxel(center), v.x(), v.y(), v.z()),
              "worldToVoxel(center of v) == v (center), v=" + vecStr(v));
    }
    std::printf("  voxelToWorld round trip: %zu corners/centers (incl. negative origin)\n",
                sizeof(corners) / sizeof(Eigen::Vector3i));
}

// --- raycast smoke: exercises the private getTSDF path end to end -------------

// raycast() internally samples getTSDF() and, on a hit, calls worldToVoxel() for
// the color. Running it over a dirtied volume drives the getTSDF/worldToVoxel
// production path (including the below-origin marching region) through a public
// entry point. Assert only what is robust: no crash, deterministic output, and
// every emitted vertex/normal is finite (no NaN leaks from the coordinate path).
void testRaycastExercisesGetTSDF() {
    TSDFVolume vol(negOriginParams());
    for (int z = 0; z < kRes; ++z) {
        for (int y = 0; y < kRes; ++y) {
            for (int x = 0; x < kRes; ++x) {
                Voxel& v = vol.voxelAt(x, y, z);
                v.tsdf   = (x % 2) ? 0.4f : -0.4f;  // alternating sign creates crossings
                v.weight = 5.0f;
            }
        }
    }

    const int W = 16, H = 16;
    std::vector<Eigen::Vector3f> va(static_cast<size_t>(W) * H), vb(static_cast<size_t>(W) * H);
    std::vector<Eigen::Vector3f> na(static_cast<size_t>(W) * H), nb(static_cast<size_t>(W) * H);
    std::vector<uint8_t> ca(static_cast<size_t>(W) * H * 3), cb(static_cast<size_t>(W) * H * 3);

    const Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    vol.raycast(pose, 200.0f, 200.0f, 7.5f, 7.5f, W, H, va.data(), na.data(), ca.data());
    vol.raycast(pose, 200.0f, 200.0f, 7.5f, 7.5f, W, H, vb.data(), nb.data(), cb.data());

    size_t bad = 0;
    for (int i = 0; i < W * H; ++i) {
        if (!va[i].allFinite() || !na[i].allFinite()) ++bad;
    }
    CHECK(bad == 0, "raycast: every emitted vertex/normal is finite (no NaN from coord path)");

    size_t diff = 0;
    for (int i = 0; i < W * H; ++i) {
        if (va[i] != vb[i] || na[i] != nb[i]) ++diff;
    }
    for (size_t i = 0; i < ca.size(); ++i) {
        if (ca[i] != cb[i]) ++diff;
    }
    CHECK(diff == 0, "raycast: deterministic across repeated runs on identical input");
    std::printf("  raycast getTSDF path: %dx%d finite + deterministic\n", W, H);
}

// --- ICP model-pixel projection through the public track() entry point --------

// The projection loop is private, but ICPResult::valid_live_points is the public
// witness for the live-vertex finite guard: a NaN/Inf live vertex must NOT be
// counted as a valid live correspondence. The model frame is left all-invalid
// (zeros), so no inlier can form and the counters are fully determined by the
// three live vertices placed below.
void testIcpRejectsNonFiniteLiveVertex() {
    kfusion::tracking::ICPTracker tracker;
    tracker.setNumThreads(1);  // single-threaded: no reduction-order variance

    kfusion::sensor::FramePyramid live;
    kfusion::tracking::ModelFrame model;  // all vertices/normals zero => invalid model

    const int W = live.levels[0].width;
    auto put = [&](int x, int y, const Eigen::Vector3f& v) {
        live.levels[0].vertices[static_cast<size_t>(y) * W + x] = v;
        live.levels[0].normals[static_cast<size_t>(y) * W + x]  = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    };
    // (100,100): finite, z > 0.001, projects in-bounds. (200,200): NaN z.
    // (300,300): +Inf x with finite z.
    put(100, 100, Eigen::Vector3f(0.1f, 0.05f, 0.5f));
    put(200, 200, Eigen::Vector3f(0.1f, 0.05f, kNaN));
    put(300, 300, Eigen::Vector3f(kInf, 0.05f, 0.5f));

    const Eigen::Matrix4f id = Eigen::Matrix4f::Identity();
    const kfusion::tracking::ICPResult a = tracker.track(live, model, id, id);
    const kfusion::tracking::ICPResult b = tracker.track(live, model, id, id);

    // Only the finite live vertex survives the guard. Pre-fix (NaN/Inf leaking
    // through the z filter) this reports 3, not 1: this is the RED discriminator.
    CHECK(a.valid_live_points == 1,
          "track(): only the finite live vertex counts as valid live, got " +
              std::to_string(a.valid_live_points));
    CHECK(a.inliers == 0, "track(): an all-invalid model yields zero inliers, got " +
                             std::to_string(a.inliers));
    CHECK(a.projected_points == 1,
          "track(): the one finite live vertex projects once, got " +
              std::to_string(a.projected_points));
    CHECK(!a.tracking_ok, "track(): sparse invalid-model input never reports tracking_ok");
    CHECK(a.valid_live_points == b.valid_live_points && a.projected_points == b.projected_points &&
              a.inliers == b.inliers,
          "track(): counters are deterministic across repeated calls");
    std::printf("  ICP projection guard: valid_live=%d projected=%d inliers=%d (expect 1/1/0)\n",
                a.valid_live_points, a.projected_points, a.inliers);
}

} // namespace

int main() {
    testHelperFloorSemantics();
    testProjectionRoundingShape();
    testWorldToVoxelFloors();
    testGetTSDFFallbackDelegate();
    testWorldToVoxelNonFinite();
    testVoxelToWorldRoundTrip();
    testRaycastExercisesGetTSDF();
    testIcpRejectsNonFiniteLiveVertex();

    if (g_failures == 0) {
        std::printf("coordinate_rounding_contract: PASS (%d checks: floorToInt + worldToVoxel "
                    "+ getTSDF fallback + voxelToWorld + raycast + ICP projection guard)\n",
                    g_checks);
        return 0;
    }
    std::printf("coordinate_rounding_contract: FAIL (%d failed checks of %d)\n", g_failures, g_checks);
    return 1;
}
