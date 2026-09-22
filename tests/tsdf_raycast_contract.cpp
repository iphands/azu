// tsdf_raycast_contract (big-fix Todo 15): CPU-only contract for TSDFVolume::raycast.
// Public API only (TSDFVolume ctor/params/voxelAt/raycast): no device, display, GPU,
// sensor, thread timing, sleep or filesystem. Every expected number below is derived
// analytically from the fixture definition (a hand-written lattice field), never from
// product output. Locks, in order:
//   A  a known synthetic planar surface is hit within 1e-5 m on the optical axis, with
//      the outward unit normal and the color of the voxel that CONTAINS the resolved
//      crossing (not a neighbour's, not the bracket endpoint's, not the empty fill)
//   B  the configured depth band is a camera-plane Z-depth band, not a ray-parameter
//      band: an off-axis ray hits a plane whose ray parameter exceeds max_depth while
//      its Z-depth is inside the band (the two readings give opposite answers)
//   C  the configured near/far bounds ARE the gate (same field, only the band differs:
//      hit vs the documented empty state)
//   D  both crossings resolve: entry from outside, and EXIT for a ray that starts
//      inside material - the exit case is also the color-ghost discriminator, because
//      the entry-layer color is precisely the stale/adjacent-surface answer
//   E  non-finite volume samples are never emitted as a hit, and a non-finite layer
//      BEHIND a resolved surface does not erase that valid hit
//   F  global output invariant: every pixel is either exactly the empty state or a
//      finite hit inside the configured band with a unit normal, bit-reproducible
//
// The documented empty/miss state is vertex (0,0,0), normal (0,0,0), color (0,0,0).
// That state is unambiguous in this fixture because every configured near bound is
// >= 0.12 m, so a genuine hit can never sit at the camera origin.
#include "tsdf/TSDFVolume.h"
#include "utils/ColorMath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "tsdf_raycast_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

using kfusion::tsdf::EMPTY_COLOR;
using kfusion::tsdf::TSDFParams;
using kfusion::tsdf::TSDFVolume;
using kfusion::utils::srgbFloatToUint8;    // the one CPU float sRGB -> byte policy
using kfusion::utils::srgbUint8ToFloat;   // uint8 sRGB -> the volume's float sRGB domain
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

// ---------------------------------------------------------------------------
// Fixture lattice. voxelToWorld(i) == origin + i*voxel_size, so voxel index k holds
// the field AT lattice position z = k*vs (origin.z = 0). A field written analytically
// at those lattice points is reconstructed EXACTLY (linearly along z; the field is
// z-only) by the trilinear sampler, so the zero crossing the raycast must resolve is
// computable on paper.
// ---------------------------------------------------------------------------
constexpr int   kRes      = 32;
constexpr float kVs       = 0.04f;          // lattice plane every 4 cm
constexpr float kTrunc    = 0.025f;         // integration-only parameter
constexpr float kOriginXY = -0.64f;         // x,y in [-0.64, 0.64]
constexpr float kMinDepth = 0.12f;
constexpr float kMaxDepth = 0.90f;

constexpr int   kImg = 8;                   // 8x8 synthetic frame
constexpr float kFx  = 4.0f, kFy = 4.0f;
constexpr float kCx  = 3.0f, kCy = 3.0f;    // pixel (3,3) is exactly on-axis
constexpr double kTol = 1e-5;               // required planar tolerance

// Wall: half-space solid behind z = kWallZ, sdf = Z0 - z, band tau = kVs.
//   lattice k=9  (z=0.36): f = (0.38-0.36)/0.04 = +0.5
//   lattice k=10 (z=0.40): f = (0.38-0.40)/0.04 = -0.5
//   the reconstruction is linear between them and vanishes at z = 0.38.
constexpr float  kWallZ    = 0.38f;
constexpr double kWallHitZ = 0.38;

// Slab: solid for z in [0.39, 0.41] (2 cm thick), band tau = 0.02.
//   lattice k=9 (+1), k=10 (-0.5), k=11 (+1)
//   front zero = 0.36 + 0.04*(1/(1+0.5))   = 0.3866666667
//   back  zero = 0.40 + 0.04*(0.5/(0.5+1)) = 0.4133333333
constexpr float  kSlabA = 0.39f, kSlabB = 0.41f, kSlabTau = 0.02f;
constexpr double kSlabFrontZ = 0.36 + 0.04 * (1.0 / 1.5);
constexpr double kSlabBackZ  = 0.40 + 0.04 * (0.5 / 1.5);

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();

float clamp1(float v) { return v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v); }

float wallField(float z) { return clamp1((kWallZ - z) / kVs); }

float slabField(float z) {
    const bool  in_slab = (z >= kSlabA && z <= kSlabB);
    const float sdf     = in_slab ? -std::min(z - kSlabA, kSlabB - z)
                                  : (z < kSlabA ? kSlabA - z : z - kSlabB);
    return clamp1(sdf / kSlabTau);
}

using FieldFn = float (*)(float);

// Every layer gets its own color, so "the color of the wrong voxel" is always
// observable. layerColor is fixture code, never product code.
struct Rgb {
    uint8_t r, g, b;
};
Rgb layerColor(int k) {
    return Rgb{static_cast<uint8_t>(31 + 7 * (k % 29)), static_cast<uint8_t>(200 - k),
               static_cast<uint8_t>(17 + 3 * (k % 11))};
}

TSDFParams fixtureParams(float min_depth, float max_depth) {
    TSDFParams p;
    p.resolution = kRes;
    p.voxel_size = kVs;
    p.truncation = kTrunc;
    p.max_weight = 128.0f;
    p.origin     = Eigen::Vector3f(kOriginXY, kOriginXY, 0.0f);
    p.min_depth  = min_depth;
    p.max_depth  = max_depth;
    return p;
}

// A volume whose lattice holds `field` analytically. weight > 0 everywhere, so every
// sample is an observed value and never the empty sentinel.
std::unique_ptr<TSDFVolume> makeVolume(float min_depth, float max_depth, FieldFn field) {
    auto vol = std::make_unique<TSDFVolume>(fixtureParams(min_depth, max_depth));
    for (int z = 0; z < kRes; ++z) {
        const Rgb c = layerColor(z);
        for (int y = 0; y < kRes; ++y) {
            for (int x = 0; x < kRes; ++x) {
                Voxel& v = vol->voxelAt(x, y, z);
                v.tsdf   = field(kVs * static_cast<float>(z));
                v.weight = 4.0f;
                v.r      = srgbUint8ToFloat(c.r);
                v.g      = srgbUint8ToFloat(c.g);
                v.b      = srgbUint8ToFloat(c.b);
            }
        }
    }
    return vol;
}

void poisonLayer(TSDFVolume& vol, int layer, float value) {
    for (int y = 0; y < kRes; ++y) {
        for (int x = 0; x < kRes; ++x) {
            Voxel& v = vol.voxelAt(x, y, layer);
            v.tsdf   = value;
            v.weight = 4.0f;
        }
    }
}

struct Hit {
    Eigen::Vector3f v{0.0f, 0.0f, 0.0f};
    Eigen::Vector3f n{0, 0, 0};
    uint8_t         c[3] = {0, 0, 0};
    bool            empty_state = false;
};

Hit castPixel(const TSDFVolume& vol, int px, int py) {
    const size_t npix = static_cast<size_t>(kImg) * kImg;
    std::vector<Eigen::Vector3f> verts(npix), norms(npix);
    std::vector<uint8_t>         cols(npix * 3);
    vol.raycast(Eigen::Matrix4f::Identity(), kFx, kFy, kCx, kCy, kImg, kImg, verts.data(),
                norms.data(), cols.data());
    Hit           h;
    const size_t  i = static_cast<size_t>(py) * kImg + px;
    h.v             = verts[i];
    h.n             = norms[i];
    h.c[0]          = cols[i * 3 + 0];
    h.c[1]          = cols[i * 3 + 1];
    h.c[2]          = cols[i * 3 + 2];
    h.empty_state   = (h.v == Eigen::Vector3f::Zero()) && (h.n == Eigen::Vector3f::Zero()) &&
                      h.c[0] == 0 && h.c[1] == 0 && h.c[2] == 0;
    return h;
}

bool near(double got, double want, double tol = kTol) { return std::fabs(got - want) <= tol; }
bool colorEq(const Hit& h, Rgb c) { return h.c[0] == c.r && h.c[1] == c.g && h.c[2] == c.b; }
// The byte an unobserved voxel's float sRGB color publishes as.
uint8_t emptyColorByte() {
    uint8_t b = 0;
    srgbFloatToUint8(EMPTY_COLOR, b);
    return b;
}
int  layerOf(double z) { return static_cast<int>(std::floor(z / kVs)); }
std::string rgbStr(uint8_t r, uint8_t g, uint8_t b) {
    return "(" + std::to_string(r) + "," + std::to_string(g) + "," + std::to_string(b) + ")";
}
std::string rgbStr(const Hit& h) { return rgbStr(h.c[0], h.c[1], h.c[2]); }

// ---------------------------------------------------------------------------
// A. planar surface on the optical axis: position, normal, color
// ---------------------------------------------------------------------------
void gateA_planarHit() {
    const std::string tag             = "A/planar-axis";
    const auto        vol             = makeVolume(kMinDepth, kMaxDepth, wallField);
    const Hit         h               = castPixel(*vol, 3, 3);

    CHECK(h.v.allFinite() && h.n.allFinite(), tag + ": hit is finite");
    CHECK(near(h.v.x(), 0.0) && near(h.v.y(), 0.0), tag + ": hit stays on the optical axis");
    CHECK(near(h.v.z(), kWallHitZ),
          tag + ": planar hit within 1e-5 m of the derived 0.38 m crossing, got " +
              std::to_string(h.v.z()));
    // Outward normal of sdf = Z0 - z is (0,0,-1); unit length is part of the contract.
    CHECK(near(h.n.x(), 0.0) && near(h.n.y(), 0.0) && near(h.n.z(), -1.0),
          tag + ": normal is the outward unit gradient (0,0,-1)");
    CHECK(near(h.n.norm(), 1.0), tag + ": normal is unit length");

    const int want_layer = layerOf(kWallHitZ);   // 9, mid-layer, float-robust
    const Rgb want       = layerColor(want_layer);
    CHECK(colorEq(h, want), tag + ": color is layer " + std::to_string(want_layer) +
                                "'s own RGB " + rgbStr(want.r, want.g, want.b) + ", got " + rgbStr(h));
    CHECK(!colorEq(h, layerColor(want_layer - 1)) && !colorEq(h, layerColor(want_layer + 1)),
          tag + ": color is not an adjacent layer's (no neighbour ghost)");
    const uint8_t empty_c = emptyColorByte();
    CHECK(!(h.c[0] == empty_c && h.c[1] == empty_c && h.c[2] == empty_c),
          tag + ": color is not the stale unobserved fill color");
    std::printf("A hit=(%.7f,%.7f,%.7f) normal=(%.4f,%.4f,%.4f) rgb=%s want_z=%.7f\n", h.v.x(),
                h.v.y(), h.v.z(), h.n.x(), h.n.y(), h.n.z(), rgbStr(h).c_str(), kWallHitZ);
}

// ---------------------------------------------------------------------------
// B. the configured band is a Z-depth band, not a ray-parameter band
// ---------------------------------------------------------------------------
// Pixel (7,3) -> ray_cam = (1,0,1), |ray_cam| = sqrt(2). The wall crossing has Z-depth
// 0.38, i.e. ray parameter t = 0.38*sqrt(2) = 0.53740. With max_depth = 0.42 the
// Z-depth reading gives t_far = 0.42*sqrt(2) = 0.59400 > t, so the ray HITS; reading
// max_depth as a ray parameter stops the march at 0.42 and MISSES.
void gateB_zDepthBand() {
    const std::string tag = "B/zdepth-band";
    const auto        vol = makeVolume(kMinDepth, 0.42f, wallField);
    const Hit         h   = castPixel(*vol, 7, 3);

    CHECK(!h.empty_state, tag + ": off-axis ray still hits inside the Z-depth band (a "
                            "ray-parameter far bound would stop at t=0.42 and miss)");
    CHECK(near(h.v.z(), kWallHitZ),
          tag + ": off-axis hit has Z-depth 0.38 within 1e-5 m, got " + std::to_string(h.v.z()));
    CHECK(near(h.v.x(), kWallHitZ),
          tag + ": off-axis hit is on the 45-degree ray (x == z), got " + std::to_string(h.v.x()));
    const Eigen::Vector3f dir(1.0f, 0.0f, 1.0f);
    CHECK(near(h.v.cross(dir).norm(), 0.0, 1e-5), tag + ": the hit lies on the pixel ray");
    std::printf("B off-axis hit=(%.7f,%.7f,%.7f) t=%.7f want_t=%.7f\n", h.v.x(), h.v.y(), h.v.z(),
                static_cast<double>(h.v.norm()), kWallHitZ * 1.4142135623730951);
}

// ---------------------------------------------------------------------------
// C. the configured near/far bounds are the gate (same field, different band)
// ---------------------------------------------------------------------------
// min_depth 0.39 starts the march inside the solid half-space, which has no exit
// inside max_depth, so the configured band - not a literal - decides.
void gateC_configuredBounds() {
    const std::string tag      = "C/configured-bounds";
    const auto        open_    = makeVolume(kMinDepth, kMaxDepth, wallField);
    const auto        shut     = makeVolume(0.39f, kMaxDepth, wallField);
    const auto        far_shut = makeVolume(kMinDepth, 0.30f, wallField);

    const Hit in      = castPixel(*open_, 3, 3);
    const Hit out     = castPixel(*shut, 3, 3);
    const Hit out_far = castPixel(*far_shut, 3, 3);

    CHECK(!in.empty_state, tag + ": min_depth=0.12 hits the 0.38 m wall");
    CHECK(out.empty_state, tag + ": min_depth=0.39 (past the wall) returns the empty state, got (" +
                                std::to_string(out.v.x()) + "," + std::to_string(out.v.y()) + "," +
                                std::to_string(out.v.z()) + ")");
    CHECK(out_far.empty_state, tag + ": max_depth=0.30 misses the 0.38 m wall");
    CHECK(shut->params().min_depth == 0.39f && open_->params().min_depth == kMinDepth &&
              far_shut->params().max_depth == 0.30f,
          tag + ": the band is the volume's configured TSDF band");
    std::printf("C open_hit_z=%.7f near_shut=%d far_shut=%d\n", in.v.z(), out.empty_state ? 1 : 0,
                out_far.empty_state ? 1 : 0);
}

// ---------------------------------------------------------------------------
// D. both crossings resolve; the exit hit is the anti-ghost discriminator
// ---------------------------------------------------------------------------
// Exterior start -> ENTRY zero 0.3866667 (layer 9 color).
// Interior start (min_depth 0.402, inside the slab) -> EXIT zero 0.4133333, which
// lives in layer 10. Handing back the entry-layer color for an interior-start ray is
// exactly the stale/adjacent-surface ghost this todo removes.
void gateD_bothCrossings() {
    const std::string tag = "D/both-crossings";
    const auto        ext = makeVolume(kMinDepth, kMaxDepth, slabField);
    const auto        inn = makeVolume(0.402f, kMaxDepth, slabField);

    const Hit entry = castPixel(*ext, 3, 3);
    CHECK(near(entry.v.z(), kSlabFrontZ),
          tag + ": entry crossing at 0.3866667 within 1e-5 m, got " + std::to_string(entry.v.z()));
    const Rgb front_layer = layerColor(layerOf(kSlabFrontZ));
    CHECK(colorEq(entry, front_layer), tag + ": entry color is the entry layer's RGB " +
                                            rgbStr(front_layer.r, front_layer.g, front_layer.b) +
                                            ", got " + rgbStr(entry));

    const Hit exit_ = castPixel(*inn, 3, 3);
    CHECK(!exit_.empty_state, tag + ": a ray that starts inside material resolves a crossing");
    CHECK(near(exit_.v.z(), kSlabBackZ),
          tag + ": EXIT crossing at 0.4133333 within 1e-5 m, got " + std::to_string(exit_.v.z()));
    CHECK(!near(exit_.v.z(), kSlabFrontZ),
          tag + ": the interior-start hit is the back face, not a fabricated front face");
    const Rgb back_face = layerColor(layerOf(kSlabBackZ));
    CHECK(colorEq(exit_, back_face), tag + ": exit color is the back-face layer's RGB " +
                                            rgbStr(back_face.r, back_face.g, back_face.b) +
                                            ", got " + rgbStr(exit_));
    CHECK(!colorEq(exit_, front_layer),
          tag + ": exit color is NOT the entry layer's stale color (no ghost)");
    CHECK(near(exit_.n.norm(), 1.0, 1e-4), tag + ": the exit normal is unit length");
    std::printf("D entry_z=%.7f want=%.7f | exit_z=%.7f want=%.7f rgb=%s want=%s\n", entry.v.z(),
                kSlabFrontZ, exit_.v.z(), kSlabBackZ, rgbStr(exit_).c_str(),
                rgbStr(back_face.r, back_face.g, back_face.b).c_str());
}

// ---------------------------------------------------------------------------
// E. non-finite volume samples are never emitted
// ---------------------------------------------------------------------------
void gateE_nonFinite() {
    const std::string tag  = "E/non-finite";
    auto              ctrl = makeVolume(kMinDepth, kMaxDepth, wallField);
    const Hit         ref  = castPixel(*ctrl, 3, 3);
    CHECK(near(ref.v.z(), kWallHitZ), tag + ": control hit is intact");

    struct Case {
        float       value;
        const char* name;
    };
    const Case cases[] = {{kNaN, "NaN"}, {kInf, "+Inf"}, {-kInf, "-Inf"}};
    for (const Case& c : cases) {
        auto vol = makeVolume(kMinDepth, kMaxDepth, wallField);
        poisonLayer(*vol, 10, c.value);   // layer 10 is in the 0.38 m crossing stencil
        const Hit h = castPixel(*vol, 3, 3);
        CHECK(h.empty_state, std::string(tag) + ": a " + c.name +
                                 " sample at the crossing yields the empty state, not a hit (" +
                                 std::to_string(h.v.x()) + "," + std::to_string(h.v.y()) + "," +
                                 std::to_string(h.v.z()) + ")");
        CHECK(h.v.allFinite() && h.n.allFinite(),
              std::string(tag) + ": " + c.name + " never emits a non-finite vertex/normal");
        CHECK(h.c[0] == 0 && h.c[1] == 0 && h.c[2] == 0,
              std::string(tag) + ": " + c.name + " never emits a color for a rejected ray");
    }

    // Over-rejection guard: a non-finite layer BEHIND the resolved surface must not
    // erase the hit already resolved in front of it.
    auto      behind   = makeVolume(kMinDepth, kMaxDepth, wallField);
    poisonLayer(*behind, 20, kNaN);   // z = 0.80, past the 0.38 m wall
    const Hit h_behind = castPixel(*behind, 3, 3);
    CHECK(near(h_behind.v.z(), kWallHitZ),
          tag + ": a NaN behind the surface leaves the resolved hit intact");
    CHECK(h_behind.v == ref.v && h_behind.n == ref.n,
          tag + ": the surviving hit is bit-identical to the control hit");
    std::printf("E rejected NaN/+Inf/-Inf at the crossing; NaN behind it kept hit z=%.7f\n",
                h_behind.v.z());
}

// ---------------------------------------------------------------------------
// F. whole-frame output invariant + determinism
// ---------------------------------------------------------------------------
void gateF_frameInvariant() {
    const std::string tag  = "F/frame-invariant";
    const auto        vol  = makeVolume(kMinDepth, kMaxDepth, wallField);
    const size_t      npix = static_cast<size_t>(kImg) * kImg;
    std::vector<Eigen::Vector3f> v1(npix), n1(npix), v2(npix), n2(npix);
    std::vector<uint8_t>         c1(npix * 3), c2(npix * 3);

    vol->raycast(Eigen::Matrix4f::Identity(), kFx, kFy, kCx, kCy, kImg, kImg, v1.data(),
                 n1.data(), c1.data());

    int hits = 0, empty = 0, bad = 0;
    for (size_t i = 0; i < npix; ++i) {
        const Eigen::Vector3f& v = v1[i];
        const Eigen::Vector3f& n = n1[i];
        const bool e = (v == Eigen::Vector3f::Zero()) && (n == Eigen::Vector3f::Zero()) &&
                       c1[i * 3] == 0 && c1[i * 3 + 1] == 0 && c1[i * 3 + 2] == 0;
        if (e) { ++empty; continue; }
        ++hits;
        // A non-empty entry must be a finite hit inside the configured Z-depth band,
        // on the wall, with a unit normal. Anything else is a defect.
        if (!v.allFinite() || !n.allFinite()) { ++bad; continue; }
        if (v.z() < kMinDepth - 1e-4f || v.z() > kMaxDepth + 1e-4f) { ++bad; continue; }
        if (std::fabs(static_cast<double>(v.z()) - kWallHitZ) > kTol) { ++bad; continue; }
        if (std::fabs(static_cast<double>(n.norm()) - 1.0) > 1e-4) { ++bad; continue; }
    }
    CHECK(hits == kImg * kImg, tag + ": every pixel hits the wall, hits=" + std::to_string(hits));
    CHECK(empty == 0,
          tag + ": no pixel fell back to the empty state, empty=" + std::to_string(empty));
    CHECK(bad == 0,
          tag + ": every hit is finite, in band, on the wall, unit-normal, bad=" + std::to_string(bad));

    vol->raycast(Eigen::Matrix4f::Identity(), kFx, kFy, kCx, kCy, kImg, kImg, v2.data(),
                 n2.data(), c2.data());
    int diff = 0;
    for (size_t i = 0; i < npix; ++i) {
        if (v1[i] != v2[i] || n1[i] != n2[i]) ++diff;
    }
    for (size_t i = 0; i < c1.size(); ++i) {
        if (c1[i] != c2[i]) ++diff;
    }
    CHECK(diff == 0, tag + ": a repeated raycast is bit-identical, diff=" + std::to_string(diff));
    std::printf("F hits=%d empty=%d bad=%d deterministic=%d\n", hits, empty, bad, diff == 0 ? 1 : 0);
}

} // namespace

int main() {
    gateA_planarHit();
    gateB_zDepthBand();
    gateC_configuredBounds();
    gateD_bothCrossings();
    gateE_nonFinite();
    gateF_frameInvariant();

    if (g_failures == 0) {
        std::printf("tsdf_raycast_contract: PASS (%d checks: 1e-5 m planar hit, Z-depth band, "
                    "configured bounds, entry+exit crossings, NaN/Inf rejection, invariant)\n",
                    g_checks);
        return 0;
    }
    std::printf("tsdf_raycast_contract: FAIL (%d failed checks of %d)\n", g_failures, g_checks);
    return 1;
}
