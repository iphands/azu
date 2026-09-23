// color_convergence_contract (big-fix Todo 19): CPU-only contract for the canonical
// color chain (docs/CANONICAL_SEMANTICS.md, "Color pipeline and export"):
//   device uint8 sRGB -> volume float sRGB [0,1] -> extraction-boundary uint8 sRGB.
// Public CPU API only (TSDFVolume::integrate/raycast/extractGlobalPointCloud + the
// shared utils::ColorMath policy); no device, display, GPU, sensor, thread timing,
// filesystem or network.
//
// Canonical rules under test:
//   K1 widening is one division by 255 (no gamma: the volume domain IS sRGB-encoded),
//      is strictly increasing over all 256 bytes and lands inside [0,1];
//   K2 every boundary that turns a float back into a byte uses the ONE shared policy,
//      which rounds to nearest and round-trips all 256 bytes exactly;
//   K3 ONE clamp policy at every extraction boundary: every FINITE value outside
//      [0,1] saturates to the endpoint byte (clamping precedes the multiply, so even
//      FLT_MAX cannot overflow the scale) while NaN and +/-Inf are refused and leave
//      the output byte untouched - a raw static_cast<uint8_t> of a non-finite float is
//      undefined behaviour, and std::min/std::max alone cannot express the distinction;
//   K4 repeated fusion converges to the true mean instead of stalling in a per-update
//      uint8 rounding dead zone, and the fused float stays quantizable at every voxel;
//   K5 the emitted byte at each extraction boundary equals the shared policy applied to
//      the voxel it came from, so no second clamp/rounding policy can exist;
//   K6 the raycast publishes the policy applied to the voxel the hit falls in: a real
//      resolved hit (never the zero non-hit state) is required before any color claim,
//      a non-finite voxel color WITHDRAWS the hit, and a finite out-of-range voxel
//      color keeps the hit and publishes the saturated byte;
//   K7 Marching Cubes saturates a finite out-of-range endpoint or blend (a uniform
//      out-of-range color yields a full mesh whose every byte is the endpoint byte,
//      with geometry bit-identical to the in-range baseline) and drops the crossing
//      edge only for a non-finite endpoint or blend (a uniform NaN or Inf color yields
//      an empty mesh over the same field that otherwise meshes).
//
// Sections that were already green before Todo 19 are labelled LOCK so the evidence
// never overclaims a fail-before-fix; the dead-zone/rounding sections are labelled
// FIX and are the ones the uint8 voxel failed.

#include "meshing/MarchingCubes.h"
#include "meshing/MeshData.h"
#include "tsdf/TSDFVolume.h"
#include "utils/ColorMath.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

using kfusion::meshing::MarchingCubes;
using kfusion::meshing::MeshData;
using kfusion::tsdf::EMPTY_COLOR;
using kfusion::tsdf::TSDFParams;
using kfusion::tsdf::TSDFVolume;
using kfusion::tsdf::Voxel;
using kfusion::utils::srgbFloatToUint8;   // the one CPU float sRGB -> byte policy
using kfusion::utils::srgbUint8ToFloat;   // uint8 sRGB -> the volume's float sRGB domain

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, what)                                                       \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, std::string(cond ? "" : what).c_str()); \
        }                                                                       \
    } while (false)

std::string& g_reason() { static std::string s; return s; }

const float* nanF() {
    static const float v = std::numeric_limits<float>::quiet_NaN();
    return &v;
}
const float* infF() {
    static const float v = std::numeric_limits<float>::infinity();
    return &v;
}

// ---------------------------------------------------------------------------
// K1 widening: exact, strictly increasing, inside [0,1]
// ---------------------------------------------------------------------------
void k1_widening() {
    CHECK(srgbUint8ToFloat(0) == 0.0f, "K1: byte 0 widens to exactly 0.0");
    CHECK(srgbUint8ToFloat(255) == 1.0f, "K1: byte 255 widens to exactly 1.0 (no overshoot)");
    CHECK(srgbUint8ToFloat(128) == 128.0f / 255.0f, "K1: 128 widens to the exact EMPTY_COLOR quotient");

    bool monotonic = true, in_range = true, exact_division = true;
    for (int b = 0; b <= 255; ++b) {
        const float f = srgbUint8ToFloat(static_cast<uint8_t>(b));
        if (b > 0 && !(f > srgbUint8ToFloat(static_cast<uint8_t>(b - 1)))) monotonic = false;
        if (!(f >= 0.0f && f <= 1.0f)) in_range = false;
        // One division, no gamma, no LUT: the widened value is b/255 to the last bit.
        if (f != static_cast<float>(b) / 255.0f) exact_division = false;
    }
    CHECK(monotonic, "K1: widening is strictly increasing across all 256 bytes");
    CHECK(in_range, "K1: every widened value is inside [0,1]");
    CHECK(exact_division, "K1: widening is exactly b/255 for all 256 bytes (no gamma applied)");

    // A raw byte value in a float color field is out of range, not a normalized
    // color. The boundary saturates it to 255 rather than wrapping it, so the
    // mistake a fixture makes stays visible instead of turning random.
    uint8_t probe = 7;
    CHECK(srgbFloatToUint8(srgbUint8ToFloat(200) * 255.0f, probe) && probe == 255,
          "K1: the un-normalized byte value 200 in a float color field saturates to 255");
    std::printf("K1 widen(11)=%.9f widen(128)=%.9f widen(244)=%.9f\n",
                srgbUint8ToFloat(11), srgbUint8ToFloat(128), srgbUint8ToFloat(244));
}

// ---------------------------------------------------------------------------
// K2 round-trip identity for all 256 bytes
// ---------------------------------------------------------------------------
void k2_round_trip() {
    bool identity = true, all_accepted = true;
    for (int b = 0; b <= 255; ++b) {
        const float  f = srgbUint8ToFloat(static_cast<uint8_t>(b));
        uint8_t      q = 0;
        const bool   ok = srgbFloatToUint8(f, q);
        if (!ok) all_accepted = false;
        else if (q != static_cast<uint8_t>(b)) identity = false;
    }
    CHECK(all_accepted, "K2: every widened byte is accepted by the quantizer");
    CHECK(identity, "K2: widen->quantize is the identity on all 256 bytes");

    uint8_t zero = 0, one = 0;
    CHECK(srgbFloatToUint8(0.0f, zero) && zero == 0, "K2: 0.0 -> byte 0");
    CHECK(srgbFloatToUint8(1.0f, one) && one == 255, "K2: 1.0 -> byte 255");
}

// ---------------------------------------------------------------------------
// K3 ONE clamp policy: every finite value saturates, only non-finite is rejected
// ---------------------------------------------------------------------------
void k3_clamp_and_reject() {
    struct Sat {
        float   in;
        uint8_t want;
    };
    // A finite value outside [0,1] is an interpolation overshoot and saturates to
    // the endpoint byte. Clamping precedes the multiply, so even the widest finite
    // float lands on an endpoint instead of overflowing the scale.
    const Sat sat[] = {
        {-0.001f, 0},
        {-0.5f, 0},
        {-1.0f, 0},
        {-std::numeric_limits<float>::max(), 0},
        {std::numeric_limits<float>::lowest(), 0},
        {-1e30f, 0},
        {1.001f, 255},
        {1.5f, 255},
        {2.0f, 255},
        {1e30f, 255},
        {std::numeric_limits<float>::max(), 255},
    };
    for (size_t i = 0; i < sizeof sat / sizeof sat[0]; ++i) {
        uint8_t     out = 177;  // a finite value must overwrite this sentinel
        const bool  ok  = srgbFloatToUint8(sat[i].in, out);
        CHECK(ok, "K3: a finite out-of-range value is accepted, not refused");
        CHECK(out == sat[i].want, "K3: a finite out-of-range value saturates to the endpoint byte");
    }

    // Non-finite is refused outright and the output byte stays untouched, so a
    // caller can never publish a half-written triple built from an upstream bug.
    const float non_finite[] = {*nanF(), -*nanF(), *infF(), -*infF()};
    for (size_t i = 0; i < sizeof non_finite / sizeof non_finite[0]; ++i) {
        uint8_t     out = 177;
        const bool  ok  = srgbFloatToUint8(non_finite[i], out);
        CHECK(!ok, "K3: NaN and +/-Inf are rejected, never saturated");
        CHECK(out == 177, "K3: a rejected value leaves the output byte untouched");
    }
    uint8_t out = 3;
    CHECK(!srgbFloatToUint8(*nanF(), out) && out == 3, "K3: quiet NaN is refused and out stays 3");

    // -0.0 is in range and must quantize to 0 rather than be refused.
    uint8_t neg_zero = 9;
    CHECK(srgbFloatToUint8(-0.0f, neg_zero) && neg_zero == 0, "K3: negative zero is in range and is byte 0");
    std::printf("K3 saturated %zu finite values, rejected %zu non-finite values\n",
                sizeof sat / sizeof sat[0], sizeof non_finite / sizeof non_finite[0]);
}

// ---------------------------------------------------------------------------
// K4a rounding is nearest, not truncation  [FIX: the uint8 stage truncated]
// ---------------------------------------------------------------------------
void k4_rounding() {
    // (b + 0.6)/255 is closer to b+1 than to b, so nearest gives b+1 and the old
    // truncating cast gives b. One value per byte, no tolerance needed.
    bool nearest = true, differs_from_truncation = true;
    for (int b = 0; b < 255; ++b) {
        const float v = (static_cast<float>(b) + 0.6f) / 255.0f;
        uint8_t     q = 0;
        if (!srgbFloatToUint8(v, q)) { nearest = false; continue; }
        if (q != static_cast<uint8_t>(b + 1)) nearest = false;
        if (static_cast<uint8_t>(v * 255.0f) != static_cast<uint8_t>(b)) differs_from_truncation = false;
    }
    CHECK(nearest, "K4: quantization rounds to nearest for every byte's upper half-step");
    CHECK(differs_from_truncation,
          "K4: the policy is demonstrably not the old truncating cast (it would land one byte low)");

    uint8_t q = 0;
    CHECK(srgbFloatToUint8(0.999f, q) && q == 255, "K4: 0.999 -> 255 (truncation gave 254)");
    CHECK(srgbFloatToUint8(128.5f / 255.0f, q) && q == 129, "K4: a half-step rounds away from zero");
    CHECK(srgbFloatToUint8(127.5f / 255.0f, q) && q == 128, "K4: 127.5/255 -> 128");
}

// ---------------------------------------------------------------------------
// K4b the empty sentinel is a real color in the float domain  [LOCK + new byte]
// ---------------------------------------------------------------------------
void k4b_empty_sentinel() {
    CHECK(EMPTY_COLOR == 128.0f / 255.0f, "K5: EMPTY_COLOR is the exact 128/255 quotient");
    CHECK(EMPTY_COLOR > 0.0f && EMPTY_COLOR < 1.0f, "K5: EMPTY_COLOR is inside the float sRGB domain");
    uint8_t q = 0;
    CHECK(srgbFloatToUint8(EMPTY_COLOR, q) && q == 128, "K5: EMPTY_COLOR quantizes back to byte 128");
    Voxel v;
    CHECK(srgbFloatToUint8(v.r, q) && q == 128, "K5: a default voxel's red publishes as byte 128");
    std::printf("K5 EMPTY_COLOR=%.9f byte=%u\n", EMPTY_COLOR, q);
}

// ---------------------------------------------------------------------------
// Fixture volume: one flat wall, per-pixel colors, identity pose
// ---------------------------------------------------------------------------
constexpr int   kRes = 48;
constexpr float kVs  = 0.02f;
constexpr int   kW   = 40, kH = 30;

TSDFParams fixtureParams() {
    TSDFParams p;
    p.resolution = kRes;
    p.voxel_size = kVs;
    p.truncation = 0.05f;
    p.max_weight = 128.0f;
    p.origin     = Eigen::Vector3f(-0.48f, -0.48f, 0.0f);
    return p;
}

struct Frames {
    std::vector<float>    depth;
    std::vector<uint8_t>  rgb;
};

Frames makeFrame(uint8_t r, uint8_t g, uint8_t b) {
    Frames f;
    f.depth.assign(static_cast<size_t>(kW) * kH, 0.5f);
    f.rgb.assign(static_cast<size_t>(kW) * kH * 3, 0);
    for (int i = 0; i < kW * kH; ++i) {
        f.rgb[static_cast<size_t>(i) * 3 + 0] = r;
        f.rgb[static_cast<size_t>(i) * 3 + 1] = g;
        f.rgb[static_cast<size_t>(i) * 3 + 2] = b;
    }
    return f;
}

// A fused SURFACE voxel. Voxel-projective integration also observes free space
// in front of the surface (tsdf == 1, weight > 0), and colour is only fused
// within half a truncation of the surface, so "first voxel with weight" would
// pick an uncoloured free-space voxel.
const Voxel& observedVoxel(const TSDFVolume& vol) {
    const std::vector<Voxel>& all = vol.voxelData();
    for (size_t i = 0; i < all.size(); ++i) {
        if (all[i].weight > 0.0f && std::fabs(all[i].tsdf) < 0.25f) return all[i];
    }
    return all.front();
}

// ---------------------------------------------------------------------------
// K4c fusion convergence
// ---------------------------------------------------------------------------
void k4c_convergence() {
    const uint8_t R = 200, G = 100, B = 40;

    // One update from the neutral sentinel: exactly the normalized byte, and the
    // byte stage returns the frame's own RGB.
    {
        TSDFVolume       vol(fixtureParams());
        const Frames     f = makeFrame(R, G, B);
        vol.integrate(f.depth.data(), f.rgb.data(), Eigen::Matrix4f::Identity(), 300.f, 300.f,
                      19.5f, 14.5f, kW, kH, 0.1f, 3.0f);
        const Voxel& v = observedVoxel(vol);
        CHECK(v.weight > 0.0f, "K4: the fixture volume has an observed voxel");
        CHECK(std::fabs(static_cast<double>(v.r) - static_cast<double>(R) / 255.0) < 1e-6,
              "K4: one update lands on the normalized input byte, not on the byte itself");
        uint8_t q = 0;
        CHECK(srgbFloatToUint8(v.r, q) && q == R, "K4: one update publishes the frame's exact R byte");
        std::printf("K4c single r=%.9f -> byte %u (frame R=%u)\n", v.r, q, R);
    }

    // Repeated identical frames: a uint8 accumulator stalls after the first
    // rounding; the float accumulator stays pinned on the true value.
    {
        TSDFVolume   vol(fixtureParams());
        const Frames f = makeFrame(R, G, B);
        for (int n = 0; n < 32; ++n) {
            vol.integrate(f.depth.data(), f.rgb.data(), Eigen::Matrix4f::Identity(), 300.f, 300.f,
                          19.5f, 14.5f, kW, kH, 0.1f, 3.0f);
        }
        const Voxel& v   = observedVoxel(vol);
        const double want = static_cast<double>(R) / 255.0;
        CHECK(std::fabs(static_cast<double>(v.r) - want) < 1e-5,
              "K4: 32 identical frames converge to the normalized byte, no rounding dead zone");
        uint8_t q = 0;
        CHECK(srgbFloatToUint8(v.r, q) && q == R, "K4: the converged float still publishes the input byte");
        std::printf("K4c x32 r=%.9f want=%.9f byte=%u\n", v.r, want, q);
    }

    // Alternating two colors: the running mean stays strictly between the two
    // widened endpoints (never outside [0,1]) and every voxel stays quantizable.
    {
        TSDFVolume        vol(fixtureParams());
        const Frames      dark = makeFrame(20, 20, 20);
        const Frames      lit  = makeFrame(230, 230, 230);
        const double      lo   = static_cast<double>(20) / 255.0;
        const double      hi   = static_cast<double>(230) / 255.0;
        bool              bounded = true, quantizable = true;
        float             first_r = 0.0f, last_r = 0.0f;
        for (int n = 0; n < 24; ++n) {
            const Frames& f = (n % 2 == 0) ? dark : lit;
            vol.integrate(f.depth.data(), f.rgb.data(), Eigen::Matrix4f::Identity(), 300.f, 300.f,
                          19.5f, 14.5f, kW, kH, 0.1f, 3.0f);
            const Voxel& v = observedVoxel(vol);
            if (!(v.r >= lo - 1e-6 && v.r <= hi + 1e-6)) bounded = false;
            if (n == 0) first_r = v.r;
            last_r = v.r;
        }
        for (const Voxel& v : vol.voxelData()) {
            if (v.weight <= 0.0f) continue;
            uint8_t q = 0;
            if (!srgbFloatToUint8(v.r, q) || !srgbFloatToUint8(v.g, q) || !srgbFloatToUint8(v.b, q))
                quantizable = false;
        }
        CHECK(bounded, "K4: the fused color stays inside the widened input range, never outside [0,1]");
        CHECK(quantizable, "K4: every observed voxel of a fused volume is quantizable by the shared policy");
        CHECK(last_r > first_r, "K4: ending on the brighter frame raises the fused color (a uint8 fold would saturate)");
        std::printf("K4c alternate first=%.9f last=%.9f bounds=[%.9f,%.9f]\n", first_r, last_r, lo, hi);
    }
}

// ---------------------------------------------------------------------------
// K5 extraction boundaries use the one policy
// ---------------------------------------------------------------------------
void k5_boundaries() {
    TSDFVolume        vol(fixtureParams());
    const Frames      f = makeFrame(11, 128, 244);   // includes the sRGB knee neighbourhood
    for (int n = 0; n < 5; ++n) {
        vol.integrate(f.depth.data(), f.rgb.data(), Eigen::Matrix4f::Identity(), 300.f, 300.f, 19.5f,
                      14.5f, kW, kH, 0.1f, 3.0f);
    }

    std::vector<Eigen::Vector3f> pts;
    std::vector<uint8_t>         cols;
    vol.extractGlobalPointCloud(pts, cols);
    CHECK(!pts.empty(), "K5: the fused volume yields a point cloud");
    CHECK(cols.size() == pts.size() * 3, "K5: the point cloud carries one RGB triple per point");
    bool emitted_valid = true;
    for (size_t i = 0; i + 2 < cols.size(); i += 3) {
        uint8_t q = 0;
        if (!srgbFloatToUint8(srgbUint8ToFloat(cols[i]), q) || q != cols[i]) emitted_valid = false;
    }
    CHECK(emitted_valid, "K5: every emitted point-cloud byte round-trips through the shared policy");
    if (!cols.empty()) {
        const int e = static_cast<int>(srgbUint8ToFloat(EMPTY_COLOR) * 255.0 + 0.5);
        CHECK(!(cols[0] == e && cols[1] == e && cols[2] == e),
              "K5: an observed point does not publish the unobserved sentinel byte");
        std::printf("K5 point[0]=(%u,%u,%u) frame=(11,128,244)\n", cols[0], cols[1], cols[2]);
    }

    // The raycast boundary itself is K6, which demands a real resolved hit before
    // it says anything about color.
}

// ---------------------------------------------------------------------------
// K6 raycast boundary: a real hit, the policy applied to the hit voxel, and the
//    two poison classes (non-finite withdraws the hit, finite out-of-range saturates)
//
// The lattice is written analytically so the zero crossing is computable on paper:
// voxelToWorld(i) == origin + i*voxel_size, so voxel k holds the field at
// z = k*vs and the trilinear sampler reconstructs a z-only field exactly.
// ---------------------------------------------------------------------------
constexpr int   kRcRes      = 32;
constexpr float kRcVs       = 0.04f;      // lattice plane every 4 cm
constexpr float kRcTrunc    = 0.025f;
constexpr float kRcOriginXY = -0.64f;     // x,y in [-0.64, 0.64]
constexpr int   kRcImg      = 8;
constexpr float kRcFx = 4.0f, kRcFy = 4.0f, kRcCx = 3.0f, kRcCy = 3.0f;
constexpr float kRcWallZ    = 0.38f;      // k=9 -> +0.5, k=10 -> -0.5, zero at 0.38
constexpr double kRcHitZ    = 0.38;

struct RcHit {
    Eigen::Vector3f v = Eigen::Vector3f::Zero();
    Eigen::Vector3f n = Eigen::Vector3f::Zero();
    uint8_t         c[3] = {0, 0, 0};
};

TSDFParams rcParams() {
    TSDFParams p;
    p.resolution = kRcRes;
    p.voxel_size = kRcVs;
    p.truncation = kRcTrunc;
    p.max_weight = 128.0f;
    p.origin     = Eigen::Vector3f(kRcOriginXY, kRcOriginXY, 0.0f);
    p.min_depth  = 0.12f;
    p.max_depth  = 0.90f;
    return p;
}

float rcClamp1(float v) { return v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v); }

// Half-space solid behind z = kRcWallZ: sdf = Z0 - z, band tau = kRcVs. Every voxel
// is observed (weight > 0) so the sampler never falls back to the empty sentinel.
std::unique_ptr<TSDFVolume> makeWallVolume(float cr, float cg, float cb) {
    auto vol = std::make_unique<TSDFVolume>(rcParams());
    for (int z = 0; z < kRcRes; ++z) {
        const float f = rcClamp1((kRcWallZ - kRcVs * static_cast<float>(z)) / kRcVs);
        for (int y = 0; y < kRcRes; ++y) {
            for (int x = 0; x < kRcRes; ++x) {
                Voxel& v = vol->voxelAt(x, y, z);
                v.tsdf   = f;
                v.weight = 4.0f;
                v.r      = cr;
                v.g      = cg;
                v.b      = cb;
            }
        }
    }
    return vol;
}

std::unique_ptr<TSDFVolume> makeWallVolume(const float* per_layer_r, const float* per_layer_g,
                                           const float* per_layer_b) {
    auto vol = std::make_unique<TSDFVolume>(rcParams());
    for (int z = 0; z < kRcRes; ++z) {
        const float f = rcClamp1((kRcWallZ - kRcVs * static_cast<float>(z)) / kRcVs);
        for (int y = 0; y < kRcRes; ++y) {
            for (int x = 0; x < kRcRes; ++x) {
                Voxel& v = vol->voxelAt(x, y, z);
                v.tsdf   = f;
                v.weight = 4.0f;
                v.r      = per_layer_r[z];
                v.g      = per_layer_g[z];
                v.b      = per_layer_b[z];
            }
        }
    }
    return vol;
}

RcHit castCenter(const TSDFVolume& vol) {
    const size_t npix = static_cast<size_t>(kRcImg) * kRcImg;
    std::vector<Eigen::Vector3f> verts(npix), norms(npix);
    std::vector<uint8_t>         cols(npix * 3, 0);
    vol.raycast(Eigen::Matrix4f::Identity(), kRcFx, kRcFy, kRcCx, kRcCy, kRcImg, kRcImg,
                verts.data(), norms.data(), cols.data());
    RcHit h;
    const size_t i = static_cast<size_t>(3) * kRcImg + 3;   // exactly on the optical axis
    h.v            = verts[i];
    h.n            = norms[i];
    h.c[0]         = cols[i * 3 + 0];
    h.c[1]         = cols[i * 3 + 1];
    h.c[2]         = cols[i * 3 + 2];
    return h;
}

bool isResolved(const Eigen::Vector3f& v) { return v != Eigen::Vector3f::Zero(); }

void k6_raycast_boundary() {
    // Layer 9 is where the resolved hit falls (0.38 / 0.04 = 9.5 -> voxel 9), and its
    // color is the ONLY admissible answer: an adjacent layer's color is a neighbour
    // ghost, and the shared policy applied to that voxel is the byte itself here.
    std::vector<float> lr(kRcRes), lg(kRcRes), lb(kRcRes);
    for (int z = 0; z < kRcRes; ++z) {
        lr[z] = srgbUint8ToFloat(static_cast<uint8_t>(17 + 3 * z));
        lg[z] = srgbUint8ToFloat(static_cast<uint8_t>(200 - z));
        lb[z] = srgbUint8ToFloat(static_cast<uint8_t>(91));
    }
    const auto vol = makeWallVolume(lr.data(), lg.data(), lb.data());
    const RcHit h  = castCenter(*vol);

    // The precondition every color claim below depends on: a REAL hit, not the
    // finite-but-empty (0,0,0) non-hit state an isfinite() test would wave through.
    CHECK(isResolved(h.v), "K6: the centre pixel resolves a hit (not the zero non-hit state)");
    CHECK(std::fabs(static_cast<double>(h.v.z()) - kRcHitZ) <= 1e-5,
          "K6: the hit sits on the paper-derived 0.38 m crossing");
    CHECK(std::fabs(h.n.norm() - 1.0f) < 1e-4f, "K6: the hit carries a unit normal");

    const int hit_layer = static_cast<int>(std::floor(kRcHitZ / kRcVs));
    uint8_t   qr = 0, qg = 0, qb = 0;
    const bool ok = srgbFloatToUint8(lr[hit_layer], qr) && srgbFloatToUint8(lg[hit_layer], qg) &&
                    srgbFloatToUint8(lb[hit_layer], qb);
    CHECK(ok, "K6: the hit voxel's color is representable");
    CHECK(h.c[0] == qr && h.c[1] == qg && h.c[2] == qb,
          "K6: the emitted RGB is the shared policy applied to the hit voxel");
    const bool not_neighbour =
        !(h.c[0] == static_cast<uint8_t>(17 + 3 * (hit_layer - 1))) &&
        !(h.c[0] == static_cast<uint8_t>(17 + 3 * (hit_layer + 1)));
    CHECK(not_neighbour, "K6: the emitted color is not an adjacent layer's (no neighbour ghost)");
    std::printf("K6 hit=(%.4f,%.4f,%.4f) layer=%d rgb=(%u,%u,%u) policy=(%u,%u,%u)\n", h.v.x(),
                h.v.y(), h.v.z(), hit_layer, h.c[0], h.c[1], h.c[2], qr, qg, qb);

    // Class 1 poison - non-finite: the hit is withdrawn, not recolored.
    {
        const auto poisoned = makeWallVolume(*nanF(), srgbUint8ToFloat(128), srgbUint8ToFloat(91));
        const RcHit p       = castCenter(*poisoned);
        CHECK(!isResolved(p.v) && p.n == Eigen::Vector3f::Zero(),
              "K6: a NaN voxel color withdraws the whole surface output for the pixel");
        const auto inf_poison = makeWallVolume(srgbUint8ToFloat(17), srgbUint8ToFloat(128), *infF());
        const RcHit i         = castCenter(*inf_poison);
        CHECK(!isResolved(i.v) && i.n == Eigen::Vector3f::Zero(),
              "K6: a +Inf voxel color withdraws the whole surface output for the pixel");
        const auto ninf_poison = makeWallVolume(srgbUint8ToFloat(17), -*infF(), srgbUint8ToFloat(91));
        const RcHit q            = castCenter(*ninf_poison);
        CHECK(!isResolved(q.v), "K6: a -Inf voxel color withdraws the hit too");
    }

    // Class 2 poison - finite but out of range: the hit SURVIVES and publishes the
    // saturated byte. This is the correction the clamp policy demands: saturate, do
    // not withdraw.
    {
        const auto over = makeWallVolume(-0.25f, 1.5f, 2.0f);
        const RcHit p   = castCenter(*over);
        CHECK(isResolved(p.v), "K6: a finite out-of-range voxel color keeps the hit (no withdrawal)");
        CHECK(std::fabs(static_cast<double>(p.v.z()) - kRcHitZ) <= 1e-5,
              "K6: the saturated path resolves the same crossing");
        CHECK(p.c[0] == 0 && p.c[1] == 255 && p.c[2] == 255,
              "K6: finite out-of-range channels publish 0 / 255 / 255, not a refusal");
        std::printf("K6 saturated rgb=(%u,%u,%u) from (-0.25,1.5,2.0)\n", p.c[0], p.c[1], p.c[2]);

        const auto extreme = makeWallVolume(std::numeric_limits<float>::lowest(),
                                           std::numeric_limits<float>::max(), 0.5f);
        const RcHit e      = castCenter(*extreme);
        CHECK(isResolved(e.v), "K6: FLT_MAX / lowest() colors keep the hit");
        CHECK(e.c[0] == 0 && e.c[1] == 255 && e.c[2] == 128,
              "K6: lowest() -> 0, FLT_MAX -> 255, 0.5 -> 128 at the raycast boundary");
        std::printf("K6 extreme saturated rgb=(%u,%u,%u)\n", e.c[0], e.c[1], e.c[2]);
    }
}

// ---------------------------------------------------------------------------
// K7 Marching Cubes: saturation keeps the mesh, non-finite drops the crossing edge
//
// A UNIFORM color makes the interpolated edge color independent of the shared t
// (blend of equal endpoints is that value), so the expected byte is known without
// reasoning about which edge a vertex came from.
// ---------------------------------------------------------------------------
constexpr int   kMcRes = 32;
constexpr float kMcVs  = 0.05f;

TSDFParams mcParams() {
    TSDFParams p;
    p.resolution = kMcRes;
    p.voxel_size = kMcVs;
    p.truncation = 0.1f;
    p.max_weight = 128.0f;
    p.origin     = Eigen::Vector3f(-0.8f, -0.8f, 0.0f);
    return p;
}

// Analytic sphere: sdf = (R - |p - c|) / tau, clamped, observed everywhere. Centered
// at world (0, 0, 0.4) with R = 0.25, so it is fully inside the lattice.
std::unique_ptr<TSDFVolume> makeSphereVolume(float cr, float cg, float cb) {
    auto vol = std::make_unique<TSDFVolume>(mcParams());
    for (int z = 0; z < kMcRes; ++z) {
        for (int y = 0; y < kMcRes; ++y) {
            for (int x = 0; x < kMcRes; ++x) {
                const Eigen::Vector3f p(-0.8f + kMcVs * static_cast<float>(x),
                                        -0.8f + kMcVs * static_cast<float>(y),
                                        kMcVs * static_cast<float>(z));
                const Eigen::Vector3f c(0.0f, 0.0f, 0.4f);
                Voxel&                v = vol->voxelAt(x, y, z);
                v.tsdf   = rcClamp1((0.25f - (p - c).norm()) / 0.05f);
                v.weight = 4.0f;
                v.r      = cr;
                v.g      = cg;
                v.b      = cb;
            }
        }
    }
    return vol;
}

struct McSummary {
    size_t verts = 0;
    size_t tris  = 0;
    bool   valid = false;
    bool   has_colors = false;
    bool   all_r = true, all_g = true, all_b = true;   // every byte equals the expected one
    bool   round_trips = true;
    std::vector<uint8_t> colors;
};

McSummary extractSummary(const TSDFVolume& vol, uint8_t want_r, uint8_t want_g, uint8_t want_b) {
    MarchingCubes mc;
    const std::shared_ptr<MeshData> m = mc.extract(vol);
    McSummary s;
    s.verts      = m->positions.size();
    s.tris       = m->triangleCount();
    s.has_colors = m->hasColors();
    std::string reason;
    s.valid = m->validate(&reason);
    s.colors = m->colors;
    for (size_t i = 0; i + 2 < s.colors.size(); i += 3) {
        if (s.colors[i] != want_r) s.all_r = false;
        if (s.colors[i + 1] != want_g) s.all_g = false;
        if (s.colors[i + 2] != want_b) s.all_b = false;
        uint8_t q = 0;
        if (!srgbFloatToUint8(srgbUint8ToFloat(s.colors[i]), q) || q != s.colors[i])
            s.round_trips = false;
    }
    return s;
}

void k7_marching_cubes() {
    MarchingCubes mc;

    // Baseline: an in-range uniform color meshes a real closed surface.
    const auto base_vol = makeSphereVolume(0.5f, 0.5f, 0.5f);
    const auto base     = mc.extract(*base_vol);
    CHECK(!base->empty(), "K7: the analytic sphere meshes");
    CHECK(base->hasColors(), "K7: the mesh carries colors");
    CHECK(base->validate(&g_reason()), "K7: the baseline mesh is valid");
    const size_t base_verts = base->positions.size(), base_tris = base->triangleCount();
    CHECK(base_verts > 0 && base_tris > 0, "K7: the baseline mesh has geometry");

    // Finite out-of-range endpoints saturate. The geometry must be untouched by the
    // clamp: same field, same positions/normals/indices, only the bytes differ.
    const McSummary over = extractSummary(*makeSphereVolume(1.4f, -0.3f, 2.0f), 255, 0, 255);
    CHECK(over.verts > 0 && over.tris > 0,
          "K7: finite out-of-range endpoint colors still produce a mesh");
    CHECK(over.valid && over.has_colors, "K7: the saturated mesh is a valid colored mesh");
    CHECK(over.verts == base_verts && over.tris == base_tris,
          "K7: clamping changes no vertex and no triangle (positions/normals/indices untouched)");
    CHECK(over.all_r && over.all_g && over.all_b,
          "K7: every saturated byte is the endpoint byte (1.4 -> 255, -0.3 -> 0, 2.0 -> 255)");
    CHECK(over.round_trips, "K7: every saturated byte round-trips through the shared policy");
    std::printf("K7 saturated verts=%zu tris=%zu first=(%u,%u,%u) baseline verts=%zu tris=%zu\n",
                over.verts, over.tris, over.colors.size() ? over.colors[0] : 0,
                over.colors.size() > 1 ? over.colors[1] : 0, over.colors.size() > 2 ? over.colors[2] : 0,
                base_verts, base_tris);

    const McSummary neg = extractSummary(*makeSphereVolume(std::numeric_limits<float>::lowest(),
                                                           std::numeric_limits<float>::max(),
                                                           -1e30f), 0, 255, 0);
    CHECK(neg.verts == base_verts && neg.all_r && neg.all_g && neg.all_b,
          "K7: lowest()/FLT_MAX/-1e30 saturate to 0/255/0 on the mesh path as well");

    // Non-finite is the opposite: the crossing edge is unusable, so every triangle
    // row touching it is dropped, and a wholly non-finite field meshes nothing.
    const McSummary nan_all = extractSummary(*makeSphereVolume(*nanF(), 0.5f, 0.5f), 0, 128, 128);
    CHECK(nan_all.verts == 0 && nan_all.tris == 0,
          "K7: a NaN endpoint color drops every crossing edge (empty mesh over a meshing field)");
    const McSummary inf_all = extractSummary(*makeSphereVolume(0.5f, 0.5f, *infF()), 128, 128, 0);
    CHECK(inf_all.verts == 0 && inf_all.tris == 0,
          "K7: an +Inf endpoint color drops every crossing edge too");

    // Partial poison: a non-finite band removes only the triangles that touch it.
    {
        auto vol = makeSphereVolume(0.5f, 0.5f, 0.5f);
        int  poisoned_layers = 0;
        for (int z = 8; z < 14; ++z) {   // a band across the sphere
            for (int y = 0; y < kMcRes; ++y) {
                for (int x = 0; x < kMcRes; ++x) vol->voxelAt(x, y, z).r = *nanF();
            }
            ++poisoned_layers;
        }
        MarchingCubes mc2;
        const auto    part = mc2.extract(*vol);
        CHECK(!part->empty(), "K7: a partially poisoned field still meshes the clean bands");
        CHECK(part->triangleCount() < base_tris,
              "K7: the triangles touching a non-finite band are dropped, not emitted with a wrapped byte");
        CHECK(part->validate(&g_reason()), "K7: the partially dropped mesh stays valid");
        bool in_lockstep = (part->colors.size() == part->positions.size() * 3);
        CHECK(in_lockstep, "K7: dropped edges never break the color/vertex lockstep");
        std::printf("K7 partial poison layers=%d tris=%zu of %zu\n", poisoned_layers,
                    part->triangleCount(), base_tris);
    }
}

}  // namespace

int main() {
    k1_widening();
    k2_round_trip();
    k3_clamp_and_reject();
    k4_rounding();
    k4b_empty_sentinel();
    k4c_convergence();
    k5_boundaries();
    k6_raycast_boundary();
    k7_marching_cubes();

    if (g_failures != 0) {
        std::printf("color_convergence_contract: %d FAILED of %d checks\n", g_failures, g_checks);
        return 1;
    }
    std::printf("color_convergence_contract: all %d checks passed\n", g_checks);
    return 0;
}
