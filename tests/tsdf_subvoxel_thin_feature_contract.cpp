// tsdf_subvoxel_thin_feature_contract (big-fix Todo 15): CPU-only contract that the CPU
// raycast resolves sub-voxel-thin surfaces at the reconstructed zero crossing and honors
// the configured depth band at both ends. Public API only (TSDFVolume ctor/setParams/
// raycast/voxelAt): no device, display, GPU, sensor, thread timing, sleep or filesystem.
// Expected values are hand-derived from the fixture's own lattice definition through the
// mathematical definition of linear reconstruction, never from product output.
//
// Fixture: 32^3 volume, 40 mm voxels, origin (-0.64,-0.64,0), identity pose, a 1x1 image
// with cx=cy=0 and fx=fy=1 so the single ray is exactly (0,0,1). The lattice is written
// directly (weight>0 everywhere) as tsdf = clamp(signed_dist(z)/tau, -1, 1) with tau =
// 20 mm, uniform in x/y, so the trilinear value along the ray is the piecewise-linear
// interpolation through the lattice z-samples and the ray stays in voxel column x=y=16.
// Reconstruction oracle: the first lattice plane k with field(k) < 0 puts the surface at
//   z* = (k-1)*vs + vs * field(k-1) / (field(k-1) - field(k))
// If NO lattice plane goes negative the feature is below the lattice's own representable
// limit and the honest answer is "no surface" (asserted, not hidden).
#include "tsdf/TSDFVolume.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "tsdf_subvoxel_thin_feature_contract must be compiled with AZU_PIPELINE_TEST_SEAM"
#endif

namespace {

using kfusion::tsdf::EMPTY_COLOR;
using kfusion::tsdf::EMPTY_TSDF;
using kfusion::tsdf::EMPTY_WEIGHT;
using kfusion::tsdf::TSDFParams;
using kfusion::tsdf::TSDFVolume;

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

constexpr int   kRes   = 32;
constexpr float kVs    = 0.04f;
constexpr float kTau   = 0.02f;
constexpr float kNear  = 0.12f;   // configured near bound used by most gates
constexpr float kFar   = 0.90f;   // configured far bound used by most gates
constexpr float kEps   = 1e-5f;

const float kNaN = std::numeric_limits<float>::quiet_NaN();

using Slab = std::pair<float, float>;

TSDFParams fixtureParams() {
    TSDFParams p;
    p.resolution = kRes;
    p.voxel_size = kVs;
    p.truncation = 0.025f;
    p.max_weight = 128.0f;
    p.origin     = Eigen::Vector3f(-0.64f, -0.64f, 0.0f);
    p.min_depth  = kNear;
    p.max_depth  = kFar;
    return p;
}

// Signed distance to the union of slabs along z (negative inside a slab).
float signedDist(float z, const std::vector<Slab>& slabs) {
    float best_inside = 0.0f, best_outside = std::numeric_limits<float>::infinity();
    bool  inside = false;
    for (const Slab& s : slabs) {
        if (z >= s.first && z <= s.second) {
            inside   = true;
            best_inside = std::max(best_inside, std::min(z - s.first, s.second - z));
        } else {
            best_outside = std::min(best_outside, z < s.first ? s.first - z : z - s.second);
        }
    }
    return inside ? -best_inside : best_outside;
}

// The fixture's own lattice: index k is the plane z = k*vs.
std::vector<float> buildLattice(const std::vector<Slab>& slabs) {
    std::vector<float> f(static_cast<size_t>(kRes));
    for (int k = 0; k < kRes; ++k) {
        const float z  = static_cast<float>(k) * kVs;
        const float v  = signedDist(z, slabs) / kTau;
        f[static_cast<size_t>(k)] = std::max(-1.0f, std::min(1.0f, v));
    }
    return f;
}

// Hand-derived reconstruction oracle: first zero of the piecewise-linear interpolation
// through the lattice samples, or NaN when the lattice never goes negative.
float oracleZero(const std::vector<float>& f) {
    for (int k = 1; k < kRes; ++k) {
        const float prev = f[static_cast<size_t>(k - 1)];
        const float cur  = f[static_cast<size_t>(k)];
        if (cur < 0.0f) {
            if (prev <= 0.0f) return static_cast<float>(k - 1) * kVs;  // tangent/degenerate
            return static_cast<float>(k - 1) * kVs +
                   kVs * prev / (prev - cur);
        }
    }
    return kNaN;
}

uint8_t layerR(int z) { return static_cast<uint8_t>(31 + 7 * (z % 29)); }
uint8_t layerG(int z) { return static_cast<uint8_t>(200 - z); }
uint8_t layerB(int z) { return static_cast<uint8_t>(17 + 3 * (z % 11)); }

struct Hit {
    Eigen::Vector3f vertex{Eigen::Vector3f::Zero()};
    Eigen::Vector3f normal{Eigen::Vector3f::Zero()};
    uint8_t         rgb[3] = {0, 0, 0};
    uint8_t r() const { return rgb[0]; }
    uint8_t g() const { return rgb[1]; }
    uint8_t b() const { return rgb[2]; }
    bool miss() const {
        return vertex.isZero(0.0f) && normal.isZero(0.0f) && rgb[0] == 0 && rgb[1] == 0 &&
               rgb[2] == 0;
    }
};

// One raycast of the single on-axis pixel against a lattice built from `slabs`.
Hit castOnce(const std::vector<Slab>& slabs, float min_depth, float max_depth) {
    TSDFParams p   = fixtureParams();
    p.min_depth    = min_depth;
    p.max_depth    = max_depth;
    TSDFVolume vol(p);
    const std::vector<float> f = buildLattice(slabs);
    for (int z = 0; z < kRes; ++z) {
        for (int y = 0; y < kRes; ++y) {
            for (int x = 0; x < kRes; ++x) {
                auto& v            = vol.voxelAt(x, y, z);
                v.tsdf             = f[static_cast<size_t>(z)];
                v.weight           = 1.0f;
                v.r                = layerR(z);
                v.g                = layerG(z);
                v.b                = layerB(z);
            }
        }
    }

    Hit out;
    vol.raycast(Eigen::Matrix4f::Identity(), 1.0f, 1.0f, 0.0f, 0.0f, 1, 1, &out.vertex,
                &out.normal, out.rgb);
    return out;
}

void expectHit(const std::string& tag, const Hit& h, float want_z, int want_layer,
               const std::string& detail) {
    const std::string where = tag + ": " + detail;
    CHECK(!h.miss(), where + " -> expected a hit at z=" + std::to_string(want_z));
    if (h.miss()) {
        std::printf("%s MISS vertex=(%.6f,%.6f,%.6f)\n", tag.c_str(), h.vertex.x(), h.vertex.y(),
                    h.vertex.z());
        return;
    }
    CHECK(std::fabs(h.vertex.z() - want_z) < kEps,
          where + " z == " + std::to_string(want_z) + ", got " + std::to_string(h.vertex.z()));
    CHECK(std::fabs(h.vertex.x()) < kEps && std::fabs(h.vertex.y()) < kEps,
          where + " stays on the axis, got (" + std::to_string(h.vertex.x()) + "," +
              std::to_string(h.vertex.y()) + ")");
    CHECK(h.r() == layerR(want_layer) && h.g() == layerG(want_layer) && h.b() == layerB(want_layer),
          where + " color is layer " + std::to_string(want_layer) + ", got (" +
              std::to_string(h.r()) + "," + std::to_string(h.g()) + "," + std::to_string(h.b()) +
              ")");
    std::printf("%s hit z=%.7f want=%.7f color=(%d,%d,%d) want=(%d,%d,%d)\n", tag.c_str(),
                h.vertex.z(), want_z, h.r(), h.g(), h.b(), layerR(want_layer), layerG(want_layer),
                layerB(want_layer));
}

void expectMiss(const std::string& tag, const Hit& h, const std::string& detail) {
    CHECK(h.miss(), tag + ": " + detail + " -> expected no surface, got z=" +
                       std::to_string(h.vertex.z()) + " normal=(" + std::to_string(h.normal.x()) +
                       "," + std::to_string(h.normal.y()) + "," + std::to_string(h.normal.z()) + ")");
    std::printf("%s miss=%d z=%.7f\n", tag.c_str(), static_cast<int>(h.miss()), h.vertex.z());
}

// ---------------------------------------------------------------------------
// A. a sub-voxel plate CLOSER than the historical 0.3 m start is still found
// ---------------------------------------------------------------------------
void gateA_nearBound() {
    const std::string     tag     = "A/near-bound";
    const std::vector<Slab> slabs = {{0.19f, 0.21f}};
    const float             want  = oracleZero(buildLattice(slabs));
    // Hand-derived: field(4)=+1 (dist 0.03/tau), field(5)=-0.5 (inside, 0.01/tau)
    //   -> z* = 0.16 + 0.04 * 1/1.5 = 0.1866667
    CHECK(std::fabs(want - 0.18666667f) < 1e-6f,
          tag + ": fixture oracle is 0.1866667, got " + std::to_string(want));

    const Hit h = castOnce(slabs, kNear, kFar);
    expectHit(tag, h, want, 4, "a 20 mm plate whose front face is at 0.19 m is detected with min_depth=0.12");
    CHECK(std::fabs(h.normal.z() + 1.0f) < 1e-4f && std::fabs(h.normal.x()) < 1e-4f &&
              std::fabs(h.normal.y()) < 1e-4f,
          tag + ": front face normal points back at the camera, got (" +
              std::to_string(h.normal.x()) + "," + std::to_string(h.normal.y()) + "," +
              std::to_string(h.normal.z()) + ")");
    // A near bound INSIDE the plate cannot hide it: the first crossing then is the back
    // face, at field(5)=-0.5 -> field(6)=+1, i.e. 0.20 + 0.04*0.5/1.5 = 0.2133333 with the
    // normal flipped. Only a bound past the back face (0.22) makes the plate disappear.
    const Hit inside = castOnce(slabs, 0.20f, kFar);
    expectHit(tag, inside, 0.21333333f, 5, "min_depth=0.20 starts inside the plate");
    CHECK(std::fabs(inside.normal.z() - 1.0f) < 1e-4f && std::fabs(inside.normal.x()) < 1e-4f &&
              std::fabs(inside.normal.y()) < 1e-4f,
          tag + ": back face normal points away from the camera, got (" +
              std::to_string(inside.normal.x()) + "," + std::to_string(inside.normal.y()) + "," +
              std::to_string(inside.normal.z()) + ")");
    expectMiss(tag, castOnce(slabs, 0.22f, kFar),
               "min_depth=0.22 past the back face hides the plate entirely");
}

// ---------------------------------------------------------------------------
// B. mid-range sub-voxel plates resolve at the crossing, not at a voxel plane
// ---------------------------------------------------------------------------
void gateB_subvoxelPrecision() {
    const std::string tag = "B/subvoxel-precision";
    struct Case {
        float    a, b;
        const char* name;
    };
    const Case cases[] = {
        {0.19f, 0.21f, "half-voxel plate"},
        {0.39f, 0.41f, "half-voxel plate at mid range"},
        {0.395f, 0.405f, "quarter-voxel plate"},
    };
    for (const Case& c : cases) {
        const std::vector<Slab> slabs = {{c.a, c.b}};
        const std::vector<float> lat  = buildLattice(slabs);
        const float              want = oracleZero(lat);
        CHECK(!std::isnan(want), tag + ": " + std::string(c.name) + " is lattice-representable");
        if (std::isnan(want)) continue;
        const Hit h = castOnce(slabs, kNear, kFar);
        expectHit(tag, h, want, static_cast<int>(std::floor(want / kVs)), c.name);
        if (h.miss()) continue;
        // Not a voxel-plane snap: the resolved z sits strictly inside its voxel.
        const float frac = h.vertex.z() / kVs - std::floor(h.vertex.z() / kVs);
        CHECK(frac > 1e-3f && frac < 1.0f - 1e-3f,
              tag + ": " + std::string(c.name) + " resolved off the lattice planes, frac=" +
                  std::to_string(frac));
    }
}

// ---------------------------------------------------------------------------
// C. two plates one voxel apart: the FIRST crossing wins
// ---------------------------------------------------------------------------
void gateC_firstHitWins() {
    const std::string       tag     = "C/first-hit";
    const std::vector<Slab> slabs   = {{0.39f, 0.41f}, {0.47f, 0.49f}};
    const std::vector<float> lat    = buildLattice(slabs);
    const float             want    = oracleZero(lat);        // the near plate
    const float second = oracleZero(buildLattice({{0.47f, 0.49f}}));  // the far plate alone
    CHECK(std::fabs(second - 0.46666667f) < 1e-6f,
          tag + ": the far plate alone would resolve at 0.4666667, got " + std::to_string(second));
    CHECK(std::fabs(want - 0.38666667f) < 1e-6f,
          tag + ": oracle picks the near plate 0.3866667, got " + std::to_string(want));
    const Hit h = castOnce(slabs, kNear, kFar);
    expectHit(tag, h, want, 9, "the near plate of a 1-voxel-gap doublet is the reported surface");
    if (!h.miss()) {
        CHECK(std::fabs(h.vertex.z() - second) > 1e-3f,
              tag + ": the far plate was not reported instead, z=" + std::to_string(h.vertex.z()));
    }
}

// ---------------------------------------------------------------------------
// D. the configured far bound truncates the march
// ---------------------------------------------------------------------------
void gateD_farBound() {
    const std::string       tag     = "D/far-bound";
    const std::vector<Slab> slabs   = {{0.91f, 0.93f}};
    const std::vector<float> lat    = buildLattice(slabs);
    const float             want    = oracleZero(lat);        // 0.88 + 0.04*1/1.5 = 0.9066667
    CHECK(std::fabs(want - 0.90666667f) < 1e-6f,
          tag + ": fixture oracle is 0.9066667, got " + std::to_string(want));
    expectMiss(tag, castOnce(slabs, kNear, 0.90f), "max_depth=0.90 never reaches the 0.9067 m plate");
    expectHit(tag, castOnce(slabs, kNear, 0.95f), want, 22,
              "max_depth=0.95 covers the same plate");
}

// ---------------------------------------------------------------------------
// E. march-phase sweep: every representable phase resolves, every
//    unrepresentable phase reports nothing (no phantom surface)
// ---------------------------------------------------------------------------
void gateE_phaseSweep() {
    const std::string tag = "E/phase-sweep";
    const float       deltas[] = {0.0f, 0.005f, 0.0075f, 0.015f, 0.02f, 0.025f};
    int               hits     = 0;
    for (const float d : deltas) {
        const std::vector<Slab> slabs{{0.19f + d, 0.21f + d}};
        const std::vector<float> lat = buildLattice(slabs);
        const float              want = oracleZero(lat);
        const Hit                h    = castOnce(slabs, kNear, kFar);
        const std::string        name = std::string("plate front 0.19+") + std::to_string(d);
        if (std::isnan(want)) {
            expectMiss(tag, h, name + " has no negative lattice plane -> unrepresentable");
            continue;
        }
        ++hits;
        expectHit(tag, h, want, static_cast<int>(std::floor(want / kVs)),
                  name + " resolves at the reconstructed zero " + std::to_string(want));
    }
    CHECK(hits == 3, tag + ": sweep covers both representable and unrepresentable phases, hits=" +
                        std::to_string(hits));
}

} // namespace

int main() {
    gateA_nearBound();
    gateB_subvoxelPrecision();
    gateC_firstHitWins();
    gateD_farBound();
    gateE_phaseSweep();

    if (g_failures == 0) {
        std::printf("tsdf_subvoxel_thin_feature_contract: PASS (%d checks: sub-voxel crossings, "
                    "configured near/far bounds, first-hit precedence, phase sweep)\n",
                    g_checks);
        return 0;
    }
    std::printf("tsdf_subvoxel_thin_feature_contract: FAIL (%d failed checks of %d)\n", g_failures,
                g_checks);
    return 1;
}
