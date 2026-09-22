// tsdf_integration_race_contract (big-fix Todo 14): CPU-only contract that the
// TSDF integration merge is deterministic. Public API only (TSDFVolume, integrate,
// voxelData, MarchingCubes::extract, MeshData): no device, display, GPU, sensor,
// thread-timing, sleep or filesystem. It locks three things the OpenMP read-modify-
// write race used to break:
//   1. three repeated runs over one input produce byte-identical serialized voxels
//   2. the same three runs produce byte-identical MarchingCubes mesh bytes
//   3. thread counts 1, 2 and 4 produce bytes identical to each other and to the
//      canonical (single-thread) fold
// plus a hand-derived double oracle on a tiny fixture whose target voxel is hit by
// two march steps, so the exact weight / TSDF / color blend is asserted from first
// principles (a lost update or a dropped step changes it), never from product output.

#include "tsdf/TSDFVolume.h"
#include "meshing/MarchingCubes.h"
#include "meshing/MeshData.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "tsdf_integration_race_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using kfusion::meshing::MarchingCubes;
using kfusion::meshing::MeshData;
using kfusion::tsdf::EMPTY_COLOR;
using kfusion::tsdf::EMPTY_TSDF;
using kfusion::tsdf::TSDFParams;
using kfusion::tsdf::TSDFVolume;
using kfusion::tsdf::Voxel;

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, what)                                                          \
    do {                                                                           \
        ++g_checks;                                                                \
        if (!(cond)) {                                                             \
            std::printf("FAIL: %s  [%s:%d]\n", std::string(what).c_str(), __FILE__,\
                        __LINE__);                                                 \
            ++g_failures;                                                          \
        }                                                                          \
    } while (false)

// ---- byte-level serialization + hashing (the compared payloads) ----------------

// FNV-1a 64-bit: a deterministic, order-sensitive digest over an exact byte stream.
uint64_t fnv1a(const std::string& bytes) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : bytes) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ULL;
    }
    return h;
}

void appendF32(std::string& out, float f) {
    char tmp[sizeof(float)];
    std::memcpy(tmp, &f, sizeof(float));
    out.append(tmp, sizeof(float));
}
void appendU32(std::string& out, uint32_t v) {
    char tmp[sizeof(uint32_t)];
    std::memcpy(tmp, &v, sizeof(uint32_t));
    out.append(tmp, sizeof(uint32_t));
}

// Voxel payload: tsdf, weight, R, G, B — every field, exact IEEE bits. The color
// channels are float sRGB since Todo 19, so they widen to the same 4-byte record
// as tsdf/weight (the host Voxel is five floats).
std::string serializeVolume(const TSDFVolume& vol) {
    std::string out;
    for (const Voxel& v : vol.voxelData()) {
        appendF32(out, v.tsdf);
        appendF32(out, v.weight);
        appendF32(out, v.r);
        appendF32(out, v.g);
        appendF32(out, v.b);
    }
    return out;
}

// Mesh payload: positions, normals, colors, indices — every field, exact bits.
std::string serializeMesh(const MeshData& m) {
    std::string out;
    for (const auto& p : m.positions) { appendF32(out, p.x()); appendF32(out, p.y()); appendF32(out, p.z()); }
    for (const auto& n : m.normals)   { appendF32(out, n.x()); appendF32(out, n.y()); appendF32(out, n.z()); }
    for (uint8_t c : m.colors)         out.push_back(static_cast<char>(c));
    for (uint32_t i : m.indices)       appendU32(out, i);
    return out;
}

// ---- OpenMP thread control (restore prior state after the test) ----------------
int g_savedThreads = 0;
void pinThreads(int n) {
#ifdef _OPENMP
    g_savedThreads = omp_get_max_threads();
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}
void restoreThreads() {
#ifdef _OPENMP
    omp_set_num_threads(g_savedThreads);
#endif
}

// ---- Scenario A fixture: an overlapped wall (cross-pixel + cross-frame overlap) -
//
// A fronto-parallel wall (constant Z-depth => planar zero-crossing surface) viewed
// head-on. Intrinsics are chosen so ~12 consecutive pixels project into the same
// voxel, so many pixels (and two frames) fold into the same voxels — exactly the
// overlap the old kernel raced on. Spatially varying RGB makes any lost or reordered
// color update show up in the mesh/volume byte hash.
struct WallResult {
    uint64_t volHash;
    uint64_t meshHash;
    size_t   meshVerts;
    size_t   dirtyVoxels;
};

WallResult runWall() {
    TSDFParams p;
    p.resolution = 48;
    p.voxel_size = 0.02f;
    p.truncation = 0.05f;
    p.max_weight = 128.0f;
    p.origin     = Eigen::Vector3f(-0.48f, -0.48f, 0.05f);
    TSDFVolume vol(p);

    const int W = 40, H = 30;
    const float fx = 300.0f, fy = 300.0f, cx = 19.5f, cy = 14.5f;
    std::vector<float> depth(static_cast<size_t>(W) * H, 0.5f);   // flat wall at z=0.5
    std::vector<uint8_t> rgb(static_cast<size_t>(W) * H * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int i = (y * W + x) * 3;
            rgb[i + 0] = static_cast<uint8_t>((x * 7 + y * 3) & 0xFF);
            rgb[i + 1] = static_cast<uint8_t>((x * 5 + y * 11 + 40) & 0xFF);
            rgb[i + 2] = static_cast<uint8_t>((x * 13 + y * 2 + 90) & 0xFF);
        }
    }

    Eigen::Matrix4f pose0 = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f pose1 = Eigen::Matrix4f::Identity();
    pose1(0, 3) = p.voxel_size;                 // one-voxel lateral shift: cross-frame overlap

    vol.integrate(depth.data(), rgb.data(), pose0, fx, fy, cx, cy, W, H, 0.1f, 3.0f);
    vol.integrate(depth.data(), rgb.data(), pose1, fx, fy, cx, cy, W, H, 0.1f, 3.0f);

    WallResult r;
    r.volHash = fnv1a(serializeVolume(vol));
    size_t dirty = 0;
    for (const Voxel& v : vol.voxelData()) {
        if (v.weight > 0.0f) ++dirty;
    }
    r.dirtyVoxels = dirty;

    MarchingCubes mc;
    std::shared_ptr<MeshData> mesh = mc.extract(vol);
    r.meshHash  = fnv1a(serializeMesh(*mesh));
    r.meshVerts = mesh->positions.size();
    return r;
}

void runDeterminismAndThreadEquivalence() {
    const std::string stage = "wall determinism";

    // Canonical reference is the single-thread fold; compare every later hash to it.
    pinThreads(1);
    WallResult canon = runWall();
    restoreThreads();

    CHECK(canon.dirtyVoxels > 0, std::string(stage) + ": the integrated wall observed voxels");
    CHECK(canon.meshVerts > 0, std::string(stage) + ": MarchingCubes produced triangles to compare");

    // 1 & 2: three repeated runs, byte-identical volume AND mesh, at each thread count.
    uint64_t perThreadVol[3] = {0, 0, 0};
    uint64_t perThreadMesh[3] = {0, 0, 0};
    const int threadCounts[3] = {1, 2, 4};
    for (int ti = 0; ti < 3; ++ti) {
        pinThreads(threadCounts[ti]);
        WallResult prev = runWall();
        for (int trial = 0; trial < 3; ++trial) {
            WallResult r = runWall();
            std::string tag = stage + ": threads=" + std::to_string(threadCounts[ti]) +
                              " trial=" + std::to_string(trial);
            CHECK(r.volHash == prev.volHash, tag + " volume bytes identical to previous trial");
            CHECK(r.meshHash == prev.meshHash, tag + " mesh bytes identical to previous trial");
            CHECK(r.meshVerts == prev.meshVerts, tag + " mesh vertex count identical");
            CHECK(r.volHash == canon.volHash, tag + " volume identical to the single-thread canonical fold");
            CHECK(r.meshHash == canon.meshHash, tag + " mesh identical to the single-thread canonical fold");
            prev = r;
        }
        restoreThreads();
        perThreadVol[ti]  = prev.volHash;
        perThreadMesh[ti] = prev.meshHash;
    }

    // 3: thread-count equivalence (1 == 2 == 4), volume and mesh independently.
    CHECK(perThreadVol[0] == perThreadVol[1] && perThreadVol[1] == perThreadVol[2],
          std::string(stage) + ": volume hash identical across thread counts 1/2/4");
    CHECK(perThreadMesh[0] == perThreadMesh[1] && perThreadMesh[1] == perThreadMesh[2],
          std::string(stage) + ": mesh hash identical across thread counts 1/2/4");

    // Echo the exact payload hashes so the evidence file can capture real digests.
    std::printf("WALL canonical_volume_fnv=%016llx canonical_mesh_fnv=%016llx verts=%zu dirty=%zu\n",
                static_cast<unsigned long long>(canon.volHash),
                static_cast<unsigned long long>(canon.meshHash),
                canon.meshVerts, canon.dirtyVoxels);
    for (int ti = 0; ti < 3; ++ti) {
        std::printf("WALL threads=%d volume_fnv=%016llx mesh_fnv=%016llx\n",
                    threadCounts[ti],
                    static_cast<unsigned long long>(perThreadVol[ti]),
                    static_cast<unsigned long long>(perThreadMesh[ti]));
    }
}

// ---- Scenario B: hand-derived double oracle on a two-hit voxel -----------------
//
// A 4x4 frame with a single valid pixel on the optical axis (cx=px, cy=py so its ray
// is exactly (0,0,1)). Camera is at the world origin with identity pose, so a march
// sample at parameter t has world position (0,0,t) and sdf = D - t. origin.z = 0 and
// the numbers below place march steps 0 and 1 in voxel (32,32,47) and step 2 alone in
// voxel (32,32,48). The expected voxel is folded by hand in double with the documented
// formulas; the product's float result is compared to it. weight and color are exact;
// tsdf is compared within a float-scaled epsilon.

const Voxel& atVoxel(const TSDFVolume& v, int x, int y, int z) {
    const int res = v.params().resolution;
    return v.voxelData()[static_cast<size_t>((z * res + y) * res + x)];
}

void runHandDerivedOracle() {
    const std::string stage = "hand-oracle";

    // Product geometry constants (must match the derivation below).
    const float vs = 0.01f, trunc = 0.03f;
    const float D  = 0.5015f;
    TSDFParams p;
    p.resolution = 64;
    p.voxel_size = vs;
    p.truncation = trunc;
    p.max_weight = 128.0f;
    p.origin     = Eigen::Vector3f(-0.32f, -0.32f, 0.0f);

    const int W = 4, H = 4, px = 2, py = 2;
    const int ox = 32, oy = 32;            // floor((0 - (-0.32)) / 0.01) = 32
    const uint8_t R = 200, G = 100, B = 40;

    std::vector<float> depth(static_cast<size_t>(W) * H, 0.0f);   // invalid everywhere ...
    depth[py * W + px] = D;                                       // ... except the axis pixel
    std::vector<uint8_t> rgb(static_cast<size_t>(W) * H * 3);
    for (int i = 0; i < W * H; ++i) { rgb[i*3+0]=R; rgb[i*3+1]=G; rgb[i*3+2]=B; }

    // The march sequence is a pure function of the input, so the fold is the same at
    // any thread count; run at 4 and assert the two derived voxels.
    pinThreads(4);
    TSDFVolume vol(p);
    vol.integrate(depth.data(), rgb.data(), Eigen::Matrix4f::Identity(),
                  500.0f, 500.0f, static_cast<float>(px), static_cast<float>(py), W, H, 0.1f, 3.0f);
    restoreThreads();

    // Reference fold in double, applied in canonical (step-ascending) order.
    const double cap = 128.0, wnew = 1.0, eps = 1e-6;
    auto blendTsdf = [&](double tsdf_old, double w_old, double v) {
        return (tsdf_old * w_old + v * wnew) / (w_old + wnew + eps);
    };
    auto blendColor = [&](double old_c, double w_old, uint8_t src) -> double {
        // Canonical CPU color fold (Todo 19): the incoming byte becomes float
        // sRGB and blends through the same denominator as tsdf, with no
        // per-update rounding back to a byte.
        return (old_c * w_old + static_cast<double>(src) / 255.0) / (w_old + wnew + eps);
    };
    const double ec = static_cast<double>(EMPTY_COLOR);

    // Two-hit voxel (32,32,47): steps 0 and 1 => tsdf_new = 1.0 then 0.75.
    const double a = 1.0, b = 0.75;
    double w1 = std::min(0.0 + wnew, cap);
    double s1 = blendTsdf(EMPTY_TSDF, 0.0, a);
    double w2 = std::min(w1 + wnew, cap);
    double s2 = blendTsdf(s1, w1, b);
    const double r1 = blendColor(ec, 0.0, R), r2 = blendColor(r1, w1, R);
    const double g1 = blendColor(ec, 0.0, G), g2 = blendColor(g1, w1, G);
    const double b1 = blendColor(ec, 0.0, B), b2 = blendColor(b1, w1, B);

    const Voxel& v47 = atVoxel(vol, ox, oy, 47);
    CHECK(v47.weight == static_cast<float>(w2), stage + ": two-hit voxel weight == 2 (both steps counted, none lost)");
    CHECK(std::fabs(static_cast<double>(v47.tsdf) - s2) < 2e-6, stage + ": two-hit voxel tsdf matches the double fold");
    CHECK(std::fabs(static_cast<double>(v47.r) - r2) < 2e-6 &&
          std::fabs(static_cast<double>(v47.g) - g2) < 2e-6 &&
          std::fabs(static_cast<double>(v47.b) - b2) < 2e-6,
          stage + ": two-hit voxel color matches the double fold");
    // Anti-single-apply discriminators: a lost/undropped step would leave tsdf at one
    // of the individual candidate values or weight at 1, not the mean of both.
    CHECK(std::fabs(static_cast<double>(v47.tsdf) - a) > 1e-3 &&
          std::fabs(static_cast<double>(v47.tsdf) - b) > 1e-3,
          stage + ": two-hit tsdf is the mean of BOTH steps, not one alone");
    CHECK(v47.weight != 1.0f, stage + ": two-hit weight is not the single-update value");

    // Single-hit voxel (32,32,48): step 2 => tsdf_new = 0.5, so the fixture is proven
    // to distinguish hit counts (it is not "every voxel weight 2").
    const double c = 0.5;
    double wSingle = std::min(0.0 + wnew, cap);
    double sSingle = blendTsdf(EMPTY_TSDF, 0.0, c);
    const double rSingle = blendColor(ec, 0.0, R);
    const double gSingle = blendColor(ec, 0.0, G);
    const double bSingle = blendColor(ec, 0.0, B);
    const Voxel& v48 = atVoxel(vol, ox, oy, 48);
    CHECK(v48.weight == static_cast<float>(wSingle) && v48.weight == 1.0f,
          stage + ": single-hit voxel weight == 1 (fixture distinguishes hit counts)");
    CHECK(std::fabs(static_cast<double>(v48.tsdf) - sSingle) < 2e-6,
          stage + ": single-hit voxel tsdf matches the single double fold");
    CHECK(std::fabs(static_cast<double>(v48.r) - rSingle) < 2e-6 &&
          std::fabs(static_cast<double>(v48.g) - gSingle) < 2e-6 &&
          std::fabs(static_cast<double>(v48.b) - bSingle) < 2e-6,
          stage + ": single-hit voxel color matches the single double fold");
    // A single update from the neutral EMPTY_COLOR must land exactly on the
    // normalized input byte, not on the byte value itself (120 would be an
    // out-of-range sRGB component) and not on a re-rounded byte mean.
    CHECK(std::fabs(static_cast<double>(v48.r) - static_cast<double>(R) / 255.0) < 1e-6,
          stage + ": one update equals the normalized input float R/255");

    // Thread-count equivalence for this same code path is locked by the wall scenario.

    std::printf("ORACLE two_hit tsdf=%.9g weight=%.9g rgb=(%.9g,%.9g,%.9g) single_hit tsdf=%.9g weight=%.9g rgb=(%.9g,%.9g,%.9g)\n",
                v47.tsdf, v47.weight, v47.r, v47.g, v47.b, v48.tsdf, v48.weight, v48.r, v48.g, v48.b);
}

} // namespace

int main() {
    runDeterminismAndThreadEquivalence();
    runHandDerivedOracle();

    if (g_failures == 0) {
        std::printf("tsdf_integration_race_contract: PASS (%d checks: 3-run byte-identical "
                    "volume+mesh, threads 1/2/4 equivalence, hand-derived two-hit blend)\n",
                    g_checks);
        return 0;
    }
    std::printf("tsdf_integration_race_contract: FAIL (%d failed checks of %d)\n", g_failures, g_checks);
    return 1;
}
