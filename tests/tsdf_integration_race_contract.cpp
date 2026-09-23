// tsdf_integration_race_contract (big-fix Todo 14): CPU-only contract that the
// TSDF integration merge is deterministic. Public API only (TSDFVolume, integrate,
// voxelData, MarchingCubes::extract, MeshData): no device, display, GPU, sensor,
// thread-timing, sleep or filesystem. It locks three things the OpenMP read-modify-
// write race used to break:
//   1. three repeated runs over one input produce byte-identical serialized voxels
//   2. the same three runs produce byte-identical MarchingCubes mesh bytes
//   3. thread counts 1, 2 and 4 produce bytes identical to each other and to the
//      canonical (single-thread) fold
// plus a hand-derived double oracle on a tiny single-ray fixture: every voxel on the
// ray takes exactly one update per frame, so the exact weight / TSDF / colour fold
// over two frames is asserted from first principles, never from product output.

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

// ---- Scenario B: hand-derived double oracle (voxel-projective) -----------------
//
// A 4x4 frame with one valid pixel on the optical axis (cx=px, cy=py). Camera at
// the world origin, identity pose. Voxel corners on the axis, world (0,0,k*vs),
// project exactly onto that pixel; every off-axis voxel within the volume
// projects outside the 4x4 image (|x|*fx/z >= 0.5 px). So voxel k on the axis
// gets exactly ONE update per frame with sdf = D - k*vs, and nothing else is
// touched. Two frames with different depth and colour are folded by hand in
// double with the documented formulas:
//   tsdf   <- (tsdf*w + min(1, sdf/trunc)) / (w + 1)   when sdf >= -trunc
//   weight <- min(w + 1, max_weight)
//   colour <- (c*w + byte/255) / (w + 1)                when |sdf| < trunc/2

const Voxel& atVoxel(const TSDFVolume& v, int x, int y, int z) {
    const int res = v.params().resolution;
    return v.voxelData()[static_cast<size_t>((z * res + y) * res + x)];
}

void runHandDerivedOracle() {
    const std::string stage = "hand-oracle";

    const float vs = 0.01f, trunc = 0.03f;
    TSDFParams p;
    p.resolution = 64;
    p.voxel_size = vs;
    p.truncation = trunc;
    p.max_weight = 128.0f;
    p.origin     = Eigen::Vector3f(-0.32f, -0.32f, 0.0f);

    const int W = 4, H = 4, px = 2, py = 2;
    const int ox = 32, oy = 32;             // (0 - (-0.32)) / 0.01 = 32: the axis
    struct Frame { float D; uint8_t R, G, B; };
    const Frame frames[2] = {{0.5015f, 200, 100, 40}, {0.5065f, 60, 180, 250}};

    pinThreads(4);
    TSDFVolume vol(p);
    for (const Frame& f : frames) {
        std::vector<float> depth(static_cast<size_t>(W) * H, 0.0f);
        depth[py * W + px] = f.D;
        std::vector<uint8_t> rgb(static_cast<size_t>(W) * H * 3);
        for (int i = 0; i < W * H; ++i) { rgb[i*3+0]=f.R; rgb[i*3+1]=f.G; rgb[i*3+2]=f.B; }
        vol.integrate(depth.data(), rgb.data(), Eigen::Matrix4f::Identity(),
                      500.0f, 500.0f, static_cast<float>(px), static_cast<float>(py),
                      W, H, 0.1f, 3.0f);
    }
    restoreThreads();

    // Reference fold for axis voxel k.
    struct Ref { double tsdf, w, r, g, b; };
    auto fold = [&](int k) {
        Ref v{EMPTY_TSDF, 0.0, EMPTY_COLOR, EMPTY_COLOR, EMPTY_COLOR};
        for (const Frame& f : frames) {
            const double sdf = static_cast<double>(f.D) - static_cast<double>(static_cast<float>(k) * vs);
            if (sdf < -trunc) continue;
            const double tn = std::min(1.0, sdf / trunc);
            const double w  = v.w;
            v.tsdf = (v.tsdf * w + tn) / (w + 1.0);
            if (std::fabs(sdf) < 0.5 * trunc) {
                v.r = (v.r * w + f.R / 255.0) / (w + 1.0);
                v.g = (v.g * w + f.G / 255.0) / (w + 1.0);
                v.b = (v.b * w + f.B / 255.0) / (w + 1.0);
            }
            v.w = std::min(w + 1.0, 128.0);
        }
        return v;
    };

    int surface_checked = 0;
    for (int k = 1; k < 64; ++k) {
        const Ref want = fold(k);
        const Voxel& got = atVoxel(vol, ox, oy, k);
        const std::string tag = stage + " k=" + std::to_string(k);
        CHECK(got.weight == static_cast<float>(want.w), tag + ": weight is one per frame");
        CHECK(std::fabs(got.tsdf - want.tsdf) < 2e-5, tag + ": tsdf matches the double fold");
        CHECK(std::fabs(got.r - want.r) < 2e-6 && std::fabs(got.g - want.g) < 2e-6 &&
              std::fabs(got.b - want.b) < 2e-6, tag + ": colour matches the double fold");
        if (want.w == 2.0 && std::fabs(want.tsdf) < 0.5) ++surface_checked;
    }
    CHECK(surface_checked >= 2, stage + ": the fixture exercises blended surface voxels");

    // Voxel 50 sits 1.5 mm / 6.5 mm in front of the two surfaces: both frames
    // fold in, colour included, so it is the mean of two distinct updates.
    const Voxel& v50 = atVoxel(vol, ox, oy, 50);
    CHECK(v50.weight == 2.0f, stage + ": surface voxel took exactly two updates");
    CHECK(std::fabs(v50.r - (200.0 / 255.0 + 60.0 / 255.0) / 2.0) < 2e-6,
          stage + ": surface voxel colour is the mean of both frames");
    // Voxel 54 is beyond D + trunc for both frames: never touched.
    CHECK(atVoxel(vol, ox, oy, 54).weight == 0.0f, stage + ": voxel behind the band untouched");
    CHECK(atVoxel(vol, ox + 1, oy, 50).weight == 0.0f, stage + ": off-axis voxel untouched");

    std::printf("ORACLE v50 tsdf=%.9g weight=%.9g rgb=(%.9g,%.9g,%.9g)\n",
                v50.tsdf, v50.weight, v50.r, v50.g, v50.b);
}

} // namespace

int main() {
    runDeterminismAndThreadEquivalence();
    runHandDerivedOracle();

    if (g_failures == 0) {
        std::printf("tsdf_integration_race_contract: PASS (%d checks: 3-run byte-identical "
                    "volume+mesh, threads 1/2/4 equivalence, hand-derived two-frame fold)\n",
                    g_checks);
        return 0;
    }
    std::printf("tsdf_integration_race_contract: FAIL (%d failed checks of %d)\n", g_failures, g_checks);
    return 1;
}
