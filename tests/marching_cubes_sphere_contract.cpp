// marching_cubes_sphere_contract (big-fix todo 6): a CPU-only analytic sphere
// extraction contract that drives the real kfusion::meshing::MarchingCubes::extract
// over a deterministic signed-distance volume and asserts the geometry the CPU
// meshing stage is contracted to produce.
//
// Why the fixture and its gates stay UB-free: before big-fix todo 7 repaired the
// shared table, cube configurations 213/214/215 read uninitialized
// edge_verts/edge_norms/edge_colors slots (their corrupt edge masks were missing
// bits the tri rows still referenced), so extracting such a cube was undefined
// behavior. Two fail-closed gates keep this test out of UB regardless of table
// state: Gate 0 refuses to call extract() while any of those three rows deviates
// from the independently derived canonical masks (0x83f/0xb35/0xa3c), and Gate 1
// enumerates every potentially-meshed cube configuration FIRST, from the same
// corner-sign rule MarchingCubes::extract uses, proving none is 213/214/215. If
// either gate fires, the test fails through a well-defined assertion and returns
// WITHOUT calling extract() — never UB. Todo 7 repaired the three rows, so Gate 0
// now passes and the extraction assertions below run.
//
// The assertions are derived from signed-volume geometry, not from current code
// comments or current broken behavior:
//   - non-empty mesh;
//   - finite vertex positions and normals;
//   - no degenerate (zero-area) triangles;
//   - a closed, manifold surface via directed-edge pairing over the welded soup
//     (duplicate triangle records collapsed first, then each directed edge must be
//     matched by exactly one reversed edge: no boundary edge, no non-manifold edge);
//   - outward orientation via the geometric face normal (from the emitted winding)
//     against the radial direction out of the sphere center — the CPU authoritative
//     outward-normal fixture per docs/CANONICAL_SEMANTICS.md;
//   - a triangle-count band consistent with a sphere of this radius and voxel size.

#include "meshing/MarchingCubes.h"
#include "meshing/MarchingCubesTables.h"
#include "tsdf/TSDFVolume.h"
#include "utils/ColorMath.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <array>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

namespace {

using kfusion::utils::srgbUint8ToFloat;   // uint8 sRGB -> the volume's float sRGB domain

int g_failures = 0;

#define CHECK(cond, what)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL: %s  [%s:%d]\n", (what), __FILE__, __LINE__);    \
            ++g_failures;                                                      \
        }                                                                      \
    } while (false)

// Corner offsets and the corner->config rule, copied structurally from
// src/meshing/MarchingCubes.cpp so the pre-extraction enumeration sees exactly the
// cubes the meshing stage would mesh.
const int kCornerOffsets[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
};

// Endpoint corner index of each cube edge, the same pairs EDGE_CORNERS in
// src/meshing/MarchingCubes.cpp consumes. Used only by the independent
// crossing-edge oracle below.
constexpr int kEdgeCorners[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7},
};

// Rebuild an edge mask from corner sign bits: edge e crosses iff its two endpoint
// corners lie on opposite sides. Pure integer math, no table read, no parity.
constexpr int crossingEdgeMask(int config) {
    int mask = 0;
    for (int e = 0; e < 12; ++e) {
        const int a = (config >> kEdgeCorners[e][0]) & 1;
        const int b = (config >> kEdgeCorners[e][1]) & 1;
        if (a != b) mask |= (1 << e);
    }
    return mask;
}

// Safety precondition, stated as an explicit fail-before-fix assertion: the three
// cube configs whose tri rows reference edges the edge mask must actually mark. If
// the shared CPU table still carries the corrupt 213/214/215 entries, extracting
// such a cube reads uninitialized edge_verts/edge_norms/edge_colors slots (UB), so
// the sphere test must refuse to call extract() until the table is repaired. Returns
// true only when all three rows match the independently-derived canonical masks.
bool corruptTableEntriesAreRepaired() {
    struct Expect { int config; int canonical; };
    const Expect cases[3] = {{213, 0x83f}, {214, 0xb35}, {215, 0xa3c}};
    bool clean = true;
    for (const auto& c : cases) {
        // Independently re-derive the expected mask, then assert it equals both the
        // documented canonical value and the live table entry.
        const int oracle = crossingEdgeMask(c.config);
        const int actual = kfusion::meshing::tables::edge_table[c.config];
        if (oracle != c.canonical) {
            std::printf("FAIL: oracle self-check config %d derived 0x%x, expected canonical 0x%x\n",
                        c.config, oracle, c.canonical);
            clean = false;
        }
        if (actual != c.canonical) {
            std::printf("FAIL: edge_table[%d] = 0x%x, crossing-edge oracle expects 0x%x\n",
                        c.config, actual, c.canonical);
            clean = false;
        }
    }
    return clean;
}


// Fixture geometry: 48^3 voxels, 2 cm each, sphere radius 30 cm centered on the
// volume center. The sphere spans [17 cm, 77 cm] inside a [0, 94 cm] domain, so it
// is fully interior (no border truncation) and the surface is closed. This exact
// center-aligned configuration is the one whose enumerated config set provably
// excludes 213/214/215.
constexpr int    kResolution = 48;
constexpr float  kVoxelSize  = 0.02f;
constexpr float  kRadius     = 0.30f;

struct Sphere {
    Eigen::Vector3f center;
    float radius;
};

float sphereSdf(const kfusion::tsdf::TSDFVolume& vol, int x, int y, int z, const Sphere& s) {
    return (vol.voxelToWorld(x, y, z) - s.center).norm() - s.radius;
}

// Fill the whole volume with the normalized sphere SDF and full weight, so every
// cube is meshable and the only gate is the corner sign pattern.
void fillSphereVolume(kfusion::tsdf::TSDFVolume& vol, const Sphere& s) {
    const auto& p = vol.params();
    for (int z = 0; z < p.resolution; ++z) {
        for (int y = 0; y < p.resolution; ++y) {
            for (int x = 0; x < p.resolution; ++x) {
                float t = sphereSdf(vol, x, y, z, s) / p.truncation;
                if (t < -1.0f) t = -1.0f;
                if (t > 1.0f) t = 1.0f;
                auto& v = vol.voxelAt(x, y, z);
                v.tsdf   = t;
                v.weight = p.max_weight;
                v.r = srgbUint8ToFloat(120); v.g = srgbUint8ToFloat(160); v.b = srgbUint8ToFloat(200);
            }
        }
    }
}

// The cube configuration MarchingCubes::extract would compute for cube (x,y,z), or
// -1 when the cube is not meshed (an unweighted corner, or an empty edge mask).
int cubeConfig(const kfusion::tsdf::TSDFVolume& vol, int x, int y, int z) {
    float corner[8];
    for (int c = 0; c < 8; ++c) {
        const auto& v = vol.voxelAt(x + kCornerOffsets[c][0],
                                     y + kCornerOffsets[c][1],
                                     z + kCornerOffsets[c][2]);
        if (v.weight <= 0.001f) return -1;
        corner[c] = v.tsdf;
    }
    int config = 0;
    for (int c = 0; c < 8; ++c)
        if (corner[c] < 0.0f) config |= (1 << c);
    if (kfusion::meshing::tables::edge_table[config] == 0) return -1;
    return config;
}

struct PreExtraction {
    bool found_corrupt = false;
    int corrupt_config = -1;
    int meshed_cubes = 0;
};

// Enumerate every potentially-meshed cube exactly as extract() will, and prove none
// is a corrupt config. This is the independent gate that keeps the test out of UB.
PreExtraction enumerateAndGuard(const kfusion::tsdf::TSDFVolume& vol) {
    PreExtraction out;
    const int R = vol.params().resolution;
    for (int z = 0; z + 1 < R; ++z) {
        for (int y = 0; y + 1 < R; ++y) {
            for (int x = 0; x + 1 < R; ++x) {
                const int config = cubeConfig(vol, x, y, z);
                if (config < 0) continue;
                ++out.meshed_cubes;
                if (config == 213 || config == 214 || config == 215) {
                    out.found_corrupt  = true;
                    out.corrupt_config = config;
                    return out;
                }
            }
        }
    }
    return out;
}

using kfusion::meshing::MeshData;

bool allFinite(const MeshData& m) {
    for (const auto& pos : m.positions)
        if (!std::isfinite(pos.x()) || !std::isfinite(pos.y()) || !std::isfinite(pos.z()))
            return false;
    for (const auto& n : m.normals)
        if (!std::isfinite(n.x()) || !std::isfinite(n.y()) || !std::isfinite(n.z()))
            return false;
    return true;
}

void testNoDegenerateTriangles(const MeshData& m) {
    size_t degenerate = 0;
    for (size_t t = 0; t < m.triangleCount(); ++t) {
        const Eigen::Vector3f& a = m.positions[m.indices[3 * t + 0]];
        const Eigen::Vector3f& b = m.positions[m.indices[3 * t + 1]];
        const Eigen::Vector3f& c = m.positions[m.indices[3 * t + 2]];
        if ((b - a).cross(c - a).norm() <= 1e-9f) ++degenerate;
    }
    CHECK(degenerate == 0, "sphere mesh has no degenerate (zero-area) triangles");
}

// Closed + manifold: collapse exact duplicate triangle records (identical welded
// vertex-index triple in identical winding order), then require every directed edge
// - taken in the triangle's emitted winding order - to be matched by exactly one
// reversed directed edge. A closed orientable manifold has zero unmatched (boundary)
// directed edges and no edge used by more than two faces. The winding order must be
// preserved while pairing: sorting the triple would erase orientation and make the
// check meaningless.
void testClosedManifoldSurface(const MeshData& m) {
    std::set<std::array<uint32_t, 3>> unique_tris; // keyed by winding order, not sorted
    for (size_t t = 0; t < m.triangleCount(); ++t) {
        unique_tris.insert(std::array<uint32_t, 3>{
            m.indices[3 * t + 0], m.indices[3 * t + 1], m.indices[3 * t + 2]});
    }

    std::map<std::pair<uint32_t, uint32_t>, int> directed;
    for (const auto& tri : unique_tris) {
        for (int i = 0; i < 3; ++i) ++directed[{tri[i], tri[(i + 1) % 3]}];
    }

    int boundary = 0;      // directed edges with no reverse partner
    int manifold_violation = 0; // a directed edge, or its reverse, used more than once
    for (const auto& kv : directed) {
        const auto rev = directed.find({kv.first.second, kv.first.first});
        const int reverse_count = (rev == directed.end()) ? 0 : rev->second;
        if (reverse_count == 0) ++boundary;
        if (kv.second > 1 || reverse_count > 1) ++manifold_violation;
    }

    CHECK(boundary == 0, "sphere mesh is closed: every directed edge has a reverse partner");
    CHECK(manifold_violation == 0, "sphere mesh is manifold: no edge shared by more than two faces");
}

// Outward orientation from the emitted winding, judged against signed-volume geometry
// (never a hardcoded copy of current behavior): the geometric face normal must point
// away from the sphere center for every non-degenerate triangle.
void testOutwardOrientation(const MeshData& m, const Sphere& s) {
    int outward = 0;
    int inward = 0;
    for (size_t t = 0; t < m.triangleCount(); ++t) {
        const Eigen::Vector3f& a = m.positions[m.indices[3 * t + 0]];
        const Eigen::Vector3f& b = m.positions[m.indices[3 * t + 1]];
        const Eigen::Vector3f& c = m.positions[m.indices[3 * t + 2]];
        const Eigen::Vector3f face_normal = (b - a).cross(c - a);
        if (face_normal.norm() <= 1e-9f) continue; // degenerate, already asserted absent
        const Eigen::Vector3f radial = (a + b + c) / 3.0f - s.center;
        const double dot = static_cast<double>(face_normal.dot(radial));
        if (dot > 0.0) ++outward;
        else if (dot < 0.0) ++inward;
    }
    CHECK(outward > 0, "sphere mesh has outward-oriented faces");
    CHECK(inward == 0, "sphere mesh has no inward-oriented faces (winding is outward)");
}

} // namespace

int main() {
    // Gate 0 (fail-closed safety precondition, written fail-before-fix): refuse to
    // exercise extraction while the shared table mis-marks the 213/214/215 masks,
    // because extracting those cube configs reads uninitialized edge slots. This is
    // pure integer math over the table and returns before any volume work or
    // extract(), so the test can never reach undefined behavior through the table.
    // It was RED at todo 6; todo 7 repaired the rows to 0x83f/0xb35/0xa3c, so it
    // now passes and the extraction assertions below run.
    if (!corruptTableEntriesAreRepaired()) {
        std::printf("marching_cubes_sphere_contract: FAIL "
                    "(edge_table 213/214/215 not repaired; not calling extract())\n");
        return 1;
    }

    kfusion::tsdf::TSDFParams p;
    p.resolution = kResolution;
    p.voxel_size = kVoxelSize;
    p.truncation = 2.5f * kVoxelSize;
    p.max_weight = 64.0f;
    p.origin = Eigen::Vector3f(0.0f, 0.0f, 0.0f);

    kfusion::tsdf::TSDFVolume vol(p);
    const float half = 0.5f * static_cast<float>(kResolution - 1) * kVoxelSize;
    const Sphere sphere{Eigen::Vector3f(half, half, half), kRadius};
    fillSphereVolume(vol, sphere);

    // Gate 1: prove no potentially-meshed cube is a corrupt config BEFORE extract().
    const PreExtraction guard = enumerateAndGuard(vol);
    if (guard.found_corrupt) {
        std::printf("FAIL: pre-extraction enumeration hit corrupt cube config %d; "
                    "refusing to call extract() (would read uninitialized edge data)\n",
                    guard.corrupt_config);
        std::printf("marching_cubes_sphere_contract: FAIL (fixture touches a corrupt config)\n");
        return 1;
    }
    CHECK(guard.meshed_cubes > 0, "fixture produces meshed cubes");

    // Gate 2: real extraction, now provably UB-free for this fixture.
    auto mesh = kfusion::meshing::MarchingCubes().extract(vol);
    CHECK(mesh != nullptr, "extract returns a mesh");
    if (!mesh || mesh->empty()) {
        std::printf("FAIL: extract produced an empty mesh\n");
        ++g_failures;
    } else {
        const size_t tris = mesh->triangleCount();
        std::printf("sphere fixture: %d meshed cubes, %zu triangles\n", guard.meshed_cubes, tris);
        CHECK(tris > 0, "mesh is non-empty");
        CHECK(mesh->positions.size() * 3 == mesh->colors.size(), "colors are one RGB triple per vertex");
        CHECK(allFinite(*mesh), "all vertex positions and normals are finite");
        testNoDegenerateTriangles(*mesh);
        testClosedManifoldSurface(*mesh);
        testOutwardOrientation(*mesh, sphere);
        // Band: a radius-30 cm sphere at a 2 cm voxel has ~4*pi*(15 vox)^2 ~ 28k
        // surface voxels; the 48^3 grid tables 8588 triangles. Assert a wide band
        // that a broken mesh (empty, exploded, or single-octant) could not satisfy.
        CHECK(tris > 4000 && tris < 20000, "triangle count is in the expected sphere band");
    }

    if (g_failures == 0) {
        std::printf("marching_cubes_sphere_contract: PASS (analytic sphere: closed, manifold, outward, finite, non-degenerate)\n");
        return 0;
    }
    std::printf("marching_cubes_sphere_contract: FAIL (%d failed check(s))\n", g_failures);
    return 1;
}
