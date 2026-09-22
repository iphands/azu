// marching_cubes_frontier_contract (big-fix Todo 16): CPU-only contract for the
// observed/unobserved frontier of kfusion::meshing::MarchingCubes::extract, for
// the volume-boundary normal, and for the non-finite guards on emitted
// vertices/normals. Public CPU API only (TSDFVolume + MarchingCubes::extract);
// no device, display, GPU, sensor, thread timing, filesystem or network.
//
// The canonical rules under test (docs/CANONICAL_SEMANTICS.md, "Marching Cubes
// and winding"):
//   R1 an unobserved voxel (weight <= 1e-3) or a non-finite tsdf samples as
//      +1.0f, whatever tsdf literal the voxel happens to carry;
//   R2 a cube is evaluated from its crossing edges, so an unobserved corner that
//      is not an endpoint of a crossing edge cannot suppress the whole cube;
//   R3 a crossing edge is emitted only when BOTH its endpoint voxels are
//      observed and finite (the edge's support), and a triangle is emitted only
//      when all three of its edges are;
//   R4 a normal whose gradient or interpolation is not usable yields no vertex,
//      instead of a fabricated stand-in;
//   R5 at the volume border the gradient uses a one-sided difference; it never
//      folds in an invented out-of-bounds sample;
//   R6 repeated extraction of one volume is byte-identical.
//
// Every expectation is an independent oracle (integer edge algebra over the
// shared table, hand-derived positions, topology over the emitted index stream),
// never a snapshot of current product output. Sections that are genuinely
// defective at HEAD are marked RED; sections that lock behavior which is already
// correct are marked LOCK, so the evidence file cannot overclaim.

#include "meshing/MarchingCubes.h"
#include "meshing/MarchingCubesTables.h"
#include "tsdf/TSDFVolume.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

using kfusion::meshing::MeshData;
using kfusion::tsdf::TSDFParams;
using kfusion::tsdf::TSDFVolume;

int g_failures = 0;
int g_sections = 0;

#define CHECK(cond, what)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL: %s  [%s:%d]\n", (what), __FILE__, __LINE__);    \
            ++g_failures;                                                      \
        }                                                                      \
    } while (false)

// Same corner/edge incidence the meshing stage uses (src/meshing/MarchingCubes.cpp).
const int kCornerOffsets[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
};
const int kEdgeCorners[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7},
};

constexpr float kWeightEps = 1e-3f;   // canonical observation threshold
constexpr float kNaN       = std::numeric_limits<float>::quiet_NaN();

bool isFinite(const Eigen::Vector3f& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

// ---------------------------------------------------------------------------
// Independent oracle over one volume: which crossing edges exist, where their
// vertex belongs, and whether the edge has support. Mirrors the canonical rules
// R1-R3 with plain integer/float algebra, not by calling the product.
// ---------------------------------------------------------------------------
struct Sample {
    float value;
    bool  supported;  // observed AND finite
};

Sample sampleVoxel(const TSDFVolume& vol, int x, int y, int z) {
    const auto& p = vol.params();
    if (x < 0 || y < 0 || z < 0 || x >= p.resolution || y >= p.resolution || z >= p.resolution)
        return {kfusion::tsdf::EMPTY_TSDF, false};
    const auto& v = vol.voxelAt(x, y, z);
    if (v.weight <= kWeightEps || !std::isfinite(v.tsdf)) return {kfusion::tsdf::EMPTY_TSDF, false};
    return {v.tsdf, true};
}

// Canonical zero-crossing placement along p1 -> p2 (same closed form the CPU
// meshing stage is contracted to use).
Eigen::Vector3f crossingPoint(const Eigen::Vector3f& p1, float v1,
                              const Eigen::Vector3f& p2, float v2) {
    if (std::abs(v1) < 1e-6f) return p1;
    if (std::abs(v2) < 1e-6f) return p2;
    const float diff = v1 - v2;
    if (std::abs(diff) < 1e-6f) return p1;
    float t = v1 / diff;
    t = std::max(0.0f, std::min(1.0f, t));
    return p1 + t * (p2 - p1);
}

struct EdgeSet {
    // geometric edge identity -> (expected vertex, has support)
    struct Key {
        long long v0, v1;
        int axis;
        bool operator<(const Key& o) const {
            if (v0 != o.v0) return v0 < o.v0;
            if (v1 != o.v1) return v1 < o.v1;
            return axis < o.axis;
        }
    };
    std::map<Key, std::pair<Eigen::Vector3f, bool>> edges;

    void consider(const TSDFVolume& vol, int ax, int ay, int az, int bx, int by, int bz, int axis) {
        const Sample a = sampleVoxel(vol, ax, ay, az);
        const Sample b = sampleVoxel(vol, bx, by, bz);
        if ((a.value < 0.0f) == (b.value < 0.0f)) return;  // no crossing
        const Eigen::Vector3f pa = vol.voxelToWorld(ax, ay, az);
        const Eigen::Vector3f pb = vol.voxelToWorld(bx, by, bz);
        // Each geometric edge is visited exactly once, from its lower endpoint
        // toward +axis, so the key is unique and the interpolation direction is fixed.
        const Key k{static_cast<long long>(ax) * 100000 + static_cast<long long>(ay) * 1000 + az,
                    static_cast<long long>(bx) * 100000 + static_cast<long long>(by) * 1000 + bz,
                    axis};
        edges.emplace(k, std::make_pair(crossingPoint(pa, a.value, pb, b.value),
                                        a.supported && b.supported));
    }

    void build(const TSDFVolume& vol) {
        const int R = vol.params().resolution;
        for (int z = 0; z < R; ++z)
            for (int y = 0; y < R; ++y)
                for (int x = 0; x < R; ++x) {
                    if (x + 1 < R) consider(vol, x, y, z, x + 1, y, z, 0);
                    if (y + 1 < R) consider(vol, x, y, z, x, y + 1, z, 1);
                    if (z + 1 < R) consider(vol, x, y, z, x, y, z + 1, 2);
                }
    }
};

bool meshHasVertexNear(const MeshData& m, const Eigen::Vector3f& q, float tol) {
    for (const auto& p : m.positions)
        if ((p - q).norm() <= tol) return true;
    return false;
}

// Independent triangle-coverage oracle. Replays the canonical per-cube rule
// directly over the volume - weight-guarded corner signs, the shared edge_table
// mask, and a triangle row only when all three of its edges have two supported
// endpoints - and counts the rows. edge_table itself is independently pinned by
// marching_cubes_table_contract, so the count is not a product snapshot.
size_t oracleTriangleCount(const TSDFVolume& vol) {
    const int R = vol.params().resolution;
    size_t total = 0;
    for (int z = 0; z + 1 < R; ++z)
        for (int y = 0; y + 1 < R; ++y)
            for (int x = 0; x + 1 < R; ++x) {
                float val[8]; bool ok[8];
                for (int c = 0; c < 8; ++c) {
                    Sample s = sampleVoxel(vol, x + kCornerOffsets[c][0],
                                           y + kCornerOffsets[c][1], z + kCornerOffsets[c][2]);
                    val[c] = s.value; ok[c] = s.supported;
                }
                int cfg = 0;
                for (int c = 0; c < 8; ++c) if (val[c] < 0.0f) cfg |= (1 << c);
                const int mask = kfusion::meshing::tables::edge_table[cfg];
                if (mask == 0) continue;
                bool edok[12];
                for (int e = 0; e < 12; ++e) {
                    edok[e] = false;
                    if (!(mask & (1 << e))) continue;
                    edok[e] = ok[kEdgeCorners[e][0]] && ok[kEdgeCorners[e][1]];
                }
                for (int t = 0; kfusion::meshing::tables::tri_table[cfg][t] != -1; t += 3) {
                    const int a = kfusion::meshing::tables::tri_table[cfg][t];
                    const int b = kfusion::meshing::tables::tri_table[cfg][t + 1];
                    const int c2 = kfusion::meshing::tables::tri_table[cfg][t + 2];
                    if (edok[a] && edok[b] && edok[c2]) ++total;
                }
            }
    return total;
}

struct MeshStats {
    size_t nonFiniteVerts  = 0;
    size_t nonFiniteNorms  = 0;
    size_t nonUnitNorms    = 0;
    bool   colorsLockstep  = true;
};

MeshStats auditMesh(const MeshData& m) {
    MeshStats st;
    for (const auto& p : m.positions) if (!isFinite(p)) ++st.nonFiniteVerts;
    for (const auto& n : m.normals) {
        if (!isFinite(n)) { ++st.nonFiniteNorms; continue; }
        if (std::abs(n.norm() - 1.0f) > 1e-3f) ++st.nonUnitNorms;
    }
    st.colorsLockstep = (m.colors.size() == 0) || (m.colors.size() == m.positions.size() * 3);
    return st;
}

// Directed-edge pairing over the welded triangle soup: a closed surface has no
// directed edge without its reverse partner. Duplicate triangle records collapse
// first (the product can emit the same triangle from two cubes only if welding
// kept them identical).
int boundaryEdgeCount(const MeshData& m) {
    std::set<std::array<uint32_t, 3>> unique;
    for (size_t t = 0; t < m.triangleCount(); ++t)
        unique.insert(std::array<uint32_t, 3>{m.indices[3 * t], m.indices[3 * t + 1], m.indices[3 * t + 2]});
    std::map<std::pair<uint32_t, uint32_t>, int> directed;
    for (const auto& tri : unique)
        for (int i = 0; i < 3; ++i) ++directed[{tri[i], tri[(i + 1) % 3]}];
    int boundary = 0;
    for (const auto& kv : directed) {
        const auto rev = directed.find({kv.first.second, kv.first.first});
        if (rev == directed.end()) ++boundary;
    }
    return boundary;
}

std::string serializeBytes(const MeshData& m) {
    std::string out;
    auto push = [&out](const void* src, size_t n) {
        out.append(reinterpret_cast<const char*>(src), n);
    };
    for (const auto& p : m.positions) push(p.data(), sizeof(float) * 3);
    for (const auto& n : m.normals)   push(n.data(), sizeof(float) * 3);
    for (auto c : m.colors)           out.push_back(static_cast<char>(c));
    for (auto i : m.indices)          push(&i, sizeof(uint32_t));
    return out;
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------
TSDFParams makeParams(int resolution, float voxel_size, float truncation) {
    TSDFParams p;
    p.resolution = resolution;
    p.voxel_size = voxel_size;
    p.truncation = truncation;
    p.max_weight = 8.0f;
    p.origin     = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    return p;
}

// Everything observed and outside, so a fixture's negative voxels are the only
// thing that can create geometry. `ramp_x` adds a small positive x slope so every
// outside voxel has a usable gradient; without it an isolated negative voxel has
// six identical neighbours and no normal at all, which is section 6's fixture, not
// this one's.
void fillOutside(TSDFVolume& vol, float value = 0.5f, float ramp_x = 0.0f) {
    const auto& p = vol.params();
    for (int z = 0; z < p.resolution; ++z)
        for (int y = 0; y < p.resolution; ++y)
            for (int x = 0; x < p.resolution; ++x) {
                auto& v  = vol.voxelAt(x, y, z);
                v.tsdf   = value + ramp_x * static_cast<float>(x);
                v.weight = p.max_weight;
                v.r = 10; v.g = 20; v.b = 30;
            }
}

void stamp(TSDFVolume& vol, int x, int y, int z, float tsdf, float weight) {
    auto& v  = vol.voxelAt(x, y, z);
    v.tsdf   = tsdf;
    v.weight = weight;
}

std::shared_ptr<MeshData> mesh(const TSDFVolume& vol) {
    return kfusion::meshing::MarchingCubes().extract(vol);
}

// ---------------------------------------------------------------------------
// Section 1 (RED): one unobserved corner carrying a STALE NEGATIVE tsdf, not an
// endpoint of any crossing edge, must neither suppress the cube nor read as
// inside material. Fixture: a single negative voxel (2,2,2) in an observed
// outside field, plus (3,3,3) unobserved with tsdf = -0.8. Exactly eight cubes
// touch (2,2,2); the single-corner configuration emits one triangle each
// (tri_table[1] = {0,8,3}), and the six axis neighbours of (2,2,2) are welded
// into six vertices: a closed 8-face patch. HEAD drops the one cube holding the
// unobserved corner, so the patch loses a face and stops being closed.
// ---------------------------------------------------------------------------
void section_stale_unobserved_corner() {
    ++g_sections;
    TSDFVolume vol(makeParams(8, 0.05f, 0.15f));
    fillOutside(vol, 0.5f, 0.02f);
    stamp(vol, 2, 2, 2, -0.6f, 8.0f);   // the only real interior voxel
    stamp(vol, 3, 3, 3, -0.8f, 0.0f);   // unobserved + stale negative literal

    auto m = mesh(vol);
    CHECK(m != nullptr && !m->empty(), "s1: frontier patch is meshed at all");
    if (!m || m->empty()) return;

    // The six expected vertices: the crossings of the six axis edges out of (2,2,2).
    std::vector<Eigen::Vector3f> expected;
    const Eigen::Vector3f c = vol.voxelToWorld(2, 2, 2);
    const float neg = vol.voxelAt(2, 2, 2).tsdf;
    const int dirs[6][3] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    for (auto& d : dirs) {
        const Eigen::Vector3f n = vol.voxelToWorld(2 + d[0], 2 + d[1], 2 + d[2]);
        const float nv = vol.voxelAt(2 + d[0], 2 + d[1], 2 + d[2]).tsdf;
        expected.push_back(crossingPoint(c, neg, n, nv));
    }
    const float tol = 0.05f * 1e-3f;
    for (const auto& q : expected)
        CHECK(meshHasVertexNear(*m, q, tol), "s1: every supported frontier vertex is present");

    // A stale-negative unobserved corner read as inside would mint vertices toward
    // (4,3,3)/(3,4,3)/(3,3,4). None of those may exist.
    const float neg2 = -0.8f;
    const int diagonals[3][3] = {{2,2,3},{2,3,2},{3,2,2}};  // far ends of (3,3,3)+axis
    for (auto& d : diagonals) {
        const Eigen::Vector3f q = crossingPoint(vol.voxelToWorld(3, 3, 3), neg2,
                                                vol.voxelToWorld(d[0], d[1], d[2]),
                                                vol.voxelAt(d[0], d[1], d[2]).tsdf);
        CHECK(!meshHasVertexNear(*m, q, tol), "s1: stale-negative unobserved corner emits no vertex");
    }

    // Oracle: eight cubes x one triangle (tri_table[1]) - not seven.
    size_t trisPerCube = 0;
    for (int i = 0; kfusion::meshing::tables::tri_table[1][i] != -1; i += 3) ++trisPerCube;
    const size_t want = 8 * trisPerCube;
    std::printf("s1: triangles=%zu oracle=%zu (8 cubes x %zu) boundary=%d verts=%zu\n",
                m->triangleCount(), want, trisPerCube, boundaryEdgeCount(*m), m->positions.size());
    CHECK(m->triangleCount() == want, "s1: all eight cubes around the frontier vertex contribute");
    CHECK(boundaryEdgeCount(*m) == 0, "s1: the frontier patch is closed (no lost cube)");
    const auto st = auditMesh(*m);
    CHECK(st.nonFiniteVerts == 0 && st.nonFiniteNorms == 0, "s1: finite vertices and normals");
    CHECK(st.nonUnitNorms == 0, "s1: every normal is unit length");
    std::string why;
    CHECK(m->validate(&why), "s1: emitted patch passes MeshData::validate()");
}

// ---------------------------------------------------------------------------
// Section 2 (RED): a whole analytic sphere whose observed region stops just
// outside the surface. Observation is "everything inside plus 1.2 voxels out",
// so every true crossing edge has both endpoints inside 1 voxel of the surface
// (an axis edge of length vs changes a 1-Lipschitz SDF by at most vs) and is
// therefore supported, while cubes whose far corners exceed 1.2 voxels are
// allowed to have unobserved corners. HEAD drops those whole cubes, which opens
// the surface; the canonical rules keep it closed.
// ---------------------------------------------------------------------------
void section_frontier_sphere_closure() {
    ++g_sections;
    const int    R   = 32;
    const float  vs  = 0.03f;
    const float  rad = 0.30f;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    const auto& p = vol.params();
    const Eigen::Vector3f center(0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs);

    int observed = 0, frontierCubes = 0;
    for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                const float sdf = (vol.voxelToWorld(x, y, z) - center).norm() - rad;
                auto& v = vol.voxelAt(x, y, z);
                v.r = 200; v.g = 100; v.b = 50;
                if (sdf <= 1.2f * vs) {                 // interior + thin outer shell
                    float t = sdf / p.truncation;
                    v.tsdf   = std::max(-1.0f, std::min(1.0f, t));
                    v.weight = p.max_weight;
                    ++observed;
                } else {                                // never seen: canonical empty
                    v.tsdf   = kfusion::tsdf::EMPTY_TSDF;
                    v.weight = kfusion::tsdf::EMPTY_WEIGHT;
                }
            }
    CHECK(observed > 0, "s2: fixture has an observed region");

    // Fixture sanity, independent of the product: at least one cube holds both a
    // crossing edge and an unobserved corner. Without such cubes the closure
    // assertion below would be vacuous.
    for (int z = 0; z + 1 < R; ++z)
        for (int y = 0; y + 1 < R; ++y)
            for (int x = 0; x + 1 < R; ++x) {
                bool anyUnobserved = false, anyCrossing = false;
                float val[8]; bool ok[8];
                for (int c = 0; c < 8; ++c) {
                    Sample s = sampleVoxel(vol, x + kCornerOffsets[c][0],
                                          y + kCornerOffsets[c][1], z + kCornerOffsets[c][2]);
                    val[c] = s.value; ok[c] = s.supported;
                    if (!ok[c]) anyUnobserved = true;
                }
                int cfg = 0;
                for (int c = 0; c < 8; ++c) if (val[c] < 0.0f) cfg |= (1 << c);
                const int mask = kfusion::meshing::tables::edge_table[cfg];
                for (int e = 0; e < 12 && !anyCrossing; ++e) {
                    if (!(mask & (1 << e))) continue;
                    const int a = kEdgeCorners[e][0], b = kEdgeCorners[e][1];
                    if (ok[a] && ok[b]) anyCrossing = true;
                }
                if (anyUnobserved && anyCrossing) ++frontierCubes;
            }
    std::printf("s2: observed voxels=%d frontier cubes=%d\n", observed, frontierCubes);
    CHECK(frontierCubes > 0, "s2: fixture really has crossing cubes with unobserved corners");

    auto m = mesh(vol);
    CHECK(m != nullptr && !m->empty(), "s2: frontier sphere produces a mesh");
    if (!m || m->empty()) return;
    const auto st = auditMesh(*m);
    const size_t wantTris = oracleTriangleCount(vol);
    std::printf("s2: triangles=%zu oracle=%zu verts=%zu boundary=%d nonfinite=%zu nonunit=%zu\n",
                m->triangleCount(), wantTris, m->positions.size(), boundaryEdgeCount(*m),
                st.nonFiniteVerts + st.nonFiniteNorms, st.nonUnitNorms);
    CHECK(boundaryEdgeCount(*m) == 0, "s2: the observed/unobserved frontier is closed (no cracks)");
    CHECK(m->triangleCount() == wantTris, "s2: every supported frontier triangle is emitted");
    CHECK(st.nonFiniteVerts == 0 && st.nonFiniteNorms == 0, "s2: no non-finite vertex or normal");
    CHECK(st.nonUnitNorms == 0, "s2: all frontier normals are unit length");
    CHECK(st.colorsLockstep, "s2: colors stay one RGB triple per vertex");
    CHECK(m->triangleCount() > 1000 && m->triangleCount() < 20000,
          "s2: triangle count stays in the band for a radius-10-voxel sphere");

    // Coverage: every supported crossing edge's vertex is present, and no
    // unsupported-edge vertex appears.
    EdgeSet es; es.build(vol);
    size_t missing = 0, spurious = 0;
    const float tol = vs * 1e-3f;
    for (const auto& kv : es.edges) {
        const bool present = meshHasVertexNear(*m, kv.second.first, tol);
        if (kv.second.second && !present) ++missing;
        if (!kv.second.second && present) ++spurious;
    }
    std::printf("s2: oracle edges=%zu missing_supported=%zu unsupported_present=%zu\n",
                es.edges.size(), missing, spurious);
    CHECK(missing == 0, "s2: every supported crossing edge on the frontier is meshed");
    CHECK(spurious == 0, "s2: no vertex comes from an unsupported crossing edge");
}

// ---------------------------------------------------------------------------
// Section 3 (RED): a crossing edge whose endpoint is unobserved is refused, and
// the refusal must not take the surrounding triangles with it. A planar front
// normal to x, with one positive-side voxel switched to unobserved-with-stale-
// negative. HEAD deletes every cube touching that voxel (four whole crossing
// cubes); the canonical rule keeps their supported edges.
// ---------------------------------------------------------------------------
void section_unsupported_edge() {
    ++g_sections;
    const int   R  = 12;
    const float vs = 0.05f;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    const auto& p = vol.params();
    const float planeX = 2.5f * vs;                     // crossing between voxels 2 and 3
    for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                const float s = (vol.voxelToWorld(x, y, z).x() - planeX) / p.truncation;
                auto& v = vol.voxelAt(x, y, z);
                v.tsdf   = std::max(-1.0f, std::min(1.0f, s));
                v.weight = p.max_weight;
                v.r = 70; v.g = 70; v.b = 70;
            }
    stamp(vol, 3, 3, 3, -0.9f, 0.0f);                   // unobserved + stale negative

    auto m = mesh(vol);
    CHECK(m != nullptr && !m->empty(), "s3: planar front is meshed");
    if (!m || m->empty()) return;

    EdgeSet es; es.build(vol);
    const float tol = vs * 1e-3f;
    size_t missing = 0, spurious = 0, supported = 0, unsupported = 0;
    for (const auto& kv : es.edges) {
        const bool present = meshHasVertexNear(*m, kv.second.first, tol);
        if (kv.second.second) { ++supported; if (!present) ++missing; }
        else                  { ++unsupported; if (present) ++spurious; }
    }
    std::printf("s3: triangles=%zu oracle=%zu supported=%zu unsupported=%zu missing=%zu spurious=%zu\n",
                m->triangleCount(), oracleTriangleCount(vol), supported, unsupported, missing, spurious);
    CHECK(unsupported >= 1, "s3: fixture really has an unsupported crossing edge");
    CHECK(supported > 20, "s3: fixture really has many supported crossing edges");
    CHECK(missing == 0, "s3: supported neighbours of an unsupported edge still mesh");
    CHECK(spurious == 0, "s3: the unsupported crossing edge contributes no vertex");
    CHECK(m->triangleCount() == oracleTriangleCount(vol),
          "s3: refusing one edge costs only the triangles that need it");
    const auto st = auditMesh(*m);
    CHECK(st.nonFiniteVerts == 0 && st.nonFiniteNorms == 0, "s3: finite vertices and normals");
    CHECK(st.nonUnitNorms == 0, "s3: unit normals");
    std::string why;
    CHECK(m->validate(&why), "s3: emitted mesh passes MeshData::validate()");
}

// ---------------------------------------------------------------------------
// Section 4 (RED): normal interpolation is guarded. A one-dimensional zigzag
// along x, -0.9, -0.5, +0.5, -0.9, repeated, makes a crossing edge whose two
// endpoint gradients point exactly opposite with t = 0.5, so the blended normal
// is the zero vector. HEAD calls Eigen .normalized() on it and pushes
// (NaN,NaN,NaN) into MeshData. The canonical rule refuses the edge and keeps the
// triangles built from the other crossing layer.
// ---------------------------------------------------------------------------
void section_cancelled_normal() {
    ++g_sections;
    const int   R  = 16;
    const float vs = 0.05f;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    const float pattern[4] = {-0.9f, -0.5f, 0.5f, -0.9f};
    const auto& p = vol.params();
    for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                auto& v  = vol.voxelAt(x, y, z);
                v.tsdf   = pattern[x % 4];
                v.weight = p.max_weight;
                v.r = 30; v.g = 60; v.b = 90;
            }

    auto m = mesh(vol);
    CHECK(m != nullptr && !m->empty(), "s4: the surviving crossing layer still meshes");
    if (!m || m->empty()) return;

    const auto st = auditMesh(*m);
    int firstBad = -1;
    for (size_t i = 0; i < m->normals.size(); ++i) {
        const Eigen::Vector3f& n = m->normals[i];
        if (!isFinite(n) || std::abs(n.norm() - 1.0f) > 1e-3f) { firstBad = static_cast<int>(i); break; }
    }
    std::printf("s4: triangles=%zu nonfinite_pos=%zu nonfinite_norm=%zu nonunit=%zu",
                m->triangleCount(), st.nonFiniteVerts, st.nonFiniteNorms, st.nonUnitNorms);
    if (firstBad >= 0)
        std::printf(" first_invalid_normal=(%.9g,%.9g,%.9g)|%.9g",
                    double(m->normals[firstBad].x()), double(m->normals[firstBad].y()),
                    double(m->normals[firstBad].z()), double(m->normals[firstBad].norm()));
    std::printf("\n");
    CHECK(st.nonFiniteNorms == 0, "s4: a cancelled normal pair never reaches MeshData");
    CHECK(st.nonUnitNorms == 0, "s4: a cancelled normal pair never yields a non-unit normal");
    CHECK(st.nonFiniteVerts == 0, "s4: no non-finite vertex position");

    // The cancelled layer is x = 4k+1 -> 4k+2 (t = 0.5); the surviving layer is
    // x = 4k+2 -> 4k+3 at t = 0.5/(0.5+0.9).
    const float tol = vs * 1e-3f;
    float cancelledX[4] = {0, 0, 0, 0};
    for (int k = 0; k < 4; ++k) cancelledX[k] = (4 * k + 1.5f) * vs;
    int leaked = 0;
    for (const auto& pos : m->positions)
        for (float cx : cancelledX)
            if (std::abs(pos.x() - cx) <= tol) ++leaked;
    std::printf("s4: vertices on a cancelled plane=%d\n", leaked);
    CHECK(leaked == 0, "s4: no vertex survives on the cancelled crossing plane");

    const float okT = 0.5f / (0.5f + 0.9f);
    bool survivor = false;
    for (int k = 0; k < 4 && !survivor; ++k) {
        const float xw = (4 * k + 2.0f + okT) * vs;
        for (const auto& pos : m->positions)
            if (std::abs(pos.x() - xw) <= tol) { survivor = true; break; }
    }
    CHECK(survivor, "s4: the un-cancelled crossing layer is untouched (no collateral loss)");
    std::string why;
    CHECK(m->validate(&why), "s4: emitted mesh passes MeshData::validate()");
}

// ---------------------------------------------------------------------------
// Section 5: NaN and +/-Inf tsdf literals in *observed* voxels. The weight guard
// must treat them as unobserved (+1.0f), so nothing non-finite can propagate and
// the invented crossings they create are refused as unsupported instead of
// becoming extra geometry. RED witness here is the +/-Inf case: HEAD divides
// Inf by the Inf it just computed for the gradient length and pushes a NaN
// normal, and the invented crossings roughly double the triangle count.
// ---------------------------------------------------------------------------
void section_non_finite_tsdf() {
    ++g_sections;
    const int    R   = 24;
    const float  vs  = 0.04f;
    const float  rad = 0.30f;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    const auto& p = vol.params();
    const Eigen::Vector3f center(0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs);

    auto clean = std::make_shared<TSDFVolume>(makeParams(R, vs, 3.0f * vs));
    for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                const float s = (vol.voxelToWorld(x, y, z) - center).norm() - rad;
                const float t = std::max(-1.0f, std::min(1.0f, s / p.truncation));
                for (auto* target : {&vol, clean.get()}) {
                    auto& v  = target->voxelAt(x, y, z);
                    v.tsdf   = t;
                    v.weight = p.max_weight;
                    v.r = 11; v.g = 22; v.b = 33;
                }
            }

    size_t poisoned = 0;
    const float inf = std::numeric_limits<float>::infinity();
    for (int z = 8; z < 16; ++z)
        for (int y = 8; y < 16; ++y)
            for (int x = 8; x < 16; ++x) {
                if (((x + y + z) % 3) == 0) { stamp(vol, x, y, z, kNaN, p.max_weight); ++poisoned; }
                else if (((x + y + z) % 3) == 1) { stamp(vol, x, y, z, inf, p.max_weight); ++poisoned; }
                else { stamp(vol, x, y, z, -inf, p.max_weight); ++poisoned; }
            }
    CHECK(poisoned > 0, "s5: fixture poisons voxels");

    auto mClean = mesh(*clean);
    auto mPoison = mesh(vol);
    CHECK(mClean != nullptr && !mClean->empty(), "s5: clean sphere meshes");
    CHECK(mPoison != nullptr, "s5: poisoned sphere returns a mesh");
    if (!mPoison) return;
    const auto st = auditMesh(*mPoison);
    std::printf("s5: poisoned=%zu clean_tris=%zu poison_tris=%zu nonfinite=%zu\n",
                poisoned, mClean ? mClean->triangleCount() : 0, mPoison->triangleCount(),
                st.nonFiniteVerts + st.nonFiniteNorms);
    CHECK(st.nonFiniteVerts == 0, "s5: a non-finite tsdf never yields a non-finite vertex");
    CHECK(st.nonFiniteNorms == 0, "s5: a non-finite tsdf never yields a non-finite normal");
    CHECK(st.nonUnitNorms == 0, "s5: unit normals after poisoning");
    CHECK(mPoison->triangleCount() <= mClean->triangleCount(),
          "s5: non-finite voxels add no invented geometry over the clean sphere");
    const Eigen::Vector3f origin(0.f, 0.f, 0.f);
    bool atOrigin = false;
    for (const auto& q : mPoison->positions) if ((q - origin).norm() < 1e-6f) atOrigin = true;
    CHECK(!atOrigin, "s5: no non-finite collapse is hidden as the world origin vertex");
    if (!mPoison->empty()) {
        std::string why;
        CHECK(mPoison->validate(&why), "s5: poisoned mesh passes MeshData::validate()");
    }
}

// ---------------------------------------------------------------------------
// Section 6 (RED): a corner whose gradient is not usable yields no vertex. One
// isolated negative voxel inside a uniform +0.5 field has six equal neighbours,
// i.e. a zero gradient, so a normal does not exist there. HEAD substitutes the
// fabricated (0,0,1) and emits eight triangles; the canonical rule refuses them.
// The control underneath proves the refusal is about the gradient, not about the
// feature being one voxel wide.
// ---------------------------------------------------------------------------
void section_degenerate_corner() {
    ++g_sections;
    TSDFVolume vol(makeParams(10, 0.05f, 0.15f));
    fillOutside(vol, 0.5f);
    stamp(vol, 5, 5, 5, -0.6f, 8.0f);

    auto m = mesh(vol);
    CHECK(m != nullptr, "s6: degenerate fixture returns a mesh");
    if (!m) return;
    std::printf("s6: degenerate triangles=%zu verts=%zu\n", m->triangleCount(), m->positions.size());
    CHECK(m->positions.empty() && m->indices.empty(),
          "s6: a corner with no usable gradient yields no vertex (no fabricated +Z)");

    // Control: break the gradient degeneracy by moving one neighbour only. The
    // feature is the same single voxel, so it must now mesh.
    TSDFVolume ctrl(makeParams(10, 0.05f, 0.15f));
    fillOutside(ctrl, 0.5f);
    stamp(ctrl, 5, 5, 5, -0.6f, 8.0f);
    stamp(ctrl, 6, 5, 5, 0.9f, 8.0f);
    auto mc = mesh(ctrl);
    CHECK(mc != nullptr && !mc->empty(), "s6: control - a usable gradient meshes the same voxel");
    if (!mc || mc->empty()) return;
    const auto st = auditMesh(*mc);
    CHECK(st.nonFiniteNorms == 0 && st.nonUnitNorms == 0, "s6: control normals are finite and unit");
}

// ---------------------------------------------------------------------------
// Section 7 (RED): the volume-boundary normal. A plane front normal to x inside
// an otherwise fully observed volume has the analytic normal (+1,0,0) every-
// where. HEAD's central difference steps out of bounds and folds the invented
// 1.0 out-of-bounds sample into the gradient, so every vertex sitting on the
// y=0 / y=RES-1 / z=0 / z=RES-1 border ring gets a visibly tilted normal. The
// canonical rule takes a one-sided difference there.
// ---------------------------------------------------------------------------
void section_border_one_sided_normal() {
    ++g_sections;
    const int   R  = 20;
    const float vs = 0.05f;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    const auto& p = vol.params();
    const float planeX = 9.5f * vs;
    for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                const float s = (vol.voxelToWorld(x, y, z).x() - planeX) / p.truncation;
                auto& v = vol.voxelAt(x, y, z);
                v.tsdf   = std::max(-1.0f, std::min(1.0f, s));
                v.weight = p.max_weight;
                v.r = 90; v.g = 90; v.b = 90;
            }

    auto m = mesh(vol);
    CHECK(m != nullptr && !m->empty(), "s7: planar sheet meshes");
    if (!m || m->empty()) return;

    const float hi = (R - 1) * vs;
    int borderVerts = 0, tilted = 0;
    for (size_t i = 0; i < m->positions.size(); ++i) {
        const Eigen::Vector3f& q = m->positions[i];
        const Eigen::Vector3f& n = m->normals[i];
        if (!isFinite(n) || (n - Eigen::Vector3f(1.0f, 0.0f, 0.0f)).norm() > 1e-3f) ++tilted;
        if (std::abs(q.y()) < 1e-6f || std::abs(q.y() - hi) < 1e-6f ||
            std::abs(q.z()) < 1e-6f || std::abs(q.z() - hi) < 1e-6f)
            ++borderVerts;
    }
    std::printf("s7: verts=%zu border-ring verts=%d normals off (+1,0,0) =%d\n",
                m->positions.size(), borderVerts, tilted);
    CHECK(borderVerts > 0, "s7: the sheet really reaches the volume border (non-vacuous)");
    CHECK(tilted == 0, "s7: border normals are one-sided, never an invented out-of-bounds sample");
}

// ---------------------------------------------------------------------------
// Section 8 (LOCK): repeated extraction of one volume is byte-identical, so the
// per-cube corner normal cache and the slice merge introduce no scheduling
// dependence.
// ---------------------------------------------------------------------------
void section_determinism() {
    ++g_sections;
    const int    R   = 24;
    const float  vs  = 0.04f;
    const float  rad = 0.25f;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    const auto& p = vol.params();
    const Eigen::Vector3f center(0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs);
    for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                const float s = (vol.voxelToWorld(x, y, z) - center).norm() - rad;
                auto& v = vol.voxelAt(x, y, z);
                if (s <= 1.2f * vs) {
                    v.tsdf   = std::max(-1.0f, std::min(1.0f, s / p.truncation));
                    v.weight = p.max_weight;
                } else {
                    v.tsdf   = kfusion::tsdf::EMPTY_TSDF;
                    v.weight = kfusion::tsdf::EMPTY_WEIGHT;
                }
                v.r = 5; v.g = 6; v.b = 7;
            }

    const std::string a = serializeBytes(*mesh(vol));
    const std::string b = serializeBytes(*mesh(vol));
    const std::string c = serializeBytes(*mesh(vol));
    std::printf("s8: bytes run1=%zu run2=%zu run3=%zu identical=%d\n", a.size(), b.size(), c.size(),
                (a == b && b == c) ? 1 : 0);
    CHECK(!a.empty(), "s8: extraction produced payload");
    CHECK(a == b && b == c, "s8: three repeated extractions are byte-identical");
}

} // namespace

int main() {
    section_stale_unobserved_corner();
    section_frontier_sphere_closure();
    section_unsupported_edge();
    section_cancelled_normal();
    section_non_finite_tsdf();
    section_degenerate_corner();
    section_border_one_sided_normal();
    section_determinism();

    if (g_failures == 0) {
        std::printf("marching_cubes_frontier_contract: PASS (%d sections)\n", g_sections);
        return 0;
    }
    std::printf("marching_cubes_frontier_contract: FAIL (%d failed check(s), %d sections)\n",
                g_failures, g_sections);
    return 1;
}
