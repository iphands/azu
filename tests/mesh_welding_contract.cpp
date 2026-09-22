// mesh_welding_contract (big-fix Todo 17): CPU-only contract that
// kfusion::meshing::MarchingCubes::extract welds by EXACT canonical edge identity
// - (lower voxel coordinate, axis) - and never by a quantized world position.
// Public CPU API only (TSDFVolume + MarchingCubes::extract); no device, display,
// GPU, sensor, thread timing, filesystem or network.
//
// The canonical rules under test (docs/CANONICAL_SEMANTICS.md, "Marching Cubes"):
//   W1 one physical crossing edge, reached from any of its up-to-four cubes or
//      across an OpenMP slice boundary, maps to exactly ONE global vertex;
//   W2 two distinct edge keys are never merged merely because their interpolated
//      positions collide or round to the same quantized bucket;
//   W3 the number of welded vertices equals an independent oracle: the distinct
//      canonical edge keys that participate in an emitted triangle, replayed from
//      the shared table + integer corner-sign + support algebra (never the
//      product's own weld).
//
// Expectations are independent oracles, never product snapshots. Sections that are
// already correct at HEAD are marked LOCK so the evidence cannot overclaim.

#include "meshing/MarchingCubes.h"
#include "meshing/MarchingCubesTables.h"
#include "tsdf/TSDFVolume.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

namespace {

using kfusion::meshing::MarchingCubes;
using kfusion::meshing::MeshData;
using kfusion::tsdf::EMPTY_TSDF;
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

// Same incidence the meshing stage uses (src/meshing/MarchingCubes.cpp).
const int kCornerOffsets[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
};
const int kEdgeCorners[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7},
};

constexpr float kWeightEps = 1e-3f;

struct Sample { float value; bool supported; };

Sample sampleVoxel(const TSDFVolume& vol, int x, int y, int z) {
    const int R = vol.params().resolution;
    if (x < 0 || y < 0 || z < 0 || x >= R || y >= R || z >= R)
        return {EMPTY_TSDF, false};
    const auto& v = vol.voxelAt(x, y, z);
    if (v.weight <= kWeightEps || !std::isfinite(v.tsdf)) return {EMPTY_TSDF, false};
    return {v.tsdf, true};
}

// Canonical edge key: lower endpoint voxel + the axis the two endpoints differ on.
// Only one coordinate changes across a unit edge, so the lower endpoint is the one
// with the smaller value on that axis (equivalently the lexicographic minimum).
struct EdgeKey {
    int x, y, z; uint8_t axis;
    bool operator<(const EdgeKey& o) const {
        if (x != o.x) return x < o.x;
        if (y != o.y) return y < o.y;
        if (z != o.z) return z < o.z;
        return axis < o.axis;
    }
};

EdgeKey canonicalKey(int cx, int cy, int cz, int e) {
    const int ca = kEdgeCorners[e][0], cb = kEdgeCorners[e][1];
    int ax = cx + kCornerOffsets[ca][0], ay = cy + kCornerOffsets[ca][1], az = cz + kCornerOffsets[ca][2];
    int bx = cx + kCornerOffsets[cb][0], by = cy + kCornerOffsets[cb][1], bz = cz + kCornerOffsets[cb][2];
    // Exactly one coordinate differs across a unit edge; that axis identifies the
    // edge and its lower endpoint (the smaller coordinate on that axis).
    if (ax != bx) return (ax <= bx) ? EdgeKey{ax, ay, az, 0} : EdgeKey{bx, by, bz, 0};
    if (ay != by) return (ay <= by) ? EdgeKey{ax, ay, az, 1} : EdgeKey{bx, by, bz, 1};
    return (az <= bz) ? EdgeKey{ax, ay, az, 2} : EdgeKey{bx, by, bz, 2};
}

// Independent oracle of the welded vertex set. Replays the canonical per-cube rule
// (weight-guarded corner signs, the shared edge mask, a crossing edge needs two
// supported endpoints, a triangle needs all three) and collects the DISTINCT
// canonical edge keys used by an emitted triangle. edge_table is independently
// pinned by marching_cubes_table_contract, so this count is not a product snapshot.
std::set<EdgeKey> oracleWeldedKeys(const TSDFVolume& vol) {
    std::set<EdgeKey> keys;
    const int R = vol.params().resolution;
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
                    const int c = kfusion::meshing::tables::tri_table[cfg][t + 2];
                    if (!edok[a] || !edok[b] || !edok[c]) continue;
                    const int tri[3] = {a, b, c};
                    for (int e : tri) keys.insert(canonicalKey(x, y, z, e));
                }
            }
    return keys;
}

size_t boundaryEdges(const MeshData& m) {
    std::set<std::array<uint32_t, 3>> uniq;
    for (size_t t = 0; t < m.triangleCount(); ++t)
        uniq.insert({m.indices[3 * t], m.indices[3 * t + 1], m.indices[3 * t + 2]});
    std::map<std::pair<uint32_t, uint32_t>, int> dir;
    for (const auto& tr : uniq)
        for (int i = 0; i < 3; ++i) ++dir[{tr[i], tr[(i + 1) % 3]}];
    size_t boundary = 0;
    for (const auto& kv : dir)
        if (dir.find({kv.first.second, kv.first.first}) == dir.end()) ++boundary;
    return boundary;
}

size_t vertsNear(const MeshData& m, const Eigen::Vector3f& q, float tol) {
    size_t n = 0;
    for (const auto& p : m.positions)
        if (std::abs(p.x() - q.x()) <= tol && std::abs(p.y() - q.y()) <= tol &&
            std::abs(p.z() - q.z()) <= tol) ++n;
    return n;
}

TSDFParams makeParams(int resolution, float voxel_size, float trunc) {
    TSDFParams p;
    p.resolution = resolution; p.voxel_size = voxel_size; p.truncation = trunc;
    p.max_weight = 8.0f; p.origin = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    return p;
}
std::shared_ptr<MeshData> mesh(const TSDFVolume& vol) { return MarchingCubes().extract(vol); }

// ---------------------------------------------------------------------------
// Section 1 (LOCK/RED): fully observed analytic sphere. Every crossing edge is
// supported, so the welded vertex count must equal the distinct-edge oracle, and a
// closed surface needs each shared edge to carry ONE index across adjacent cubes
// and slice boundaries. A quantized weld either false-merges (count < oracle) or
// false-splits a shared edge across a rounding boundary (count > oracle); exact
// edge welding lands exactly on the oracle.
// ---------------------------------------------------------------------------
void section_sphere_matches_oracle() {
    ++g_sections;
    const int   R   = 28;
    const float vs  = 0.03f;
    const float rad = 0.30f;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    const auto& p = vol.params();
    const Eigen::Vector3f center(0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs);
    for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                float t = ((vol.voxelToWorld(x, y, z) - center).norm() - rad) / p.truncation;
                auto& v = vol.voxelAt(x, y, z);
                v.tsdf   = std::max(-1.0f, std::min(1.0f, t));
                v.weight = p.max_weight;
                v.r = 120; v.g = 160; v.b = 200;
            }

    auto m = mesh(vol);
    CHECK(m != nullptr && !m->empty(), "s1: sphere meshes");
    if (!m || m->empty()) return;
    const auto oracle = oracleWeldedKeys(vol);
    std::printf("s1: verts=%zu oracle_keys=%zu tris=%zu boundary=%zu\n",
                m->positions.size(), oracle.size(), m->triangleCount(), boundaryEdges(*m));
    CHECK(m->positions.size() == oracle.size(),
          "s1: welded vertex count equals the distinct-edge oracle (no false merge / false split)");
    CHECK(boundaryEdges(*m) == 0, "s1: shared edges carry one index (closed surface)");
    CHECK(m->validate(nullptr), "s1: welded sphere passes MeshData::validate()");
}

// ---------------------------------------------------------------------------
// Section 2 (RED): the false-merge witness. A solid negative octant meets the
// positive field at an EXACTLY-zero corner voxel V. The three crossing edges out of
// V each snap their crossing to V's center (their zero endpoint), so they share one
// exact coordinate while carrying three DIFFERENT edge identities. A quantized weld
// collapses all three into one vertex; exact edge welding keeps three distinct
// vertices. This is the invariant Todo 17 makes exact rather than coincidental.
// ---------------------------------------------------------------------------
void section_zero_corner_not_false_merged() {
    ++g_sections;
    const int   R  = 10;
    const float vs = 0.05f;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    const int c = 5;                        // V at (5,5,5); negative block x,y,z <= 5 minus V
    for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                const bool inBlock = (x <= c && y <= c && z <= c);
                auto& v  = vol.voxelAt(x, y, z);
                v.tsdf   = inBlock ? -0.5f : (0.5f + 0.01f * static_cast<float>(x));
                v.weight = 8.0f;
                v.r = 40; v.g = 80; v.b = 120;
            }
    vol.voxelAt(c, c, c).tsdf = 0.0f;       // the exactly-zero shared corner (still observed)

    const int   ncross = 3;                 // -x,-y,-z neighbours of V are negative
    const Eigen::Vector3f V = vol.voxelToWorld(c, c, c);
    auto m = mesh(vol);
    CHECK(m != nullptr && !m->empty(), "s2: octant corner meshes");
    if (!m || m->empty()) return;

    const float tol = vs * 1e-3f;
    const size_t atV = vertsNear(*m, V, tol);
    const auto oracle = oracleWeldedKeys(vol);
    std::printf("s2: verts_at_V=%zu (want>=%d distinct edges) total_verts=%zu oracle_keys=%zu\n",
                atV, ncross, m->positions.size(), oracle.size());
    CHECK(atV >= ncross,
          "s2: distinct zero-endpoint edges at one coordinate are NOT falsely merged");
    CHECK(m->positions.size() == oracle.size(),
          "s2: total welded count equals the distinct-edge oracle (quantization would undercount)");
    CHECK(m->validate(nullptr), "s2: mesh passes MeshData::validate()");
}

// ---------------------------------------------------------------------------
// Section 3 (LOCK): repeated extraction is byte-identical, so the deterministic
// serial weld (slice order, then local insertion order) introduces no scheduling
// dependence in the vertex order or indices.
// ---------------------------------------------------------------------------
void section_determinism() {
    ++g_sections;
    const int   R  = 22;
    const float vs = 0.04f;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    const auto& p = vol.params();
    const Eigen::Vector3f center(0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs);
    const float rad = 0.25f;
    for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                const float t = ((vol.voxelToWorld(x, y, z) - center).norm() - rad) / p.truncation;
                auto& v = vol.voxelAt(x, y, z);
                v.tsdf   = std::max(-1.0f, std::min(1.0f, t));
                v.weight = p.max_weight;
                v.r = 7; v.g = 14; v.b = 21;
            }
    auto bytes = [](const MeshData& m) {
        std::string s;
        for (const auto& q : m.positions) { s.append(reinterpret_cast<const char*>(q.data()), 12); }
        for (const auto& n : m.normals)   { s.append(reinterpret_cast<const char*>(n.data()), 12); }
        for (auto cc : m.colors)          { s.push_back(static_cast<char>(cc)); }
        for (auto ii : m.indices)         { s.append(reinterpret_cast<const char*>(&ii), 4); }
        return s;
    };
    const std::string a = bytes(*mesh(vol)), b = bytes(*mesh(vol));
    std::printf("s3: bytes run1=%zu run2=%zu identical=%d\n", a.size(), b.size(), (a == b) ? 1 : 0);
    CHECK(!a.empty(), "s3: extraction produced payload");
    CHECK(a == b, "s3: two extractions are byte-identical");
}

} // namespace

int main() {
    section_sphere_matches_oracle();
    section_zero_corner_not_false_merged();
    section_determinism();

    if (g_failures == 0) {
        std::printf("mesh_welding_contract: PASS (%d sections)\n", g_sections);
        return 0;
    }
    std::printf("mesh_welding_contract: FAIL (%d failed check(s), %d sections)\n", g_failures, g_sections);
    return 1;
}
