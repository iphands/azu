// marching_cubes_winding_contract (big-fix Todo 18): CPU-only contract that derives
// the authoritative CPU triangle-winding convention from signed-volume /
// outward-normal mathematics over an analytic signed-distance sphere, and locks
// MarchingCubes::extract against it. Public CPU API only (TSDFVolume +
// MarchingCubes::extract); no device, display, GPU, sensor, thread timing,
// filesystem or network.
//
// WHY THE RULE IS DERIVED, NOT SNAPSHOT
// The expected orientation is NEVER "whatever the product emits today" and NEVER a
// hardcoded expected tri_table order. It is derived in two independent steps:
//
//   D1 (section 0, pure math): in a right-handed coordinate system the divergence
//      theorem makes the signed volume
//          V_signed = (1/6) * sum_i a_i . (b_i x c_i)
//      positive exactly when a closed surface is oriented outward. Section 0
//      self-tests this rule on a hand-built unit cube whose outward orientation is
//      geometrically unambiguous (every face normal points away from the enclosed
//      center, asserted per face): V_signed must equal +1 = the true volume.
//      Nothing from Marching Cubes participates, so the rule cannot be a product
//      snapshot.
//   D2 (sections 1-4, the fixture): the fixture is an analytic signed-distance
//      sphere with a KNOWN sign structure - tsdf < 0 strictly inside, tsdf > 0
//      strictly outside (verified voxel-by-voxel against the analytic distance
//      BEFORE any extraction). The outward normal is then the direction of
//      increasing signed distance, i.e. away from the known center. The mesh is
//      outward iff for every face
//          ((b - a) x (c - a)) . (centroid - center) > 0
//      and, globally, V_signed > 0. These are the authoritative CPU winding
//      fixtures named by docs/CANONICAL_SEMANTICS.md; the table order must serve
//      them, never the other way around.
//
// Which emitted order (table order 0,1,2 or reverse 2,1,0) is outward is then a
// THEOREM of the fixture, reported as data and locked by section 5's per-cube
// both-candidate derivation: for single-triangle cube configurations both
// candidate orders of the SAME three independently computed crossing vertices are
// evaluated against the outward reference direction, exactly one wins, and the
// product must agree.
//
// Sections are marked LOCK (already true at HEAD, pinned so it stays true) rather
// than claimed as fixes. Repeated extraction and OMP thread counts 1/2/4 must
// reproduce identical geometry and the identical derived sign (section 6).

#include "meshing/MarchingCubes.h"
#include "meshing/MarchingCubesTables.h"
#include "tsdf/TSDFVolume.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

using kfusion::meshing::MarchingCubes;
using kfusion::meshing::MeshData;
using kfusion::meshing::tables::edge_table;
using kfusion::meshing::tables::tri_table;
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

// Same incidence the meshing stage documents (src/meshing/MarchingCubes.cpp):
// corner 0..7 offsets and the endpoint pair of each cube edge 0..11.
const int kCornerOffsets[8][3] = {
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1},
};
const int kEdgeCorners[12][2] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 0},
    {4, 5}, {5, 6}, {6, 7}, {7, 4},
    {0, 4}, {1, 5}, {2, 6}, {3, 7},
};

// ---- byte digest + OpenMP pinning (same discipline as the race contract) ----
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
std::string serializeMesh(const MeshData& m) {
    std::string out;
    for (const auto& p : m.positions) { appendF32(out, p.x()); appendF32(out, p.y()); appendF32(out, p.z()); }
    for (const auto& n : m.normals)   { appendF32(out, n.x()); appendF32(out, n.y()); appendF32(out, n.z()); }
    for (uint8_t c : m.colors)         out.push_back(static_cast<char>(c));
    for (uint32_t i : m.indices)       appendU32(out, i);
    return out;
}
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

// The signed-volume formula under test, accumulated in double over triangle
// records given in their EMITTED winding order.
double signedVolume(const std::vector<std::array<Eigen::Vector3f, 3>>& tris) {
    double v = 0.0;
    for (const auto& t : tris) {
        const Eigen::Vector3d a = t[0].cast<double>();
        const Eigen::Vector3d b = t[1].cast<double>();
        const Eigen::Vector3d c = t[2].cast<double>();
        v += a.dot(b.cross(c)) / 6.0;
    }
    return v;
}

// ---------------------------------------------------------------------------
// Section 0 (D1, pure math): DERIVE the outward rule. A hand-built unit cube
// [0,1]^3 with an unambiguously outward winding must carry V_signed = +1 through
// the same formula the sphere sections apply. LOCK of the math, not the product.
// ---------------------------------------------------------------------------
void section_derive_outward_rule() {
    ++g_sections;
    const Eigen::Vector3f rx = Eigen::Vector3f::UnitX().cross(Eigen::Vector3f::UnitY());
    CHECK((rx - Eigen::Vector3f::UnitZ()).norm() < 1e-6f,
          "s0: coordinate system is right-handed (e_x x e_y = e_z)");

    // Six quad() calls build the 12 cube triangles. For the plane at s on `axis`,
    // u x v = +axis; winding (u, u+v) faces +axis on the s=1 face, and `flip`
    // reverses both triangles so the s=0 face faces -axis. "Outward" here is the
    // geometric statement "away from the enclosed center", re-asserted by the
    // per-face radial check below - it is never taken from product code.
    std::vector<std::array<Eigen::Vector3f, 3>> tris;
    auto quad = [&](Eigen::Vector3f a, Eigen::Vector3f u, Eigen::Vector3f v, bool flip) {
        const Eigen::Vector3f b = a + u, c = a + u + v, d = a + v;
        if (!flip) { tris.push_back({a, b, c}); tris.push_back({a, c, d}); }
        else       { tris.push_back({a, d, c}); tris.push_back({a, c, b}); }
    };
    quad(Eigen::Vector3f(1, 0, 0), Eigen::Vector3f(0, 1, 0), Eigen::Vector3f(0, 0, 1), false); // x=1: e_y x e_z = +e_x
    quad(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 1, 0), Eigen::Vector3f(0, 0, 1), true);  // x=0 -> -e_x
    quad(Eigen::Vector3f(0, 1, 0), Eigen::Vector3f(0, 0, 1), Eigen::Vector3f(1, 0, 0), false); // y=1: e_z x e_x = +e_y
    quad(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), Eigen::Vector3f(1, 0, 0), true);  // y=0 -> -e_y
    quad(Eigen::Vector3f(0, 0, 1), Eigen::Vector3f(1, 0, 0), Eigen::Vector3f(0, 1, 0), false); // z=1: e_x x e_y = +e_z
    quad(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0), Eigen::Vector3f(0, 1, 0), true);  // z=0 -> -e_z
    CHECK(tris.size() == 12, "s0: hand-built cube has 12 triangles");

    int bad = 0;
    const Eigen::Vector3f ctr(0.5f, 0.5f, 0.5f);
    for (const auto& t : tris) {
        const Eigen::Vector3f n = (t[1] - t[0]).cross(t[2] - t[0]);
        const Eigen::Vector3f g = (t[0] + t[1] + t[2]) / 3.0f - ctr;
        if (n.dot(g) <= 0.0f) ++bad;
    }
    CHECK(bad == 0, "s0: cube winding is outward by the radial definition (self-check)");

    const double V = signedVolume(tris);
    std::printf("s0: DERIVATION hand-built unit cube V_signed=%.17g (must be +1 = true volume)\n", V);
    CHECK(std::abs(V - 1.0) < 1e-12,
          "s0: outward closed winding gives POSITIVE signed volume equal to the true volume");
    std::printf("s0: RULE outward orientation in right-handed world space <=> V_signed > 0 and "
                "face_normal . (centroid - interior_point) > 0; derived from math, not from product code\n");
}

// ---------------------------------------------------------------------------
// Sphere fixture: 48^3 voxels at 2 cm, radius 30 cm, centered at (0.47,0.47,0.47)
// of the [0, 0.94] voxel-center lattice. Fully interior, fully observed, finite.
// Same construction as marching_cubes_sphere_contract; the sign structure is the
// KNOWN signed-distance field: tsdf < 0 strictly inside, > 0 strictly outside.
// ---------------------------------------------------------------------------
constexpr int    kResolution = 48;
constexpr float  kVoxelSize  = 0.02f;
constexpr float  kRadius     = 0.30f;

struct Sphere { Eigen::Vector3f center; float radius; };

float analyticDistance(const TSDFVolume& vol, int x, int y, int z, const Sphere& s) {
    return (vol.voxelToWorld(x, y, z) - s.center).norm() - s.radius;
}

void fillSphereVolume(TSDFVolume& vol, const Sphere& s) {
    const auto& p = vol.params();
    for (int z = 0; z < p.resolution; ++z)
        for (int y = 0; y < p.resolution; ++y)
            for (int x = 0; x < p.resolution; ++x) {
                float t = analyticDistance(vol, x, y, z, s) / p.truncation;
                t = std::max(-1.0f, std::min(1.0f, t));
                auto& v = vol.voxelAt(x, y, z);
                v.tsdf   = t;
                v.weight = p.max_weight;
                v.r = 120; v.g = 160; v.b = 200;
            }
}

TSDFParams makeParams() {
    TSDFParams p;
    p.resolution = kResolution;
    p.voxel_size = kVoxelSize;
    p.truncation = 2.5f * kVoxelSize;
    p.max_weight = 64.0f;
    p.origin     = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    return p;
}

// The product's documented corner-sign rule (tsdf < 0 => inside; empty mask => no
// cube), recomputed here as the independent enumeration used by the UB gate and
// by section 5. No orientation is involved.
int cubeConfig(const TSDFVolume& vol, int x, int y, int z) {
    int cfg = 0;
    for (int c = 0; c < 8; ++c)
        if (vol.voxelAt(x + kCornerOffsets[c][0], y + kCornerOffsets[c][1],
                        z + kCornerOffsets[c][2]).tsdf < 0.0f)
            cfg |= (1 << c);
    if (edge_table[cfg] == 0) return -1;
    return cfg;
}

// Defensive fail-closed gate (same discipline as the sphere contract): refuse to
// extract while the fixture touches configs 213/214/215, which pre-Todo-7 read
// uninitialized edge slots. The rows are repaired, so this now only proves the
// fixture never routes through a historically UB configuration.
bool fixtureAvoidsLegacyUBConfigs(const TSDFVolume& vol) {
    const int R = vol.params().resolution;
    for (int z = 0; z + 1 < R; ++z)
        for (int y = 0; y + 1 < R; ++y)
            for (int x = 0; x + 1 < R; ++x) {
                const int cfg = cubeConfig(vol, x, y, z);
                if (cfg == 213 || cfg == 214 || cfg == 215) return false;
            }
    return true;
}

// Unique triangle records, keyed by the EMITTED winding order (sorting the triple
// would erase the very orientation under test).
std::vector<std::array<uint32_t, 3>> uniqueTriangles(const MeshData& m) {
    std::set<std::array<uint32_t, 3>> uniq;
    for (size_t t = 0; t < m.triangleCount(); ++t)
        uniq.insert({m.indices[3 * t], m.indices[3 * t + 1], m.indices[3 * t + 2]});
    return {uniq.begin(), uniq.end()};
}

// ---------------------------------------------------------------------------
// Section 1: the fixture's sign structure is KNOWN analytically BEFORE any
// extraction: negative strictly inside, positive strictly outside, finite, fully
// observed, fully interior. This is what makes the fixture authoritative.
// ---------------------------------------------------------------------------
void section_fixture_sign_is_known(const TSDFVolume& vol, const Sphere& s) {
    ++g_sections;
    const auto& p = vol.params();
    int neg = 0, pos = 0, wrong_sign = 0, unsupported = 0, nonfinite = 0, boundary_inside = 0;
    for (int z = 0; z < p.resolution; ++z)
        for (int y = 0; y < p.resolution; ++y)
            for (int x = 0; x < p.resolution; ++x) {
                const auto& v = vol.voxelAt(x, y, z);
                if (!std::isfinite(v.tsdf)) ++nonfinite;
                if (!(v.weight > 0.001f))   ++unsupported;
                const bool inside = analyticDistance(vol, x, y, z, s) < 0.0f;
                if ((v.tsdf < 0.0f) != inside) ++wrong_sign;
                if (v.tsdf < 0.0f) ++neg; else ++pos;
                const bool boundary = (x == 0 || y == 0 || z == 0 ||
                                       x == p.resolution - 1 || y == p.resolution - 1 ||
                                       z == p.resolution - 1);
                if (boundary && inside) ++boundary_inside;
            }
    std::printf("s1: voxels=%d inside_neg=%d outside_pos=%d wrong_sign=%d nonfinite=%d "
                "unsupported=%d boundary_inside=%d (fully-observed interior fixture requires 0/0/0/0)\n",
                p.resolution * p.resolution * p.resolution, neg, pos, wrong_sign,
                nonfinite, unsupported, boundary_inside);
    CHECK(nonfinite == 0,       "s1: every fixture tsdf is finite");
    CHECK(unsupported == 0,     "s1: every fixture voxel is observed (weight above the meshing epsilon)");
    CHECK(wrong_sign == 0,      "s1: sign(tsdf) == inside/outside of the analytic sphere for EVERY voxel");
    CHECK(neg > 0 && pos > 0,   "s1: fixture has both negative-interior and positive-exterior support");
    CHECK(boundary_inside == 0, "s1: sphere fully interior (no boundary voxel inside, so the surface must close)");
}

// ---------------------------------------------------------------------------
// Section 2: closed + manifold + non-degenerate over the EMITTED winding order
// (directed-edge pairing; only exact duplicate winding records collapse).
// ---------------------------------------------------------------------------
void section_closed_manifold(const MeshData& m) {
    ++g_sections;
    const auto tris = uniqueTriangles(m);
    std::map<std::pair<uint32_t, uint32_t>, int> directed;
    for (const auto& t : tris)
        for (int i = 0; i < 3; ++i) ++directed[{t[i], t[(i + 1) % 3]}];
    size_t boundary = 0, nonmanifold = 0;
    for (const auto& kv : directed) {
        const auto rev = directed.find({kv.first.second, kv.first.first});
        const int rc = (rev == directed.end()) ? 0 : rev->second;
        if (rc == 0) ++boundary;
        if (kv.second > 1 || rc > 1) ++nonmanifold;
    }
    size_t degenerate = 0;
    for (const auto& t : tris) {
        const Eigen::Vector3f& a = m.positions[t[0]];
        const Eigen::Vector3f& b = m.positions[t[1]];
        const Eigen::Vector3f& c = m.positions[t[2]];
        if ((b - a).cross(c - a).norm() <= 1e-9f) ++degenerate;
    }
    std::printf("s2: tris=%zu unique=%zu boundary_edges=%zu nonmanifold=%zu degenerate=%zu\n",
                m.triangleCount(), tris.size(), boundary, nonmanifold, degenerate);
    CHECK(boundary == 0,    "s2: emitted winding closes the surface (every directed edge has one reverse partner)");
    CHECK(nonmanifold == 0, "s2: emitted winding is manifold");
    CHECK(degenerate == 0,  "s2: no degenerate triangle records");
}

// ---------------------------------------------------------------------------
// Sections 3+4: apply the derived rule to the product mesh. Every MC crossing is
// a chord zero of the convex sphere SDF, so all vertices lie strictly INSIDE the
// true sphere and the closed polyhedron is inscribed: 0 < V_signed < V_sphere,
// with the deficit bounded by the chord-sagitta bound 3*(3*dx^2/(8r))/r ~ 0.5%.
// The 1% band is therefore derived, not tuned, and an inward mesh (V < 0) fails.
// ---------------------------------------------------------------------------
void section_signed_volume_and_per_face(const MeshData& m, const Sphere& s) {
    ++g_sections;
    const auto tris = uniqueTriangles(m);
    std::vector<std::array<Eigen::Vector3f, 3>> rec;
    rec.reserve(tris.size());
    for (const auto& t : tris) rec.push_back({m.positions[t[0]], m.positions[t[1]], m.positions[t[2]]});

    const double V = signedVolume(rec);
    const double Vs = (4.0 / 3.0) * M_PI * static_cast<double>(kRadius) * kRadius * kRadius;
    std::printf("s3: V_signed=%+.10f analytic_sphere=%+.10f ratio=%.8f (rule: 0.99*V0 < V < V0)\n",
                V, Vs, V / Vs);
    CHECK(V > 0.0,       "s3: signed volume POSITIVE => outward orientation under the derived rule");
    CHECK(V < Vs,        "s3: inscribed mesh => V_signed strictly below the analytic sphere volume");
    CHECK(V > 0.99 * Vs, "s3: discrete-MC deficit stays inside the derived 1%% chord-sagitta band");

    int outward = 0, inward = 0, tangential = 0, degenerate = 0;
    double min_cos = 10.0;
    for (const auto& t : rec) {
        const Eigen::Vector3f n = (t[1] - t[0]).cross(t[2] - t[0]);
        if (n.norm() <= 1e-9f) { ++degenerate; continue; }
        const Eigen::Vector3f g = (t[0] + t[1] + t[2]) / 3.0f - s.center;
        const double cosang = n.cast<double>().normalized().dot(g.cast<double>().normalized());
        min_cos = std::min(min_cos, cosang);
        if (cosang > 0.5)      ++outward;   // strictly more outward than tangential
        else if (cosang < 0.0) ++inward;
        else                   ++tangential;
    }
    std::printf("s4: per-face radial test: outward=%d inward=%d tangential=%d degenerate=%d min_cos=%.6f "
                "(rule: cos(face_normal, radial) > 0.5 for EVERY face)\n",
                outward, inward, tangential, degenerate, min_cos);
    CHECK(inward == 0,     "s4: zero inward faces under the strict radial test");
    CHECK(tangential == 0, "s4: zero tangential/ambiguous faces");
    CHECK(outward > 0 && outward == static_cast<int>(rec.size()) - degenerate,
          "s4: every nondegenerate face is strictly outward");
}

// ---------------------------------------------------------------------------
// Section 5: per-cube BOTH-CANDIDATE derivation. For single-triangle cube
// configurations (config 1, tri row (0,8,3)), independently recompute the three
// crossing vertices via the documented lower-endpoint linear zero-crossing,
// evaluate BOTH candidate orders against the outward reference direction, and
// prove exactly one order is mathematically required - and that the product's
// emitted triangle over those same welded vertices uses precisely that order.
// The winning order IS the derived convention; the table order is never assumed.
// ---------------------------------------------------------------------------
uint32_t fbits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
std::array<uint32_t, 3> posKey(const Eigen::Vector3f& p) { return {fbits(p.x()), fbits(p.y()), fbits(p.z())}; }

// The canonical crossing formula restated independently of the product: parameter
// measured from the LOWER endpoint, t = v_low / (v_low - v_up) clamped to [0,1];
// a zero endpoint IS the crossing (the interpolation rule locked by Todo 17; it
// carries no assumption about triangle orientation).
Eigen::Vector3f crossingOnEdge(const TSDFVolume& vol, int cx, int cy, int cz, int e) {
    const int c0 = kEdgeCorners[e][0], c1 = kEdgeCorners[e][1];
    const int d0x = cx + kCornerOffsets[c0][0], d0y = cy + kCornerOffsets[c0][1], d0z = cz + kCornerOffsets[c0][2];
    const int d1x = cx + kCornerOffsets[c1][0], d1y = cy + kCornerOffsets[c1][1], d1z = cz + kCornerOffsets[c1][2];
    int lc = c0, uc = c1;
    if (d0x != d1x)      { lc = (d0x <= d1x) ? c0 : c1; uc = (d0x <= d1x) ? c1 : c0; }
    else if (d0y != d1y) { lc = (d0y <= d1y) ? c0 : c1; uc = (d0y <= d1y) ? c1 : c0; }
    else                 { lc = (d0z <= d1z) ? c0 : c1; uc = (d0z <= d1z) ? c1 : c0; }
    const float v_low = vol.voxelAt(cx + kCornerOffsets[lc][0], cy + kCornerOffsets[lc][1],
                                    cz + kCornerOffsets[lc][2]).tsdf;
    const float v_up  = vol.voxelAt(cx + kCornerOffsets[uc][0], cy + kCornerOffsets[uc][1],
                                    cz + kCornerOffsets[uc][2]).tsdf;
    const Eigen::Vector3f pl = vol.voxelToWorld(cx + kCornerOffsets[lc][0], cy + kCornerOffsets[lc][1],
                                                cz + kCornerOffsets[lc][2]);
    const Eigen::Vector3f pu = vol.voxelToWorld(cx + kCornerOffsets[uc][0], cy + kCornerOffsets[uc][1],
                                                cz + kCornerOffsets[uc][2]);
    if (std::abs(v_low) < 1e-6f) return pl;
    if (std::abs(v_up)  < 1e-6f) return pu;
    const float diff = v_low - v_up;
    if (std::abs(diff) < 1e-6f)  return pl;
    const float t = std::max(0.0f, std::min(1.0f, v_low / diff));
    return pl + t * (pu - pl);
}

double faceRadialDot(const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Vector3f& c,
                     const Sphere& s) {
    const Eigen::Vector3f n = (b - a).cross(c - a);
    const Eigen::Vector3f g = (a + b + c) / 3.0f - s.center;
    return static_cast<double>(n.dot(g));
}

void section_per_cube_candidate_derivation(const TSDFVolume& vol, const MeshData& m, const Sphere& s) {
    ++g_sections;
    std::map<std::array<uint32_t, 3>, uint32_t> index_at;
    for (uint32_t i = 0; i < m.positions.size(); ++i) index_at[posKey(m.positions[i])] = i;

    const int R = vol.params().resolution;
    int sampled = 0, agree = 0, disagree = 0, reverse_win = 0, forward_win = 0, unlocated = 0;
    for (int z = 0; z + 1 < R && sampled < 8; ++z)
        for (int y = 0; y + 1 < R && sampled < 8; ++y)
            for (int x = 0; x + 1 < R && sampled < 8; ++x) {
                const int cfg = cubeConfig(vol, x, y, z);
                if (cfg != 1) continue;                  // exactly one tri row
                if (tri_table[cfg][3] != -1) continue;   // guard: single-triangle row only
                const Eigen::Vector3f p0 = crossingOnEdge(vol, x, y, z, tri_table[cfg][0]);
                const Eigen::Vector3f p1 = crossingOnEdge(vol, x, y, z, tri_table[cfg][1]);
                const Eigen::Vector3f p2 = crossingOnEdge(vol, x, y, z, tri_table[cfg][2]);
                const double fwd_dot = faceRadialDot(p0, p1, p2, s);          // candidate: table order
                const double rev_dot = faceRadialDot(p2, p1, p0, s);          // candidate: reverse order
                if (fwd_dot == 0.0 || rev_dot == 0.0 || (fwd_dot > 0) == (rev_dot > 0)) {
                    CHECK(false, "s5: the two candidates must be antipodal (exactly one outward)");
                    continue;
                }
                ++sampled;
                const bool rev_outward = rev_dot > 0.0;
                if (rev_outward) ++reverse_win; else ++forward_win;

                // Locate the product triangle over the SAME three welded vertices
                // (Todo 17 makes the welded positions bit-identical to an
                // independently interpolated crossing) and compare its emitted order
                // with the mathematically required one.
                const auto i0i = index_at.find(posKey(p0));
                const auto i1i = index_at.find(posKey(p1));
                const auto i2i = index_at.find(posKey(p2));
                bool found = false;
                if (i0i != index_at.end() && i1i != index_at.end() && i2i != index_at.end()) {
                    const uint32_t i0 = i0i->second, i1 = i1i->second, i2 = i2i->second;
                    for (size_t t = 0; t < m.triangleCount(); ++t) {
                        const uint32_t a = m.indices[3 * t], b = m.indices[3 * t + 1], c = m.indices[3 * t + 2];
                        const bool fwd_emit = (a == i0 && b == i1 && c == i2);
                        const bool rev_emit = (a == i2 && b == i1 && c == i0);
                        if (!fwd_emit && !rev_emit) continue;
                        found = true;
                        const bool emitted_is_outward = rev_outward ? rev_emit : fwd_emit;
                        if (emitted_is_outward) ++agree; else ++disagree;
                        break;
                    }
                }
                if (!found) ++unlocated;
                std::printf("s5: cube(%d,%d,%d) cfg=%d tri_table=(%d,%d,%d) dot(table order 0,1,2)=%+.9e "
                            "dot(reverse 2,1,0)=%+.9e -> required order: %s | product emitted %s\n",
                            x, y, z, cfg, tri_table[cfg][0], tri_table[cfg][1], tri_table[cfg][2],
                            fwd_dot, rev_dot, rev_outward ? "REVERSE (2,1,0)" : "FORWARD (0,1,2)",
                            found ? (rev_outward ? "REVERSE (2,1,0)" : "FORWARD (0,1,2)")
                                  : "triangle NOT located");
                CHECK(found, "s5: product emitted a triangle over the independently derived crossing vertices");
            }
    std::printf("s5: sampled single-triangle cubes=%d reverse_wins=%d forward_wins=%d "
                "product_agrees=%d product_disagrees=%d unlocated=%d\n",
                sampled, reverse_win, forward_win, agree, disagree, unlocated);
    CHECK(sampled > 0,                          "s5: fixture exposes single-triangle (cfg 1) cubes to derive from");
    CHECK(unlocated == 0,                       "s5: every derived candidate triangle was located in the product mesh");
    CHECK(reverse_win == 0 || forward_win == 0, "s5: one candidate order is outward for EVERY sampled cube (a single convention exists)");
    CHECK(disagree == 0,                        "s5: product emission order equals the mathematically required order");
    std::printf("s5: DERIVED CPU convention (from the signed-distance fixture, never from the table): %s\n",
                (reverse_win > 0 && forward_win == 0) ? "REVERSE table order (2,1,0)" :
                (forward_win > 0 && reverse_win == 0) ? "FORWARD table order (0,1,2)" : "AMBIGUOUS");
}

// ---------------------------------------------------------------------------
// Section 6: determinism of the derived orientation. Repeated extraction and
// OpenMP thread counts 1/2/4 produce byte-identical geometry, so the derived
// sign is a property of the field, not of the schedule.
// ---------------------------------------------------------------------------
void section_thread_determinism(const TSDFVolume& vol) {
    ++g_sections;
    const uint64_t d0 = fnv1a(serializeMesh(*MarchingCubes().extract(vol)));
    const uint64_t d1 = fnv1a(serializeMesh(*MarchingCubes().extract(vol)));
    uint64_t dt[3] = {0, 0, 0};
    const int counts[3] = {1, 2, 4};
    for (int k = 0; k < 3; ++k) {
        pinThreads(counts[k]);
        dt[k] = fnv1a(serializeMesh(*MarchingCubes().extract(vol)));
        restoreThreads();
    }
    std::printf("s6: mesh_digest repeat=%016llx/%016llx threads1=%016llx threads2=%016llx threads4=%016llx\n",
                static_cast<unsigned long long>(d0), static_cast<unsigned long long>(d1),
                static_cast<unsigned long long>(dt[0]), static_cast<unsigned long long>(dt[1]),
                static_cast<unsigned long long>(dt[2]));
    CHECK(d0 != 0 && d1 != 0, "s6: extraction produced payload");
    CHECK(d0 == d1 && d0 == dt[0] && d0 == dt[1] && d0 == dt[2],
          "s6: geometry (and therefore the derived sign) is identical across repeats and threads 1/2/4");
}

} // namespace

int main() {
    section_derive_outward_rule();
    if (g_failures > 0) {
        std::printf("marching_cubes_winding_contract: FAIL (the derived rule itself failed)\n");
        return 1;
    }

    TSDFVolume vol(makeParams());
    const float half = 0.5f * static_cast<float>(kResolution - 1) * kVoxelSize;
    const Sphere sphere{Eigen::Vector3f(half, half, half), kRadius};
    fillSphereVolume(vol, sphere);

    section_fixture_sign_is_known(vol, sphere);
    if (!fixtureAvoidsLegacyUBConfigs(vol)) {
        std::printf("marching_cubes_winding_contract: FAIL (fixture touches legacy UB configs 213/214/215; "
                    "refusing to call extract())\n");
        return 1;
    }
    if (g_failures > 0) {
        std::printf("marching_cubes_winding_contract: FAIL (fixture gate)\n");
        return 1;
    }

    auto mesh = MarchingCubes().extract(vol);
    CHECK(mesh != nullptr && !mesh->empty(), "extract returns a non-empty mesh");
    if (!mesh || mesh->empty()) {
        std::printf("marching_cubes_winding_contract: FAIL (empty mesh)\n");
        return 1;
    }
    section_closed_manifold(*mesh);
    section_signed_volume_and_per_face(*mesh, sphere);
    section_per_cube_candidate_derivation(vol, *mesh, sphere);
    section_thread_determinism(vol);

    if (g_failures == 0) {
        std::printf("marching_cubes_winding_contract: PASS (%d sections; outward winding derived from the "
                    "signed-distance sphere, table order never assumed)\n", g_sections);
        return 0;
    }
    std::printf("marching_cubes_winding_contract: FAIL (%d failed check(s), %d sections)\n", g_failures, g_sections);
    return 1;
}
