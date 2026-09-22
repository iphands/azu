// mesh_color_invariant_contract (big-fix Todo 17): CPU-only contract that
// MarchingCubes::extract keeps MeshData color and geometry in exact lockstep and
// interpolates RGB with the SAME parameter that produced the vertex position.
// Public CPU API only (TSDFVolume + MarchingCubes::extract); no device, display,
// GPU, sensor, thread timing, filesystem or network.
//
// Canonical rules under test (docs/CANONICAL_SEMANTICS.md, "Mesh color"):
//   C1 a mesh either carries NO colors or exactly one RGB triple per vertex -
//      colors.size() == positions.size()*3 - never a partial buffer; hasColors()
//      is derived from that buffer, and validate() rejects any other size;
//   C2 a vertex color is the blend of its edge endpoints at the SAME interpolation
//      parameter that produced the position, so a color field that is affine in
//      world space lands on the exact per-vertex oracle regardless of which
//      adjacent cube reached the shared edge (the shared-`t` requirement,
//      meshing:D8);
//   C3 the exact edge-key weld keeps the color of a shared vertex consistent and
//      repeated extraction stays byte-identical.
//
// These are regression guards for the exact-weld rewrite: a guard already green at
// Todo 16 is labelled LOCK so the evidence never overclaims a fail-before-fix.

#include "meshing/MarchingCubes.h"
#include "tsdf/TSDFVolume.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using kfusion::meshing::MarchingCubes;
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

TSDFParams makeParams(int resolution, float voxel_size, float trunc) {
    TSDFParams p;
    p.resolution = resolution; p.voxel_size = voxel_size; p.truncation = trunc;
    p.max_weight = 8.0f; p.origin = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
    return p;
}
std::shared_ptr<MeshData> mesh(const TSDFVolume& vol) { return MarchingCubes().extract(vol); }

// Byte scale that keeps the encoded world-x inside [0, 255] for the whole grid.
uint8_t encodeX(float world_x, float scale) {
    const float v = std::round(world_x * scale);
    return static_cast<uint8_t>(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
}

// ---------------------------------------------------------------------------
// Section 1 (LOCK): a fully observed sphere whose per-voxel red channel encodes
// world x. Rule C1: exact color lockstep and validate(); rule C2: every vertex's
// red byte equals the same oracle applied to that vertex's own world x, within the
// rounding of the endpoint bytes. A red byte interpolated with a different
// parameter than the position would drift by ~scale*|dt|, far outside tolerance.
// ---------------------------------------------------------------------------
void section_lockstep_and_shared_parameter() {
    ++g_sections;
    const int   R = 26;
    const float vs = 0.04f;
    const float scale = 180.0f / static_cast<float>(R - 1) / vs;  // world_x * scale
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    const auto& p = vol.params();
    const Eigen::Vector3f center(0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs);
    const float rad = 0.30f;
    for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                const Eigen::Vector3f w = vol.voxelToWorld(x, y, z);
                const float t = ((w - center).norm() - rad) / p.truncation;
                auto& v = vol.voxelAt(x, y, z);
                v.tsdf   = std::max(-1.0f, std::min(1.0f, t));
                v.weight = p.max_weight;
                v.r = encodeX(w.x(), scale);   // color encodes world x
                v.g = 0; v.b = 0;
            }

    auto m = mesh(vol);
    CHECK(m != nullptr && !m->empty(), "s1: colored sphere meshes");
    if (!m || m->empty()) return;

    CHECK(m->hasColors(), "s1: a colored extraction reports hasColors()");
    CHECK(m->colors.size() == m->positions.size() * 3,
          "s1: colors.size() == positions.size()*3 (exact lockstep)");
    CHECK(m->validate(nullptr), "s1: colored mesh passes MeshData::validate()");
    CHECK(!m->truncated, "s1: an unbounded extraction is not truncated");

    size_t worst = 0;
    size_t checked = 0;
    for (size_t i = 0; i < m->positions.size(); ++i) {
        const uint8_t expected = encodeX(m->positions[i].x(), scale);
        const int got = m->colors[i * 3 + 0];
        const int d = std::abs(got - static_cast<int>(expected));
        if (static_cast<size_t>(d) > worst) worst = static_cast<size_t>(d);
        ++checked;
    }
    std::printf("s1: verts=%zu colors=%zu worst_color_vs_position_oracle=%zu\n",
                m->positions.size(), m->colors.size(), worst);
    // Endpoint bytes are rounded before blending and truncated after; <=3 leaves
    // room for that rounding but rejects a mismatched interpolation parameter.
    CHECK(worst <= 3, "s1: per-vertex color matches the world-x oracle at the vertex's own position (shared t)");
}

// ---------------------------------------------------------------------------
// Section 2 (LOCK): the octant fixture (shared exactly-zero corner) exercises the
// false-merge region from the welding contract; here it proves color lockstep and
// validate() still hold for every vertex the exact weld produces.
// ---------------------------------------------------------------------------
void section_octant_color_lockstep() {
    ++g_sections;
    const int   R  = 10;
    const float vs = 0.05f;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    const int c = 5;
    for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                const bool inBlock = (x <= c && y <= c && z <= c);
                auto& v  = vol.voxelAt(x, y, z);
                v.tsdf   = inBlock ? -0.5f : (0.5f + 0.01f * static_cast<float>(x));
                v.weight = 8.0f;
                v.r = inBlock ? 30 : 200; v.g = 90; v.b = inBlock ? 40 : 160;
            }
    vol.voxelAt(c, c, c).tsdf = 0.0f;

    auto m = mesh(vol);
    CHECK(m != nullptr && !m->empty(), "s2: octant corner meshes");
    if (!m || m->empty()) return;
    std::printf("s2: verts=%zu colors=%zu lockstep=%d\n",
                m->positions.size(), m->colors.size(),
                (m->colors.size() == m->positions.size() * 3) ? 1 : 0);
    CHECK(m->colors.size() == m->positions.size() * 3,
          "s2: colors stay in exact lockstep through the shared-corner weld");
    CHECK(m->validate(nullptr), "s2: octant mesh passes MeshData::validate()");
}

// ---------------------------------------------------------------------------
// Section 3 (LOCK): a constant-colored volume still yields a full color buffer
// (every vertex colored), never a partial or empty one, so hasColors() and
// validate() agree for the trivial-color case too.
// ---------------------------------------------------------------------------
void section_constant_color() {
    ++g_sections;
    const int   R = 20;
    const float vs = 0.04f;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    const auto& p = vol.params();
    const Eigen::Vector3f center(0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs);
    const float rad = 0.24f;
    for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                const float t = ((vol.voxelToWorld(x, y, z) - center).norm() - rad) / p.truncation;
                auto& v = vol.voxelAt(x, y, z);
                v.tsdf   = std::max(-1.0f, std::min(1.0f, t));
                v.weight = p.max_weight;
                v.r = 128; v.g = 128; v.b = 128;   // constant color
            }
    auto m = mesh(vol);
    CHECK(m != nullptr && !m->empty(), "s3: constant-color sphere meshes");
    if (!m || m->empty()) return;
    bool all_128 = (m->colors.size() == m->positions.size() * 3);
    for (auto cc : m->colors) if (cc != 128) all_128 = false;
    std::printf("s3: verts=%zu colors=%zu all_128=%d\n",
                m->positions.size(), m->colors.size(), all_128 ? 1 : 0);
    CHECK(all_128, "s3: a constant-color volume yields a full, uniform color buffer");
    CHECK(m->validate(nullptr), "s3: constant-color mesh passes MeshData::validate()");
}

// ---------------------------------------------------------------------------
// Section 4 (LOCK): repeated extraction is byte-identical including colors, so the
// weld assigns each shared vertex ONE color deterministically.
// ---------------------------------------------------------------------------
void section_color_determinism() {
    ++g_sections;
    const int   R = 22;
    const float vs = 0.04f;
    const float scale = 180.0f / static_cast<float>(R - 1) / vs;
    TSDFVolume vol(makeParams(R, vs, 3.0f * vs));
    const auto& p = vol.params();
    const Eigen::Vector3f center(0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs, 0.5f * (R - 1) * vs);
    const float rad = 0.25f;
    for (int z = 0; z < R; ++z)
        for (int y = 0; y < R; ++y)
            for (int x = 0; x < R; ++x) {
                const Eigen::Vector3f w = vol.voxelToWorld(x, y, z);
                const float t = ((w - center).norm() - rad) / p.truncation;
                auto& v = vol.voxelAt(x, y, z);
                v.tsdf   = std::max(-1.0f, std::min(1.0f, t));
                v.weight = p.max_weight;
                v.r = encodeX(w.x(), scale); v.g = 10; v.b = 20;
            }
    auto bytes = [](const MeshData& m) {
        std::string s;
        for (const auto& q : m.positions) s.append(reinterpret_cast<const char*>(q.data()), 12);
        for (auto cc : m.colors)          s.push_back(static_cast<char>(cc));
        for (auto ii : m.indices)         s.append(reinterpret_cast<const char*>(&ii), 4);
        return s;
    };
    const std::string a = bytes(*mesh(vol)), b = bytes(*mesh(vol));
    std::printf("s4: color bytes run1=%zu run2=%zu identical=%d\n", a.size(), b.size(), (a == b) ? 1 : 0);
    CHECK(!a.empty(), "s4: colored extraction produced payload");
    CHECK(a == b, "s4: two extractions have identical positions+colors+indices");
}

} // namespace

int main() {
    section_lockstep_and_shared_parameter();
    section_octant_color_lockstep();
    section_constant_color();
    section_color_determinism();

    if (g_failures == 0) {
        std::printf("mesh_color_invariant_contract: PASS (%d sections)\n", g_sections);
        return 0;
    }
    std::printf("mesh_color_invariant_contract: FAIL (%d failed check(s), %d sections)\n", g_failures, g_sections);
    return 1;
}
