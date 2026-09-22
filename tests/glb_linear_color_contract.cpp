// glb_linear_color_contract (big-fix Todo 19): CPU-only contract that GLBExporter
// writes glTF 2.0 COLOR_0 as LINEAR float vertex colors, decoded from the canonical
// uint8 sRGB MeshData buffer, and refuses to export a color it cannot represent
// (docs/CANONICAL_SEMANTICS.md, "Color pipeline and export").
//
// The test exports REAL temporary .glb files through the public exporter and reads
// them back with tinygltf, so the accessor metadata is verified as it lands on disk
// (componentType FLOAT, type VEC4, count == vertex count, alpha exactly 1.0) rather
// than from the in-memory buffer. Color values are compared against an independent
// double-precision sRGB EOTF oracle written here, never against the product helper.
//
// Canonical rules under test:
//   G1 COLOR_0 is float VEC4 with one entry per (welded) vertex and alpha exactly 1.0;
//   G2 RGB channels are the linear decode of the sRGB bytes - NOT the raw byte/255,
//      which is what the old exporter emitted and which this test would fail;
//   G3 the exporter decode is checked, not sanitized: a non-finite OR a finite
//      out-of-range component is refused and no output channel is written. This is
//      deliberately STRICTER than the CPU extraction boundary
//      (utils::srgbFloatToUint8), which saturates a finite out-of-range value to the
//      endpoint byte and rejects only non-finite; the exporter's input is an already
//      validated uint8 byte, so an out-of-range float there can only be a caller bug.
//      The extraction clamp policy is locked by tests/color_convergence_contract.cpp,
//      not here.
//   G4 a refused export creates no file at all (the mesh never reaches the writer);
//   G5 the same mesh exports byte-identically twice, and PLY keeps the uint8 sRGB byte.
//
// Filesystem use is the point of this contract (a real .glb round-trip); it stays
// under std::filesystem::temp_directory_path() and every file is removed.

#include "export/ColorConversion.h"
#include "export/GLBExporter.h"
#include "export/PLYExporter.h"
#include "meshing/MeshData.h"
#include "utils/ColorMath.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

// Same tinygltf configuration as the exporter TU, minus the implementation macros:
// the tinygltf definitions live in GLBExporter.cpp.o inside azu_test_core, so this TU
// only needs the declarations and the JSON header it forwards to.
#define TINYGLTF_NO_INCLUDE_JSON
#include "json.hpp"
#include "tiny_gltf.h"

namespace {

using kfusion::export_io::GLBExporter;
using kfusion::export_io::PLYExporter;
using kfusion::export_io::srgbColorToLinear;
using kfusion::export_io::srgbComponentToLinear;
using kfusion::meshing::MeshData;
using kfusion::utils::srgbFloatToUint8;
using kfusion::utils::srgbUint8ToFloat;

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, what)                                                       \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            ++g_failures;                                                       \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, std::string(what).c_str()); \
        }                                                                       \
    } while (false)

const float* nanF() {
    static const float v = std::numeric_limits<float>::quiet_NaN();
    return &v;
}
const float* infF() {
    static const float v = std::numeric_limits<float>::infinity();
    return &v;
}

constexpr double kTol = 1e-5;

// Independent oracle: the published sRGB EOTF, in double, written here so a change
// to the product helper cannot move the expectation with it.
double oracleLinear(uint8_t byte) {
    const double c = static_cast<double>(byte) / 255.0;
    return (c <= 0.04045) ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}

bool nearAbs(double got, double want, double tol = kTol) { return std::fabs(got - want) <= tol; }

// The fixture colors deliberately include the piecewise-knee neighbourhood: byte 10
// widens to 0.039216 (linear branch), byte 11 to 0.043137 (power branch).
const uint8_t kSwatch[][3] = {
    {  0,   0,   0 },   // black
    {255, 255, 255 },   // white
    { 10,  10,  10 },   // just below the knee
    { 11,  11,  11 },   // just above the knee
    {128, 128, 128 },   // mid gray, and the EMPTY_COLOR byte
    {200, 100,  40 },   // saturated
    { 11, 128, 244 },   // mixed channels
};
constexpr size_t kSwatchCount = sizeof kSwatch / sizeof kSwatch[0];

// One distinct, non-duplicated vertex per swatch, so the exporter's position weld
// cannot merge two swatches and the file order equals the input order.
MeshData makeSwatchMesh() {
    MeshData m;
    for (size_t i = 0; i < kSwatchCount; ++i) {
        const float z = -0.01f * static_cast<float>(i);
        m.positions.push_back(Eigen::Vector3f(0.01f * static_cast<float>(i), 0.02f, z));
        m.normals.push_back(Eigen::Vector3f(0.0f, 0.0f, 1.0f));
        m.colors.push_back(kSwatch[i][0]);
        m.colors.push_back(kSwatch[i][1]);
        m.colors.push_back(kSwatch[i][2]);
    }
    for (size_t i = 0; i + 2 < kSwatchCount; i += 1) {
        m.indices.push_back(static_cast<uint32_t>(i));
        m.indices.push_back(static_cast<uint32_t>(i + 1));
        m.indices.push_back(static_cast<uint32_t>(i + 2));
    }
    return m;
}

std::filesystem::path tempPath(const std::string& name) {
    return std::filesystem::temp_directory_path() / ("azu_t19_" + name);
}

struct ReadColor {
    bool        found   = false;
    int         compType = -1;
    int         accType  = -1;
    size_t      count    = 0;
    size_t      positionCount = 0;
    std::vector<float> values;   // interleaved RGBA as read from the file
};

ReadColor readColor0(const std::filesystem::path& path, std::string& err_out) {
    ReadColor out;
    tinygltf::TinyGLTF  loader;
    tinygltf::Model     model;
    std::string         warn;
    if (!loader.LoadBinaryFromFile(&model, &err_out, &warn, path.string())) {
        return out;
    }
    if (model.meshes.empty() || model.meshes[0].primitives.empty()) {
        err_out = "no mesh/primitive in the parsed glb";
        return out;
    }
    const tinygltf::Primitive& prim = model.meshes[0].primitives[0];
    const auto                 posIt = prim.attributes.find("POSITION");
    if (posIt != prim.attributes.end() && posIt->second < static_cast<int>(model.accessors.size())) {
        out.positionCount = static_cast<size_t>(model.accessors[posIt->second].count);
    }
    const auto colIt = prim.attributes.find("COLOR_0");
    if (colIt == prim.attributes.end()) {
        err_out = "primitive carries no COLOR_0 attribute";
        return out;
    }
    const tinygltf::Accessor& acc = model.accessors[colIt->second];
    out.found     = true;
    out.compType  = acc.componentType;
    out.accType   = acc.type;
    out.count     = static_cast<size_t>(acc.count);
    ReadColor&    c = out;

    const tinygltf::BufferView& bv  = model.bufferViews[acc.bufferView];
    const tinygltf::Buffer&     buf = model.buffers[bv.buffer];
    const size_t channels = (acc.type == TINYGLTF_TYPE_VEC4) ? 4u : 3u;
    const size_t stride   = channels * sizeof(float);
    const size_t base     = bv.byteOffset + static_cast<size_t>(acc.byteOffset);
    const size_t avail    = (base <= buf.data.size()) ? buf.data.size() - base : 0;
    const size_t span     = std::min(avail, std::min(bv.byteLength, out.count * stride));
    const uint8_t* src    = buf.data.data();
    // Byte-wise float reads: no unaligned reinterpret_cast over an offset buffer.
    for (size_t off = 0; off + stride <= span; off += stride) {
        float v[4] = {0.f, 0.f, 0.f, 1.f};
        for (size_t ch = 0; ch < channels; ++ch) {
            std::memcpy(&v[ch], src + base + off + ch * sizeof(float), sizeof(float));
        }
        c.values.push_back(v[0]);
        c.values.push_back(v[1]);
        c.values.push_back(v[2]);
        c.values.push_back(v[3]);
    }
    return out;
}
// ---------------------------------------------------------------------------
// G3 the checked decode refuses instead of sanitizing
// ---------------------------------------------------------------------------
void g3_rejection() {
    const float bad[] = {*nanF(), -*nanF(), *infF(), -*infF(), -0.001f, 1.001f, 2.0f,
                         std::numeric_limits<float>::max()};
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; ++i) {
        float lin = 0.25f;
        CHECK(!srgbComponentToLinear(bad[i], lin), "G3: a non-finite or out-of-range component is refused");
        CHECK(lin == 0.25f, "G3: a refused component leaves its output untouched");
    }

    // The triple fails as a unit: a bad green must not leave a decoded red behind.
    float r = 0.0f, g = 0.0f, b = 0.0f;
    CHECK(!srgbColorToLinear(0.5f, *nanF(), 0.5f, r, g, b), "G3: a non-finite channel rejects the triple");
    CHECK(r == 0.0f && g == 0.0f && b == 0.0f, "G3: no partial channel is written on rejection");
    CHECK(!srgbColorToLinear(0.5f, 1.2f, 0.5f, r, g, b), "G3: an out-of-range channel rejects the triple");
    CHECK(r == 0.0f && g == 0.0f && b == 0.0f, "G3: no partial channel is written on range rejection");

    float lr = 0.f, lg = 0.f, lb = 0.f;
    CHECK(srgbColorToLinear(0.0f, 0.5f, 1.0f, lr, lg, lb), "G3: in-range channels are accepted");
    std::printf("G3 rejected %zu components, triple rejected atomically\n", sizeof bad / sizeof bad[0]);
}

// ---------------------------------------------------------------------------
// G1 + G2 a real .glb carries linear float VEC4 COLOR_0
// ---------------------------------------------------------------------------
void g1g2_glb_round_trip() {
    const MeshData mesh = makeSwatchMesh();
    std::string    reason;
    CHECK(mesh.validate(&reason), "G1: the fixture mesh is exportable");

    const std::filesystem::path path = tempPath("swatch.glb");
    std::filesystem::remove(path);
    CHECK(GLBExporter::write(mesh, path.string()), "G1: the exporter accepts the fixture mesh");
    CHECK(std::filesystem::exists(path), "G1: the .glb exists on disk");

    std::string err;
    const ReadColor c = readColor0(path, err);
    CHECK(c.found, ("G1: COLOR_0 is present in the written file (" + err + ")").c_str());
    if (c.found) {
        CHECK(c.compType == TINYGLTF_COMPONENT_TYPE_FLOAT, "G2: COLOR_0 componentType is FLOAT");
        CHECK(c.accType == TINYGLTF_TYPE_VEC4, "G1: COLOR_0 accessor type is VEC4");
        CHECK(c.count == c.positionCount, "G1: COLOR_0 count equals the POSITION vertex count");
        CHECK(c.values.size() == c.count * 4, "G1: COLOR_0 holds four floats per vertex");

        bool alpha_one = true, linear_ok = true, raw_differs = true;
        for (size_t i = 0; i < c.count; ++i) {
            if (c.values[i * 4 + 3] != 1.0f) alpha_one = false;
            const uint8_t* sw = kSwatch[i % kSwatchCount];
            // The welded vertex order is the insertion order for distinct positions,
            // and the mesh has one vertex per swatch.
            for (int ch = 0; ch < 3; ++ch) {
                const double want = oracleLinear(sw[ch]);
                const double got  = static_cast<double>(c.values[i * 4 + ch]);
                if (!nearAbs(got, want)) linear_ok = false;
                // The pre-Todo-19 exporter wrote byte/255 (sRGB-encoded, float-typed
                // only in name). Any swatch that is not a fixed point of the EOTF
                // distinguishes the two; 0 and 255 are fixed points, so skip them.
                const double raw = static_cast<double>(sw[ch]) / 255.0;
                if (sw[ch] != 0 && sw[ch] != 255 && std::fabs(got - raw) < 1e-3) raw_differs = false;
            }
        }
        CHECK(alpha_one, "G1: every alpha is exactly 1.0 (alpha is never converted)");
        CHECK(linear_ok, "G2: every RGB channel matches the independent double EOTF oracle within 1e-5");
        CHECK(raw_differs, "G2: the file is NOT the raw byte/255 the old exporter emitted");

        // Knee evidence, printed rather than asserted twice.
        for (size_t i = 0; i < c.count && i < kSwatchCount; ++i) {
            std::printf("G2 srgb(%3u,%3u,%3u) -> linear(%.7f,%.7f,%.7f) a=%.1f  oracle(%.7f,%.7f,%.7f)\n",
                        kSwatch[i][0], kSwatch[i][1], kSwatch[i][2],
                        c.values[i * 4 + 0], c.values[i * 4 + 1], c.values[i * 4 + 2],
                        c.values[i * 4 + 3], oracleLinear(kSwatch[i][0]), oracleLinear(kSwatch[i][1]),
                        oracleLinear(kSwatch[i][2]));
        }
    }
    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// G4 a mesh the exporter must refuse never creates a file
// ---------------------------------------------------------------------------
void g4_refusal_writes_nothing() {
    const std::filesystem::path path = tempPath("refused.glb");
    std::filesystem::remove(path);

    MeshData half;              // one color triple short of one per vertex
    half.positions.push_back(Eigen::Vector3f(0.f, 0.f, 0.f));
    half.positions.push_back(Eigen::Vector3f(0.01f, 0.f, 0.f));
    half.positions.push_back(Eigen::Vector3f(0.f, 0.01f, 0.f));
    half.normals.assign(3, Eigen::Vector3f(0.f, 0.f, 1.f));
    half.colors.push_back(10); half.colors.push_back(20); half.colors.push_back(30);
    half.colors.push_back(40); half.colors.push_back(50);
    half.indices = {0, 1, 2};
    std::string why;
    CHECK(!half.validate(&why), "G4: the fixture mesh is invalid by construction");
    CHECK(!GLBExporter::write(half, path.string()), "G4: the exporter refuses a half-filled color buffer");
    CHECK(!std::filesystem::exists(path), "G4: a refused export creates no file");

    MeshData nanPos = makeSwatchMesh();   // non-finite position, same refusal guarantee
    nanPos.positions[0] = Eigen::Vector3f(*nanF(), 0.f, 0.f);
    CHECK(!GLBExporter::write(nanPos, path.string()), "G4: the exporter refuses a non-finite position");
    CHECK(!std::filesystem::exists(path), "G4: still no file after the second refusal");

    // The byte domain the exporter decodes from can never itself be out of range, so
    // the decode refusal is exercised at helper level (G3); what is provable here is
    // that nothing reaches the writer when the mesh contract fails.
    std::filesystem::remove(path);
}

// ---------------------------------------------------------------------------
// G5 determinism + the PLY boundary keeps uint8 sRGB
// ---------------------------------------------------------------------------
void g5_determinism_and_ply() {
    const MeshData mesh = makeSwatchMesh();
    const auto     a    = tempPath("det_a.glb");
    const auto     b    = tempPath("det_b.glb");
    std::filesystem::remove(a);
    std::filesystem::remove(b);
    CHECK(GLBExporter::write(mesh, a.string()), "G5: first export succeeds");
    CHECK(GLBExporter::write(mesh, b.string()), "G5: second export succeeds");

    auto slurp = [](const std::filesystem::path& p) {
        std::ifstream in(p, std::ios::binary);
        return std::vector<char>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    };
    const std::vector<char> ba = slurp(a), bb = slurp(b);
    CHECK(!ba.empty() && ba == bb, "G5: the same mesh exports byte-identically twice");
    std::printf("G5 glb bytes=%zu identical=%d\n", ba.size(), static_cast<int>(ba == bb));
    std::filesystem::remove(a);
    std::filesystem::remove(b);

    // PLY stays the byte domain: no EOTF happens on that path, so the bytes a mesh
    // carries are the bytes a reader sees.
    const std::filesystem::path ply = tempPath("swatch.ply");
    std::filesystem::remove(ply);
    CHECK(kfusion::export_io::PLYExporter::writeASCII(mesh, ply.string()), "G5: the PLY export succeeds");
    std::ifstream in(ply, std::ios::binary);
    std::string   body((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    bool          bytes_present = true;
    for (size_t i = 0; i < kSwatchCount && bytes_present; ++i) {
        const std::string triple = std::to_string(kSwatch[i][0]) + " " + std::to_string(kSwatch[i][1]) + " " +
                                   std::to_string(kSwatch[i][2]);
        if (body.find(triple) == std::string::npos) bytes_present = false;
    }
    CHECK(bytes_present, "G5: PLY carries the exact uint8 sRGB triples (no gamma applied)");
    std::filesystem::remove(ply);
}

}  // namespace

int main() {
    g3_rejection();
    g1g2_glb_round_trip();
    g4_refusal_writes_nothing();
    g5_determinism_and_ply();

    if (g_failures != 0) {
        std::printf("glb_linear_color_contract: %d FAILED of %d checks\n", g_failures, g_checks);
        return 1;
    }
    std::printf("glb_linear_color_contract: all %d checks passed\n", g_checks);
    return 0;
}
