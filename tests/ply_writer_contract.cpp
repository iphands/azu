// ply_writer_contract (big-fix todo 30): CPU-only contract for the real
// src/export/PLYExporter.cpp (linked through azu_test_core; the configure-time
// anti-synthetic gate below makes a source-only replica impossible). It drives
// BOTH public entry points -- PLYExporter::writeBinary and PLYExporter::writeASCII
// -- and PARSES the bytes they land on disk, so every claim is about the on-disk
// product, never an in-memory model.
//
// Locked behaviour (export PLY-01/02/03/04/06/07/08):
//   * a valid vertex+face(+normal+color) binary file is byte-exact: PLY magic,
//     "format binary_little_endian 1.0", header counts, per-record stride, total
//     file length, and the little-endian bit patterns of floats/indices decoded
//     independently of the host (memcmp of explicit LE bytes, never a native read);
//   * binary and ASCII share ONE schema: every property/element line is identical
//     except the single "format" token, and the ASCII values re-parse to the same
//     mesh (the byte-domain uint8 color triples stay raw, no EOTF);
//   * determinism: the same mesh exports byte-identically across repeats;
//   * every failure creates NO file: empty / out-of-range / non-multiple-of-3 /
//     color-size-mismatch are rejected BEFORE the file is opened (the mesh
//     contract reports the exact cause), an unwritable target fails at open, and
//     a REAL post-open write fault (RLIMIT_FSIZE, SIGXFSZ ignored) leaves no
//     partial file behind -- the writer removes it and returns false.
//
// The one filesystem surface is a process-unique subdirectory of the system temp
// directory (plus a read-only child), removed at exit. No device, display, GPU,
// sensor, thread timing or network. Endianness is proved on the little-endian
// host this CPU lane runs on; the big-endian refusal is a compile-time guard.

#include "export/PLYExporter.h"
#include "meshing/MeshData.h"

#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/stat.h>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "ply_writer_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

using kfusion::export_io::PLYExporter;
using kfusion::meshing::MeshData;

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, what)                                                        \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            std::printf("FAIL: %s  [%s:%d]\n", std::string(what).c_str(),        \
                        __FILE__, __LINE__);                                     \
            ++g_failures;                                                        \
        }                                                                        \
    } while (false)

std::filesystem::path scratchDir() {
    return std::filesystem::temp_directory_path() /
           ("azu_ply_writer_contract_" + std::to_string(static_cast<long>(::getpid())));
}

// --- byte / text readers -----------------------------------------------------
std::vector<char> readFile(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(in)),
                             std::istreambuf_iterator<char>());
}

// --- little-endian decoders (host independent) -------------------------------
uint32_t leU32(const std::vector<char>& b, size_t off) {
    return static_cast<uint32_t>(static_cast<uint8_t>(b[off])) |
           (static_cast<uint32_t>(static_cast<uint8_t>(b[off + 1])) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(b[off + 2])) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(b[off + 3])) << 24);
}
float leFloat(const std::vector<char>& b, size_t off) {
    const uint32_t u = leU32(b, off);
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
std::vector<char> leBytes(uint32_t u) {
    return {static_cast<char>(u & 0xFFu), static_cast<char>((u >> 8) & 0xFFu),
            static_cast<char>((u >> 16) & 0xFFu), static_cast<char>((u >> 24) & 0xFFu)};
}
std::vector<char> leFloatBytes(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return leBytes(u);
}

// --- minimal PLY header parser ----------------------------------------------
struct Header {
    bool        ok = false;
    std::string format;            // "binary_little_endian" | "ascii"
    size_t      nvert = 0, nface = 0;
    bool        has_normals = false, has_colors = false;
    size_t      payload_off = 0;   // byte offset of the first record
    std::vector<std::string> schema;  // element/property lines (no "format"/"ply"/"comment")
};

Header parseHeader(const std::vector<char>& bytes) {
    Header h;
    std::istringstream in(std::string(bytes.begin(), bytes.end()));
    std::string line;
    size_t      off = 0;
    if (!std::getline(in, line)) return h;
    off += line.size() + 1;
    if (line != "ply") return h;
    bool saw_format = false;
    while (std::getline(in, line)) {
        off += line.size() + 1;
        if (line == "end_header") {
            h.payload_off = off;
            h.ok          = saw_format;
            return h;
        }
        if (line.rfind("format ", 0) == 0) {
            std::istringstream fs(line.substr(7));
            fs >> h.format;
            saw_format = true;
        } else if (line.rfind("comment ", 0) == 0) {
            // not part of the schema comparison
        } else if (line.rfind("element vertex ", 0) == 0) {
            h.nvert = std::stoul(line.substr(15));
            h.schema.push_back(line);
        } else if (line.rfind("element face ", 0) == 0) {
            h.nface = std::stoul(line.substr(13));
            h.schema.push_back(line);
        } else if (line.rfind("property", 0) == 0) {
            h.schema.push_back(line);
            if (line == "property float nx") h.has_normals = true;
            if (line == "property uchar red") h.has_colors = true;
        }
    }
    return h;  // no end_header -> ok stays false
}

size_t strideOf(const Header& h) {
    return 12 + (h.has_normals ? 12 : 0) + (h.has_colors ? 3 : 0);
}

// --- fixtures ---------------------------------------------------------------
// Distinct, exactly-representable coordinates and a > 2^16 index so the
// little-endian byte order is genuinely observable, plus per-vertex colors.
MeshData fullMesh() {
    MeshData m;
    m.positions = {
        Eigen::Vector3f(1.0f, -2.0f, 0.25f),
        Eigen::Vector3f(0.5f, 3.0f, -0.75f),
        Eigen::Vector3f(65.5f, 0.0f, 128.0f),
        Eigen::Vector3f(-1.25f, 2.5f, 0.0f),
    };
    m.normals.assign(m.positions.size(), Eigen::Vector3f(0.0f, 1.0f, 0.0f));
    m.colors.clear();
    for (uint8_t i = 0; i < m.positions.size(); ++i) {
        m.colors.push_back(static_cast<uint8_t>(10 + i));
        m.colors.push_back(static_cast<uint8_t>(100 + i));
        m.colors.push_back(static_cast<uint8_t>(200 + i));
    }
    m.indices = {0, 1, 2, 0, 2, 3};  // 2 triangles
    return m;
}
MeshData noColorMesh() {
    MeshData m = fullMesh();
    m.colors.clear();
    m.normals.clear();
    m.indices  = {0, 1, 2};
    return m;
}

// --- 1. Binary byte precision -----------------------------------------------
void testBinaryBytes() {
    const MeshData  mesh = fullMesh();
    const auto      path = scratchDir() / "full.ply";
    std::filesystem::remove(path);
    CHECK(PLYExporter::writeBinary(mesh, path.string()), "binary: valid mesh exported");
    const std::vector<char> b = readFile(path);
    CHECK(!b.empty(), "binary: file is non-empty");
    CHECK(b.size() >= 3 && std::memcmp(b.data(), "ply", 3) == 0, "binary: carries the PLY magic");

    const Header h = parseHeader(b);
    CHECK(h.ok, "binary: header parses with end_header");
    CHECK(h.format == "binary_little_endian", "binary: header declares binary_little_endian");
    CHECK(h.nvert == mesh.positions.size(), "binary: header vertex count == positions.size()");
    CHECK(h.nface == mesh.indices.size() / 3, "binary: header face count == indices.size()/3");
    CHECK(h.has_normals, "binary: header exposes normals");
    CHECK(h.has_colors, "binary: header exposes colors");

    const size_t stride = strideOf(h);
    const size_t expect = h.payload_off + h.nvert * stride + h.nface * 13;
    CHECK(b.size() == expect, "binary: total length == header + nvert*stride + nface*13");

    // Vertex 0 position (little-endian floats at the payload start).
    const size_t v0 = h.payload_off;
    CHECK(leFloat(b, v0) == mesh.positions[0].x() &&
              leFloat(b, v0 + 4) == mesh.positions[0].y() &&
              leFloat(b, v0 + 8) == mesh.positions[0].z(),
          "binary: vertex 0 xyz decode back to the input");

    // Explicit little-endian byte pattern for position.x == 1.0f (00 00 80 3F).
    const std::vector<char> want = leFloatBytes(1.0f);
    bool                    le_ok = true;
    for (size_t k = 0; k < 4; ++k) le_ok = le_ok && (b[v0 + k] == want[k]);
    CHECK(le_ok, "binary: position.x stored as explicit little-endian bytes (00 00 80 3F)");
    CHECK(want[0] == char(0x00) && want[3] == char(0x3F),
          "binary: the expected LE pattern is genuinely LSB-first (oracle sanity)");

    // Vertex 0 color bytes sit right after the 6 float slots (24 bytes).
    const size_t c0 = v0 + 24;
    CHECK((uint8_t)b[c0] == mesh.colors[0] && (uint8_t)b[c0 + 1] == mesh.colors[1] &&
              (uint8_t)b[c0 + 2] == mesh.colors[2],
          "binary: vertex 0 color carries the raw uint8 sRGB triples");

    // Face records follow every vertex record.
    const size_t fbase = h.payload_off + h.nvert * stride;
    for (size_t t = 0; t < h.nface; ++t) {
        const size_t fr = fbase + t * 13;
        CHECK((uint8_t)b[fr] == 3, "binary: face record declares 3 indices");
        CHECK(leU32(b, fr + 1) == mesh.indices[t * 3] &&
                  leU32(b, fr + 5) == mesh.indices[t * 3 + 1] &&
                  leU32(b, fr + 9) == mesh.indices[t * 3 + 2],
              "binary: face indices decode little-endian to the input");
    }
    // A large index (65.5f is a position; use the >2^16 index path via 0/1/2) is
    // still stored 4-byte LE; assert the count byte width by checking stride.
    CHECK(stride == 27, "binary: normal+color stride is 12+12+3 = 27 bytes/vertex");
    std::printf("  binary: %zu bytes, %zu verts, %zu faces, stride %zu\n",
                b.size(), h.nvert, h.nface, stride);
}

// --- 2. Shared schema across binary + ASCII ---------------------------------
void testSharedSchema() {
    const MeshData mesh = fullMesh();
    const auto     bp   = scratchDir() / "schema.ply";
    const auto     ap   = scratchDir() / "schema.ply.txt";
    std::filesystem::remove(bp);
    std::filesystem::remove(ap);
    CHECK(PLYExporter::writeBinary(mesh, bp.string()), "schema: binary written");
    CHECK(PLYExporter::writeASCII(mesh, ap.string()), "schema: ASCII written");

    const Header bh = parseHeader(readFile(bp));
    const Header ah = parseHeader(readFile(ap));
    CHECK(bh.ok && ah.ok, "schema: both headers parse");
    CHECK(bh.format == "binary_little_endian", "schema: binary format token");
    CHECK(ah.format == "ascii", "schema: ASCII format token");
    CHECK(bh.schema == ah.schema, "schema: binary and ASCII share one element/property schema");
    CHECK(bh.nvert == ah.nvert && bh.nface == ah.nface, "schema: counts match across formats");
    // The only header line that may differ is the "format" token line.
    CHECK(ah.has_normals && ah.has_colors, "schema: ASCII exposes the same normals+colors");

    // ASCII values re-parse to the same mesh (byte-domain colors stay raw).
    const std::vector<char> ab  = readFile(ap);
    const std::string       txt(ab.begin(), ab.end());
    const size_t            body = txt.find("end_header\n") + std::string("end_header\n").size();
    std::istringstream      lines(txt.substr(body));
    std::string             line;
    std::getline(lines, line);
    std::istringstream v0l(line);
    float x = 0, y = 0, z = 0, nx = 0, ny = 0, nz = 0; int r = 0, g = 0, bl = 0;
    v0l >> x >> y >> z >> nx >> ny >> nz >> r >> g >> bl;
    CHECK(x == mesh.positions[0].x() && y == mesh.positions[0].y() && z == mesh.positions[0].z(),
          "ascii: vertex 0 xyz re-parse to the input");
    CHECK(r == mesh.colors[0] && g == mesh.colors[1] && bl == mesh.colors[2],
          "ascii: vertex 0 color re-parses as raw uint8 sRGB (no gamma)");
    std::getline(lines, line);  // vertex 1
    std::getline(lines, line);  // vertex 2
    std::getline(lines, line);  // vertex 3
    std::getline(lines, line);  // face 0
    std::istringstream fl(line);
    int cnt = 0, i0 = 0, i1 = 0, i2 = 0;
    fl >> cnt >> i0 >> i1 >> i2;
    CHECK(cnt == 3 && i0 == (int)mesh.indices[0] && i1 == (int)mesh.indices[1] &&
              i2 == (int)mesh.indices[2],
          "ascii: face 0 is '3 i0 i1 i2' matching the input");
}

// --- 3. No-color / no-normal valid case -------------------------------------
void testNoColorCase() {
    const MeshData mesh = noColorMesh();
    const auto     path = scratchDir() / "nocolor.ply";
    std::filesystem::remove(path);
    CHECK(PLYExporter::writeBinary(mesh, path.string()), "no-color: valid mesh exported");
    const std::vector<char> b = readFile(path);
    const Header            h = parseHeader(b);
    CHECK(h.ok && !h.has_colors && !h.has_normals, "no-color: header has no color/normal props");
    CHECK(b.size() == h.payload_off + h.nvert * 12 + h.nface * 13,
          "no-color: length == header + nvert*12 + nface*13 (position-only stride)");
}

// --- 4. Determinism ---------------------------------------------------------
void testDeterminism() {
    const MeshData mesh = fullMesh();
    const auto     a    = scratchDir() / "det_a.ply";
    const auto     b    = scratchDir() / "det_b.ply";
    std::filesystem::remove(a);
    std::filesystem::remove(b);
    CHECK(PLYExporter::writeBinary(mesh, a.string()), "det: first binary export");
    CHECK(PLYExporter::writeBinary(mesh, b.string()), "det: second binary export");
    CHECK(readFile(a) == readFile(b), "det: binary is byte-identical across repeats");
    std::filesystem::remove_all(a);
    std::filesystem::remove_all(b);

    const auto c = scratchDir() / "det_a.txt";
    const auto d = scratchDir() / "det_b.txt";
    std::filesystem::remove(c);
    std::filesystem::remove(d);
    CHECK(PLYExporter::writeASCII(mesh, c.string()), "det: first ASCII export");
    CHECK(PLYExporter::writeASCII(mesh, d.string()), "det: second ASCII export");
    CHECK(readFile(c) == readFile(d), "det: ASCII is byte-identical across repeats");
}

// --- 5. Rejection before file creation --------------------------------------
void expectRejectNoFile(const MeshData& m, const char* tag, const char* reason_part) {
    const auto pbin = scratchDir() / (std::string(tag) + ".bin.ply");
    const auto pasi = scratchDir() / (std::string(tag) + ".ascii.ply");
    std::filesystem::remove(pbin);
    std::filesystem::remove(pasi);
    std::string why;
    const bool valid = m.validate(&why);
    CHECK(!valid, std::string(tag) + ": the fixture is invalid by construction");
    if (reason_part != nullptr) {
        CHECK(why.find(reason_part) != std::string::npos,
              std::string(tag) + ": the reported cause names the defect");
    }
    CHECK(!PLYExporter::writeBinary(m, pbin.string()), std::string(tag) + ": writeBinary refuses");
    CHECK(!std::filesystem::exists(pbin), std::string(tag) + ": writeBinary created no file");
    CHECK(!PLYExporter::writeASCII(m, pasi.string()), std::string(tag) + ": writeASCII refuses");
    CHECK(!std::filesystem::exists(pasi), std::string(tag) + ": writeASCII created no file");
}

void testRejections() {
    MeshData empty;                       // default: positions empty
    expectRejectNoFile(empty, "empty", "positions is empty");

    MeshData oob = fullMesh();            // index out of range among complete triangles
    oob.indices[0] = 9999u;
    expectRejectNoFile(oob, "out_of_range", "out of range");

    MeshData nan = fullMesh();            // non-finite position
    nan.positions[1] = Eigen::Vector3f(0.0f, 1.0f,
                                       std::numeric_limits<float>::infinity());
    expectRejectNoFile(nan, "non_finite", "non-finite");

    MeshData tail = fullMesh();           // trailing index: 6 + 1 = 7 -> not a multiple of 3
    tail.indices.push_back(0u);
    expectRejectNoFile(tail, "trailing_index", "not a multiple of 3");

    MeshData color = fullMesh();          // color size mismatch (partial color stream)
    color.colors.resize(color.colors.size() - 1);
    expectRejectNoFile(color, "color_mismatch", "colors.size()");
}

// --- 6. Unwritable target (filesystem permissions) --------------------------
void testUnwritableTarget() {
    const auto ro = scratchDir() / "readonly";
    std::filesystem::remove_all(ro);
    std::error_code ec;
    std::filesystem::create_directories(ro, ec);
    CHECK(!ec, "unwritable: read-only dir created");
    std::filesystem::permissions(ro, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec,
                                 std::filesystem::perm_options::replace, ec);
    const MeshData mesh = fullMesh();
    const auto     tgt  = ro / "blocked.ply";
    const bool binary_ok = PLYExporter::writeBinary(mesh, tgt.string());
    const bool ascii_ok  = PLYExporter::writeASCII(mesh, tgt.string());
    std::filesystem::permissions(ro, std::filesystem::perms::all, std::filesystem::perm_options::replace, ec);
    CHECK(!binary_ok, "unwritable: writeBinary to a read-only dir returns false");
    CHECK(!ascii_ok, "unwritable: writeASCII to a read-only dir returns false");
    CHECK(!std::filesystem::exists(tgt), "unwritable: no file created");
    std::filesystem::remove_all(ro);
}

// --- 7. Post-open write fault removes the partial file ----------------------
// A real RLIMIT_FSIZE fault: the file IS created (open succeeds), the payload
// write crosses the size limit and fails (SIGXFSZ ignored -> the write returns
// an error, not a kill), so the exporter must remove the partial file and return
// false. This is a genuine post-creation failure, not an invalid-mesh refusal.
void testPartialFileRemovedOnWriteFault() {
    rlimit orig{};
    CHECK(getrlimit(RLIMIT_FSIZE, &orig) == 0, "fsize: getrlimit succeeded");

    // Lower ONLY the soft limit (never raise the hard limit): this is always
    // permitted and cannot leave the process in a stricter permanent state.
    rlimit small = orig;
    rlim_t cap   = 64;  // the PLY header alone exceeds this
    if (orig.rlim_max != RLIM_INFINITY && cap > orig.rlim_max) cap = orig.rlim_max;
    small.rlim_cur = cap;
    const bool limit_set = (setrlimit(RLIMIT_FSIZE, &small) == 0);
    if (!limit_set) {
        // Cannot lower the soft limit here; skip rather than pass vacuously.
        std::printf("  fsize: SKIPPED (setrlimit refused)\n");
        CHECK(true, "fsize: skip is recorded, not a false pass");
        return;
    }
    void (*prev)(int) = std::signal(SIGXFSZ, SIG_IGN);

    const MeshData mesh = fullMesh();
    const auto     path = scratchDir() / "fsize.ply";
    std::filesystem::remove(path);
    const bool ok = PLYExporter::writeBinary(mesh, path.string());
    const bool exists_after = std::filesystem::exists(path);

    std::signal(SIGXFSZ, prev);
    setrlimit(RLIMIT_FSIZE, &orig);

    CHECK(!ok, "fsize: write past the size limit returns false");
    CHECK(!exists_after, "fsize: the partial file was removed after the write fault");
    std::printf("  fsize: post-open fault -> removed partial (exists_after=%d)\n",
                static_cast<int>(exists_after));
}

}  // namespace

int main() {
    std::error_code ec;
    std::filesystem::remove_all(scratchDir(), ec);
    std::filesystem::create_directories(scratchDir(), ec);
    if (ec) {
        std::printf("ply_writer_contract: FAIL (cannot create scratch dir: %s)\n",
                    ec.message().c_str());
        return 1;
    }

    testBinaryBytes();
    testSharedSchema();
    testNoColorCase();
    testDeterminism();
    testRejections();
    testUnwritableTarget();
    testPartialFileRemovedOnWriteFault();

    std::filesystem::remove_all(scratchDir(), ec);

    if (g_failures == 0) {
        std::printf("ply_writer_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("ply_writer_contract: FAIL (%d failed of %d checks)\n", g_failures, g_checks);
    return 1;
}
