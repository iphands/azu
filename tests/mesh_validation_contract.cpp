// mesh_validation_contract (big-fix todo 10): CPU-only contract for
// MeshData::validate() and for the three public export entry points, which must
// reject invalid CPU mesh data before writing anything. Public API only; the one
// filesystem surface is a process-unique subdirectory of the system temp
// directory, removed before the process exits. No device, display, GPU, sensor,
// thread timing or network access.

#include "export/GLBExporter.h"
#include "export/PLYExporter.h"
#include "meshing/MeshData.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <vector>
#include <unistd.h>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "mesh_validation_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

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

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();

// Valid, exportable, bare triangle: positions + one index triple, no normals, no
// colors (so it doubles as the zero-normals / zero-colors acceptance case).
MeshData bareMesh() {
    MeshData m;
    m.positions = {
        Eigen::Vector3f(0.0f, 0.0f, 0.5f),
        Eigen::Vector3f(1.0f, 0.0f, 0.5f),
        Eigen::Vector3f(0.0f, 1.0f, 0.5f),
    };
    m.indices = {0, 1, 2};
    return m;
}

void addAllNormals(MeshData& m) {
    m.normals.assign(m.positions.size(), Eigen::Vector3f(0.0f, 0.0f, 1.0f));
}

void addAllColors(MeshData& m) {
    m.colors.assign(m.positions.size() * 3, 0);
    for (size_t i = 0; i < m.positions.size(); ++i) m.colors[i * 3] = 255;
}

struct Case {
    const char* name;
    bool expect_valid;
    // Each entry must appear in the reported reason (unused when valid).
    std::vector<std::string> reason_parts;
    std::function<void(MeshData&)> mutate;
};

const Case kCases[] = {
    {"bare triangle (no normals, no colors)", true, {}, [](MeshData&) {}},
    {"all normals",              true, {}, [](MeshData& m) { addAllNormals(m); }},
    {"all colors",               true, {}, [](MeshData& m) { addAllColors(m); }},
    {"all normals + all colors", true, {},
     [](MeshData& m) { addAllNormals(m); addAllColors(m); }},

    {"default-constructed mesh", false, {"positions is empty"},
     [](MeshData& m) { m.clear(); }},
    {"empty positions",          false, {"positions is empty"},
     [](MeshData& m) { m.positions.clear(); }},
    {"empty indices",            false, {"indices is empty", "positions.size()=3"},
     [](MeshData& m) { m.indices.clear(); }},
    {"index count not a multiple of 3", false, {"indices.size()=4", "not a multiple of 3"},
     [](MeshData& m) { m.indices.push_back(0); }},
    {"index equal to positions.size()", false, {"indices[2]=3", "out of range", "positions.size()=3"},
     [](MeshData& m) { m.indices[2] = 3; }},
    {"index UINT32_MAX",         false, {"indices[0]=4294967295", "out of range"},
     [](MeshData& m) { m.indices[0] = UINT32_MAX; }},
    {"NaN position",             false, {"positions[1]", "non-finite"},
     [](MeshData& m) { m.positions[1] = Eigen::Vector3f(kNaN, 0.0f, 0.5f); }},
    {"Inf position",             false, {"positions[2]", "non-finite"},
     [](MeshData& m) { m.positions[2] = Eigen::Vector3f(0.0f, 1.0f, kInf); }},
    {"partial normals (1 of 3)", false, {"normals.size()=1", "neither 0 nor positions.size()=3"},
     [](MeshData& m) { m.normals.push_back(Eigen::Vector3f(0.0f, 0.0f, 1.0f)); }},
    {"normals exceed positions", false, {"normals.size()=4", "neither 0 nor positions.size()=3"},
     [](MeshData& m) { addAllNormals(m); m.normals.push_back(Eigen::Vector3f::Zero()); }},
    {"Inf normal",               false, {"normals[1]", "non-finite"},
     [](MeshData& m) {
         addAllNormals(m);
         m.normals[1] = Eigen::Vector3f(0.0f, kInf, 1.0f);
     }},
    {"partial colors (2 of 3 vertices)", false, {"colors.size()=6", "neither 0 nor 3 * positions.size()"},
     [](MeshData& m) { addAllColors(m); m.colors.resize(6); }},
    {"colors not a multiple of 3", false, {"colors.size()=7", "neither 0 nor 3 * positions.size()"},
     [](MeshData& m) { addAllColors(m); m.colors.resize(7); }},
};

MeshData buildCase(const Case& c) {
    MeshData m = bareMesh();
    c.mutate(m);
    return m;
}

size_t defectCaseCount() {
    size_t n = 0;
    for (const Case& c : kCases) {
        if (!c.expect_valid) ++n;
    }
    return n;
}

void testValidateContract() {
    for (const Case& c : kCases) {
        const MeshData m = buildCase(c);
        const std::string who(c.name);

        std::string reason;
        const bool ok = m.validate(&reason);
        CHECK(ok == c.expect_valid, who + (ok ? ": accepted" : ": rejected"));
        std::printf("  %-36s %-8s %s\n", c.name, ok ? "accept" : "reject", reason.c_str());

        if (c.expect_valid) {
            CHECK(reason.empty(), who + ": accepted mesh reports no reason");
        } else {
            CHECK(!reason.empty(), who + ": rejection carries a non-empty reason");
            for (const std::string& part : c.reason_parts) {
                CHECK(reason.find(part) != std::string::npos, who + ": reason names '" + part + "'");
            }
        }
        std::string again;
        CHECK(m.validate(&again) == ok, who + ": validate() is repeatable");
        CHECK(m.validate(nullptr) == ok, who + ": validate(nullptr) is safe");
    }
}

// --- export entry points -----------------------------------------------------

// Every public export surface, with the magic a written file must carry.
struct Entry {
    const char* label;
    std::function<bool(const MeshData&, const std::string&)> write;
    const char* magic;
    const char* ext;
    bool check_glb_envelope;
};

const Entry kEntries[] = {
    {"PLYExporter::writeBinary", &kfusion::export_io::PLYExporter::writeBinary, "ply\n", ".ply", false},
    {"PLYExporter::writeASCII",  &kfusion::export_io::PLYExporter::writeASCII, "ply\n", ".ply", false},
    {"GLBExporter::write",       &kfusion::export_io::GLBExporter::write, "glTF", ".glb", true},
};
constexpr size_t kEntryCount = sizeof(kEntries) / sizeof(Entry);

const std::filesystem::path& scratchDir() {
    static const std::filesystem::path dir = std::filesystem::temp_directory_path() /
                                            ("azu_mesh_validation_contract_" + std::to_string(::getpid()));
    return dir;
}

bool hasContent(const std::filesystem::path& path) {
    std::error_code ec;
    const std::uintmax_t bytes = std::filesystem::file_size(path, ec);
    return !ec && bytes > 0;
}

bool startsWith(const std::filesystem::path& path, const std::string& prefix) {
    std::ifstream f(path, std::ios::binary);
    std::vector<char> buf(prefix.size(), 0);
    return static_cast<bool>(f.read(buf.data(), static_cast<std::streamsize>(prefix.size()))) &&
           std::memcmp(buf.data(), prefix.data(), prefix.size()) == 0;
}

// GLB header: "glTF" magic plus a total-length field equal to the file size.
bool hasGlbEnvelope(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    char header[12] = {0};
    if (!f.read(header, 12)) return false;
    uint32_t total = 0;
    std::memcpy(&total, header + 8, sizeof(total));
    return std::memcmp(header, "glTF", 4) == 0 && total == static_cast<uint32_t>(std::filesystem::file_size(path));
}

void testExportContract() {
    const MeshData valid = [] {
        MeshData m = bareMesh();
        addAllNormals(m);
        addAllColors(m);
        return m;
    }();

    for (size_t e = 0; e < kEntryCount; ++e) {
        const Entry& entry = kEntries[e];
        const std::string who(entry.label);
        const std::filesystem::path path = scratchDir() / ("accept" + std::to_string(e) + entry.ext);
        std::error_code ec;
        std::filesystem::remove(path, ec);
        CHECK(!std::filesystem::exists(path), who + ": target is free before export");
        CHECK(entry.write(valid, path.string()), who + ": valid mesh is exported");
        CHECK(std::filesystem::exists(path), who + ": file created");
        CHECK(hasContent(path), who + ": file is non-empty");
        CHECK(startsWith(path, entry.magic), who + ": file starts with the format magic");
        if (entry.check_glb_envelope) CHECK(hasGlbEnvelope(path), who + ": GLB magic + total length match the file");
        const std::uintmax_t bytes = hasContent(path) ? std::filesystem::file_size(path) : 0;
        std::filesystem::remove(path, ec);
        CHECK(!std::filesystem::exists(path), who + ": scratch file cleaned up");
        std::printf("  %-36s exported %llu bytes\n", entry.label,
                    static_cast<unsigned long long>(bytes));
    }

    size_t defects = 0;
    for (const Case& c : kCases) {
        if (c.expect_valid) continue;
        const MeshData bad = buildCase(c);
        ++defects;
        for (size_t e = 0; e < kEntryCount; ++e) {
            const Entry& entry = kEntries[e];
            const std::string who = std::string(c.name) + " / " + entry.label;
            const std::filesystem::path path =
                scratchDir() / ("defect" + std::to_string(defects) + "_" + std::to_string(e) + entry.ext);
            std::error_code ec;
            std::filesystem::remove(path, ec);
            CHECK(!std::filesystem::exists(path), who + ": target is free before export");
            CHECK(!entry.write(bad, path.string()), who + ": invalid mesh is rejected");
            CHECK(!std::filesystem::exists(path), who + ": rejected export leaves no file behind");
        }
    }
    CHECK(defects == defectCaseCount(), "every defect case reached every export entry point");
    std::printf("  %zu defect cases x %zu entry points rejected without writing\n", defects, kEntryCount);

    std::error_code ec;
    std::filesystem::remove_all(scratchDir(), ec);
    CHECK(!ec, "scratch directory cleanup reports no error");
    CHECK(!std::filesystem::exists(scratchDir()), "scratch directory removed");
}

} // namespace

int main() {
    std::error_code ec;
    std::filesystem::remove_all(scratchDir(), ec);
    std::filesystem::create_directories(scratchDir(), ec);
    if (ec) {
        std::printf("mesh_validation_contract: FAIL (cannot create %s: %s)\n",
                    scratchDir().string().c_str(), ec.message().c_str());
        return 1;
    }

    testValidateContract();
    testExportContract();

    if (g_failures == 0) {
        std::printf("mesh_validation_contract: PASS (%zu validate cases, %d checks)\n",
                    sizeof(kCases) / sizeof(Case), g_checks);
        return 0;
    }
    std::printf("mesh_validation_contract: FAIL (%d failed checks of %d)\n", g_failures, g_checks);
    return 1;
}
