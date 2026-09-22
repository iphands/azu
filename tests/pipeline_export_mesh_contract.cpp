// pipeline_export_mesh_contract (big-fix todo 26): CPU-only contract for the
// shared PipelineController::exportMesh(path, writer_fn) helper that both
// exportPLY() and exportGLB() now route through. It runs against the REAL
// src/app/PipelineController.cpp (link: azu_test_pipeline) with the controller
// NEVER started, so every section is deterministic: no worker thread exists,
// no device, no display, no GPU, no network. The one bounded wait is the
// empty-mesh section's documented 5 s awaitMeshVersion timeout (there is no
// meshing worker to answer the request). The filesystem surface is a
// process-unique subdirectory of the system temp directory, removed at exit.
//
// Locked behaviour:
//   * empty shared mesh -> exportMesh returns false, writer_fn is NEVER
//     invoked, and the target file is never created;
//   * a published mesh reaches writer_fn as the SAME MeshData object with the
//     exact path, and a true writer yields a true export (export_pct 100);
//   * a false writer -> false export (export_pct 0); a throwing writer (both
//     std::exception and unknown) is a failed export, never a crash;
//   * exportPLY() and exportGLB() are thin wrappers over the same helper:
//     their on-disk bytes are byte-identical to calling PLYExporter /
//     GLBExporter directly on the same mesh, and the files carry the PLY /
//     glTF magics.

#include "app/PipelineController.h"
#include "export/GLBExporter.h"
#include "export/PLYExporter.h"
#include "meshing/MeshData.h"

#include <QCoreApplication>

#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "pipeline_export_mesh_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

using kfusion::app::PipelineController;
using kfusion::meshing::MeshData;
using kfusion::sensor::PreprocessBackend;

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

// Valid, exportable bare triangle (same shape mesh_validation_contract uses
// as its accepted case), plus normals/colors so the GLB path sees full data.
MeshData validMesh() {
    MeshData m;
    m.positions = {
        Eigen::Vector3f(0.0f, 0.0f, 0.5f),
        Eigen::Vector3f(1.0f, 0.0f, 0.5f),
        Eigen::Vector3f(0.0f, 1.0f, 0.5f),
    };
    m.indices = {0, 1, 2};
    m.normals.assign(m.positions.size(), Eigen::Vector3f(0.0f, 0.0f, 1.0f));
    m.colors.assign(m.positions.size() * 3, 0);
    for (size_t i = 0; i < m.positions.size(); ++i) m.colors[i * 3] = 255;
    return m;
}

bool fileBytes(const std::filesystem::path& p, std::vector<char>& out) {
    std::ifstream in(p, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return !in.bad();
}

bool startsWith(const std::vector<char>& b, const char* magic, size_t n) {
    if (b.size() < n) return false;
    for (size_t i = 0; i < n; ++i)
        if (static_cast<uint8_t>(b[i]) != static_cast<uint8_t>(magic[i])) return false;
    return true;
}

} // namespace

int main() {
    CHECK(qApp == nullptr,
          "seam precondition: test runs with NO QApplication (null qApp)");

    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() /
                         ("azu-export-mesh-" + std::to_string(static_cast<long>(::getpid())));
    fs::remove_all(dir);
    fs::create_directories(dir);

    // The controller is constructed but NEVER started: no worker thread, so
    // nothing can answer a mesh request and no extraction can race a write.
    PipelineController controller(PreprocessBackend::CPU);

    // --- 1. Empty shared mesh: refuse, write nothing, never call the writer.
    {
        const fs::path target = dir / "never_written.ply";
        int writer_calls = 0;
        const bool ok = controller.exportMesh(
            target.string(),
            [&](const MeshData&, const std::string&) { ++writer_calls; return true; });
        CHECK(!ok, "empty mesh: exportMesh reports failure");
        CHECK(writer_calls == 0, "empty mesh: writer_fn was never invoked");
        CHECK(!fs::exists(target), "empty mesh: no file was created");
        CHECK(controller.metricsSnapshot().export_pct == 0.0f,
              "empty mesh: export_pct stays 0 (truthful progress)");
    }

    // --- 2. Published mesh: writer_fn receives THE snapshot + exact path.
    auto mesh = std::make_shared<MeshData>(validMesh());
    controller.sharedMesh().update(mesh);
    {
        const fs::path target = dir / "via_helper.ply";
        int writer_calls = 0;
        bool saw_same_object = false;
        bool saw_path = false;
        bool saw_geometry = false;
        const bool ok = controller.exportMesh(
            target.string(),
            [&](const MeshData& m, const std::string& p) {
                ++writer_calls;
                saw_same_object = (&m == mesh.get());
                saw_path = (p == target.string());
                saw_geometry = (m.triangleCount() == 1 && m.positions.size() == 3);
                return true;
            });
        CHECK(ok, "published mesh: exportMesh reports success");
        CHECK(writer_calls == 1, "published mesh: writer_fn invoked exactly once");
        CHECK(saw_same_object, "published mesh: writer_fn got the shared snapshot itself");
        CHECK(saw_path, "published mesh: writer_fn got the exact path");
        CHECK(saw_geometry, "published mesh: writer_fn got the exported geometry");
        CHECK(controller.metricsSnapshot().export_pct == 100.0f,
              "success: export_pct ends at 100 (truthful progress)");
    }

    // --- 3. Writer failure and writer exceptions are FAILED exports, not
    //        crashes and not silent successes.
    {
        const std::string path = (dir / "writer_false.ply").string();
        CHECK(!controller.exportMesh(path,
                  [](const MeshData&, const std::string&) { return false; }),
              "writer returning false propagates as a failed export");
        CHECK(controller.metricsSnapshot().export_pct == 0.0f,
              "writer failure: export_pct falls back to 0");

        const std::string throwing = (dir / "writer_throws.ply").string();
        CHECK(!controller.exportMesh(throwing,
                  [](const MeshData&, const std::string&) -> bool {
                      throw std::runtime_error("simulated writer failure");
                  }),
              "std::exception from writer_fn becomes a failed export, not a crash");
        CHECK(!controller.exportMesh(throwing,
                  [](const MeshData&, const std::string&) -> bool {
                      throw 42;
                  }),
              "unknown exception from writer_fn becomes a failed export, not a crash");
        CHECK(!fs::exists(dir / "writer_throws.ply"),
              "throwing writer: no file was created");
    }

    // --- 4. exportPLY()/exportGLB() are this helper's two wrappers: the
    //        bytes they land on disk equal a direct exporter call on the same
    //        mesh, and the format magics prove real writers actually ran.
    {
        const fs::path ply_via_wrapper = dir / "wrapper.ply";
        const fs::path ply_direct      = dir / "direct.ply";
        const fs::path glb_via_wrapper = dir / "wrapper.glb";
        const fs::path glb_direct      = dir / "direct.glb";

        CHECK(controller.exportPLY(ply_via_wrapper.string()),
              "exportPLY succeeds through the shared helper");
        CHECK(kfusion::export_io::PLYExporter::writeBinary(*mesh, ply_direct.string()),
              "direct PLYExporter::writeBinary succeeds (reference bytes)");

        CHECK(controller.exportGLB(glb_via_wrapper.string()),
              "exportGLB succeeds through the shared helper");
        CHECK(kfusion::export_io::GLBExporter::write(*mesh, glb_direct.string()),
              "direct GLBExporter::write succeeds (reference bytes)");

        std::vector<char> a, b;
        CHECK(fileBytes(ply_via_wrapper, a) && fileBytes(ply_direct, b),
              "PLY outputs exist and read back");
        CHECK(startsWith(a, "ply", 3), "exportPLY output carries the PLY magic");
        CHECK(a == b, "exportPLY bytes == direct PLYExporter bytes (same helper path)");

        CHECK(fileBytes(glb_via_wrapper, a) && fileBytes(glb_direct, b),
              "GLB outputs exist and read back");
        CHECK(startsWith(a, "glTF", 4), "exportGLB output carries the glTF magic");
        CHECK(a == b, "exportGLB bytes == direct GLBExporter bytes (same helper path)");
    }

    fs::remove_all(dir);

    std::printf("pipeline_export_mesh_contract: %d checks, %d failures\n",
                g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
