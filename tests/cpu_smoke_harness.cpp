// cpu_smoke_harness (big-fix todo 2): proves the CPU-only CTest harness
// compiles, links the Qt-free azu_test_core library, and can assert real
// numeric + export contracts. Every check below exercises production code
// from azu_test_core; there are no placeholder assertions.

#include "sensor/FrameData.h"
#include "sensor/KinectSensor.h"
#include "meshing/MeshData.h"
#include "export/PLYExporter.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "cpu_smoke_harness must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

int g_failures = 0;

#define CHECK(cond, what)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL: %s  [%s:%d]\n", (what), __FILE__, __LINE__);    \
            ++g_failures;                                                      \
        }                                                                      \
    } while (false)

int g_checks = 0;
#define CHECK_NEAR(actual, expected, tol, what)                                \
    do {                                                                       \
        ++g_checks;                                                            \
        const double _a = (actual), _e = (expected), _t = (tol);               \
        if (!(_a >= _e - _t && _a <= _e + _t)) {                               \
            std::printf("FAIL: %s  actual=%f expected=%f tol=%f [%s:%d]\n",    \
                        (what), _a, _e, _t, __FILE__, __LINE__);               \
            ++g_failures;                                                      \
        }                                                                      \
    } while (false)

void testDepthDomainContract() {
    using kfusion::sensor::rawDepthToMeters;
    CHECK(rawDepthToMeters(0) == 0.0f, "raw 0 is an invalid sentinel");
    CHECK(rawDepthToMeters(2047) == 0.0f, "raw 2047 is invalid");
    CHECK(rawDepthToMeters(2048) == 0.0f, "raw > 2047 is invalid");
    CHECK(rawDepthToMeters(500) < rawDepthToMeters(700),
          "conversion increases with raw inside the usable band");
    const float m700 = rawDepthToMeters(700);
    CHECK(m700 > 0.3f && m700 < 2.5f, "raw 700 lands inside the default scan range");
}

void testBackProjectionContract() {
    namespace sen = kfusion::sensor;
    std::vector<uint16_t> raw_depth(sen::FRAME_W * sen::FRAME_H, 700);
    std::vector<uint8_t> raw_rgb(sen::FRAME_W * sen::FRAME_H * 3, 0);
    for (int i = 0; i < sen::FRAME_W * sen::FRAME_H; ++i) {
        raw_rgb[i * 3 + 0] = static_cast<uint8_t>(i % 251);
        raw_rgb[i * 3 + 1] = static_cast<uint8_t>((i * 7) % 251);
        raw_rgb[i * 3 + 2] = static_cast<uint8_t>((i * 13) % 251);
    }
    raw_depth[0] = 0; // invalid sentinel pixel

    sen::FrameData fd;
    sen::buildFrameData(raw_depth.data(), raw_rgb.data(), fd, 0.3f, 2.5f);

    // Independent recomputation of the documented Kinect v1 conversion for the
    // plane value (not reusing the production helper): the 1/z polynomial.
    const double d_expect =
        1.0 / (700.0 * -0.0030711016 + 3.3309495161);
    const int cx_px = 320, cy_px = 240;
    const int idx = cy_px * sen::FRAME_W + cx_px;
    CHECK_NEAR(fd.depth_meters[idx], d_expect, 1e-5, "plane depth_meters matches back-projection");
    CHECK_NEAR(fd.vertices[idx].z(), d_expect, 1e-5, "vertex z == depth");
    const double x_expect = (cx_px - sen::CX) / static_cast<double>(sen::FX) * d_expect;
    const double y_expect = (cy_px - sen::CY) / static_cast<double>(sen::FY) * d_expect;
    CHECK_NEAR(fd.vertices[idx].x(), x_expect, 1e-4, "vertex x == (u-cx)/fx*z");
    CHECK_NEAR(fd.vertices[idx].y(), y_expect, 1e-4, "vertex y == (v-cy)/fy*z");

    CHECK(fd.depth_meters[0] == 0.0f, "invalid raw depth yields zero depth_meters");
    CHECK(fd.vertices[0].isZero(), "invalid raw depth yields zero vertex");
    CHECK(fd.rgb[idx * 3 + 0] == static_cast<uint8_t>(idx % 251), "rgb channel 0 passthrough");
    CHECK(fd.rgb[idx * 3 + 1] == static_cast<uint8_t>((idx * 7) % 251), "rgb channel 1 passthrough");
    CHECK(fd.rgb[idx * 3 + 2] == static_cast<uint8_t>((idx * 13) % 251), "rgb channel 2 passthrough");
}

void testPlyExportContract() {
    namespace mesh = kfusion::meshing;
    namespace io = kfusion::export_io;

    mesh::MeshData m;
    m.positions = {
        Eigen::Vector3f(0.0f, 0.0f, 0.5f),
        Eigen::Vector3f(1.0f, 0.0f, 0.5f),
        Eigen::Vector3f(0.0f, 1.0f, 0.5f),
    };
    m.indices = {0, 1, 2};

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("azu_cpu_smoke_harness_" + std::to_string(::getpid()) + ".ply");
    std::filesystem::remove(path);

    CHECK(io::PLYExporter::writeBinary(m, path.string()), "writeBinary succeeds");
    CHECK(std::filesystem::exists(path), "PLY file created");

    std::ifstream f(path, std::ios::binary);
    CHECK(f.is_open(), "PLY file readable");
    std::string header;
    std::string line;
    size_t header_bytes = 0;
    bool end_header = false;
    while (std::getline(f, line)) {
        header_bytes += line.size() + 1;
        header += line;
        header += "\n";
        if (line == "end_header") {
            end_header = true;
            break;
        }
    }
    CHECK(end_header, "PLY header terminates");
    CHECK(header.find("format binary_little_endian 1.0") != std::string::npos,
          "PLY declares binary_little_endian");
    CHECK(header.find("element vertex 3") != std::string::npos, "PLY vertex count");
    CHECK(header.find("element face 1") != std::string::npos, "PLY face count");

    const size_t expect_bytes = header_bytes + 3 * 12 + 1 * (1 + 3 * 4);
    CHECK(static_cast<size_t>(std::filesystem::file_size(path)) == expect_bytes,
          "PLY payload size matches header + 3 float3 vertices + 1 indexed face");

    float vx = 0, vy = 0, vz = 0;
    f.read(reinterpret_cast<char*>(&vx), 4);
    f.read(reinterpret_cast<char*>(&vy), 4);
    f.read(reinterpret_cast<char*>(&vz), 4);
    CHECK(vx == 0.0f && vy == 0.0f && vz == 0.5f, "first vertex payload matches input");

    std::filesystem::remove(path);

    mesh::MeshData empty;
    const std::filesystem::path bad_path = path.string() + ".empty";
    CHECK(!io::PLYExporter::writeBinary(empty, bad_path.string()),
          "empty mesh rejected by writeBinary");
    CHECK(!std::filesystem::exists(bad_path), "rejected export leaves no file");
}

} // namespace

int main() {
    testDepthDomainContract();
    testBackProjectionContract();
    testPlyExportContract();
    if (g_failures == 0) {
        std::printf("cpu_smoke_harness: PASS (depth-domain, back-projection, PLY export contracts; "
                    "%d numeric checks)\n",
                    g_checks);
        return 0;
    }
    std::printf("cpu_smoke_harness: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
