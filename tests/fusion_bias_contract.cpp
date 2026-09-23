// fusion_bias_contract (big-fix-two T0.9): the fused surface sits where the depth
// says it is. The old per-pixel ray march stored an SDF sampled at one point into
// a voxel read back at its corner, which put a noise-free plane ~6.7 mm in front
// of the truth at 1 cm voxels and made a static camera drift under ICP.
//
// Fixture: noise-free synthetic depth of a tilted plane z = 1 + 0.2 x rendered
// through the production 640x480 intrinsics from an identity pose, integrated 10
// times into a 256^3 / 1 cm volume, meshed with marching cubes. Asserts:
//   A  |mean signed vertex distance to the plane| < 0.5 mm, and 95th-percentile
//      |distance| < 2 mm
//   B  weight counts frames: after N identical frames every observed voxel has
//      weight min(N, max_weight)
//   C  byte-identical volume for OpenMP thread counts 1, 4 and 16
// and the O(1) observedFraction() counter agrees with the full-scan
// usageFraction() after integration.
#include "meshing/MarchingCubes.h"
#include "meshing/MeshData.h"
#include "sensor/FrameData.h"
#include "tsdf/TSDFVolume.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

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

using kfusion::tsdf::TSDFParams;
using kfusion::tsdf::TSDFVolume;
using kfusion::tsdf::Voxel;
namespace sensor = kfusion::sensor;

constexpr int W = sensor::FRAME_W, H = sensor::FRAME_H;
constexpr double kSlope = 0.2, kZ0 = 1.0;
const double kNorm = std::sqrt(1.0 + kSlope * kSlope);

// Signed distance to the plane kSlope*x - z + kZ0 = 0 (positive toward the camera).
double planeDistance(const Eigen::Vector3f& p) {
    return (kSlope * p.x() - p.z() + kZ0) / kNorm;
}

std::vector<float> planeDepth() {
    std::vector<float> d(static_cast<size_t>(W) * H);
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            const double a = (u - sensor::CX) / sensor::FX;
            d[static_cast<size_t>(v) * W + u] = static_cast<float>(kZ0 / (1.0 - kSlope * a));
        }
    }
    return d;
}

TSDFParams params() {
    TSDFParams p;
    p.resolution = 256;
    p.voxel_size = 0.01f;
    p.truncation = 0.03f;
    p.max_weight = 64.0f;
    p.origin     = Eigen::Vector3f(-1.28f, -1.28f, 0.0f);
    return p;
}

void integrateN(TSDFVolume& vol, const std::vector<float>& depth, int n) {
    for (int i = 0; i < n; ++i) {
        vol.integrate(depth.data(), nullptr, Eigen::Matrix4f::Identity(), sensor::FX, sensor::FY,
                      sensor::CX, sensor::CY, W, H, 0.3f, 3.0f);
    }
}

void sectionBias() {
    const std::vector<float> depth = planeDepth();
    TSDFVolume vol(params());
    integrateN(vol, depth, 10);

    kfusion::meshing::MarchingCubes mc;
    auto mesh = mc.extract(vol);
    CHECK(mesh && mesh->positions.size() > 10000, "A: the plane meshes");
    if (!mesh || mesh->positions.empty()) return;

    std::vector<double> absd;
    double sum = 0.0;
    for (const auto& p : mesh->positions) {
        const double d = planeDistance(p);
        sum += d;
        absd.push_back(std::fabs(d));
    }
    const double mean = sum / static_cast<double>(absd.size());
    std::sort(absd.begin(), absd.end());
    const double p95 = absd[absd.size() * 95 / 100];
    std::printf("  A: %zu vertices, mean signed %.4f mm, p95 |d| %.4f mm\n", absd.size(),
                mean * 1e3, p95 * 1e3);
    CHECK(std::fabs(mean) < 0.5e-3, "A: mean signed surface error < 0.5 mm");
    CHECK(p95 < 2.0e-3, "A: 95th-percentile surface error < 2 mm");
}

void sectionWeights() {
    const std::vector<float> depth = planeDepth();
    for (int n : {1, 3, 70}) {
        TSDFVolume vol(params());
        integrateN(vol, depth, n);
        const float want = std::min(static_cast<float>(n), vol.params().max_weight);
        size_t observed = 0, wrong = 0;
        for (const Voxel& v : vol.voxelData()) {
            if (v.weight == 0.0f) continue;
            ++observed;
            if (v.weight != want) ++wrong;
        }
        CHECK(std::fabs(vol.observedFraction() - vol.usageFraction()) < 1e-9f,
              "B: O(1) observedFraction() equals the full-scan usageFraction()");
        CHECK(observed > 0 && wrong == 0,
              "B: after " + std::to_string(n) + " frames every observed voxel has weight " +
                  std::to_string(want) + " (" + std::to_string(wrong) + " of " +
                  std::to_string(observed) + " differ)");
    }
}

void sectionThreads() {
#ifdef _OPENMP
    const std::vector<float> depth = planeDepth();
    const int saved = omp_get_max_threads();
    std::vector<Voxel> ref;
    for (int t : {1, 4, 16}) {
        omp_set_num_threads(t);
        TSDFVolume vol(params());
        integrateN(vol, depth, 3);
        if (ref.empty()) {
            ref = vol.voxelData();
        } else {
            CHECK(std::memcmp(ref.data(), vol.voxelData().data(), ref.size() * sizeof(Voxel)) == 0,
                  "C: volume bytes identical at " + std::to_string(t) + " threads");
        }
    }
    omp_set_num_threads(saved);
#endif
}

}  // namespace

int main() {
    sectionBias();
    sectionWeights();
    sectionThreads();
    if (g_failures == 0) {
        std::printf("fusion_bias_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("fusion_bias_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
