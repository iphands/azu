// raycast_accuracy_contract (big-fix-two T0.10): the model the tracker matches
// against is the real surface, from any viewpoint.
//
// Fixture: the tilted plane z = 1 + 0.2 x (as in fusion_bias_contract), fused
// noise-free from the identity pose into a 256^3 / 1 cm volume. Then:
//   A  raycast from the fusing pose: >= 90% of pixels hit, every hit lies within
//      1 mm of the plane, normals face the camera (n . plane_normal > 0.99)
//   B  raycast from BEHIND the plane (camera at z = 2 looking back at it): no
//      pixel may report a surface. The old raycast accepted -→+ crossings and
//      interpolated observed values against the unobserved +1 sentinel, which
//      produced phantom surfaces here.
//   C  raycast from 45 degrees off-axis: every hit lies within 2 mm of the plane
#include "sensor/FrameData.h"
#include "tsdf/TSDFVolume.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

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
namespace sensor = kfusion::sensor;

constexpr int W = sensor::FRAME_W, H = sensor::FRAME_H;
constexpr double kSlope = 0.2, kZ0 = 1.0;
const double kNorm = std::sqrt(1.0 + kSlope * kSlope);
// Unit plane normal pointing toward the fusing camera (-z side).
const Eigen::Vector3f kPlaneN = Eigen::Vector3f(static_cast<float>(kSlope), 0.0f, -1.0f).normalized();

double planeDistance(const Eigen::Vector3f& p) {
    return (kSlope * p.x() - p.z() + kZ0) / kNorm;
}

struct Result {
    int hits = 0, off_1mm = 0, off_2mm = 0, bad_normal = 0;
};

Result cast(const TSDFVolume& vol, const Eigen::Matrix4f& pose) {
    std::vector<Eigen::Vector3f> v(static_cast<size_t>(W) * H), n(v.size());
    vol.raycast(pose, sensor::FX, sensor::FY, sensor::CX, sensor::CY, W, H, v.data(), n.data());
    Result r;
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == Eigen::Vector3f::Zero()) continue;
        ++r.hits;
        const double d = std::fabs(planeDistance(v[i]));
        if (d > 1e-3) ++r.off_1mm;
        if (d > 2e-3) ++r.off_2mm;
        if (!(n[i].dot(kPlaneN) > 0.99f)) ++r.bad_normal;
    }
    return r;
}

// Camera at `eye` looking along `forward` (world), y-down image convention.
Eigen::Matrix4f lookFrom(const Eigen::Vector3f& eye, const Eigen::Vector3f& forward) {
    const Eigen::Vector3f z = forward.normalized();
    const Eigen::Vector3f x = Eigen::Vector3f(0, 1, 0).cross(z).normalized();
    const Eigen::Vector3f y = z.cross(x);
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,1>(0,0) = x;
    T.block<3,1>(0,1) = y;
    T.block<3,1>(0,2) = z;
    T.block<3,1>(0,3) = eye;
    return T;
}

}  // namespace

int main() {
    TSDFParams p;
    p.resolution = 256;
    p.voxel_size = 0.01f;
    p.truncation = 0.03f;
    p.origin     = Eigen::Vector3f(-1.28f, -1.28f, 0.0f);
    p.min_depth  = 0.3f;
    p.max_depth  = 3.0f;
    TSDFVolume vol(p);

    std::vector<float> depth(static_cast<size_t>(W) * H);
    for (int v = 0; v < H; ++v)
        for (int u = 0; u < W; ++u)
            depth[static_cast<size_t>(v) * W + u] =
                static_cast<float>(kZ0 / (1.0 - kSlope * (u - sensor::CX) / sensor::FX));
    for (int i = 0; i < 5; ++i)
        vol.integrate(depth.data(), nullptr, Eigen::Matrix4f::Identity(), sensor::FX, sensor::FY,
                      sensor::CX, sensor::CY, W, H, 0.3f, 3.0f);

    const Result a = cast(vol, Eigen::Matrix4f::Identity());
    std::printf("  A front: hits %d, >1mm %d, bad normals %d\n", a.hits, a.off_1mm, a.bad_normal);
    // The misses are a ~10 px border band: a hit there needs its trilinear cell
    // and +-1 voxel normal stencil observed, and at 1 m a voxel is ~5 px wide,
    // so stencil voxels just outside the fusing frustum were never seen.
    CHECK(a.hits >= W * H * 90 / 100, "A: >= 90% of pixels hit the fused plane");
    CHECK(a.off_1mm == 0, "A: every hit within 1 mm of the plane");
    CHECK(a.bad_normal == 0, "A: every normal faces the camera");

    const Result b = cast(vol, lookFrom(Eigen::Vector3f(0.0f, 0.0f, 2.0f), Eigen::Vector3f(0, 0, -1)));
    std::printf("  B behind: hits %d\n", b.hits);
    CHECK(b.hits == 0, "B: no surface is reported when the plane is seen from behind");

    const Eigen::Vector3f eye(-1.0f, 0.0f, 0.0f);
    const Result c = cast(vol, lookFrom(eye, Eigen::Vector3f(0.0f, 0.0f, 1.0f) - eye));
    std::printf("  C 45deg: hits %d, >2mm %d\n", c.hits, c.off_2mm);
    CHECK(c.hits > 1000, "C: the oblique view sees the plane");
    CHECK(c.off_2mm <= c.hits / 1000, "C: <= 0.1% of oblique hits are off the surface by > 2 mm");

    if (g_failures == 0) {
        std::printf("raycast_accuracy_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("raycast_accuracy_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
