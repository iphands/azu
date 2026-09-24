// icp_lowres_model_contract (relocalization rework, step 2): ICP against a model
// image of any size.
//
// Relocalization scores many candidate poses, each against the model raycast at
// that pose. A 640x480 raycast per candidate is too slow on the CPU (19-43 ms),
// so it renders 160x120 / 320x240 images instead, with the intrinsics scaled to
// that size (sensor::scaleIntrinsics). ICP then has to project into the model
// image with the SAME scaled intrinsics and index it by its own width.
//
// Fixture: tests/support/SyntheticScene.h room, 256^3 / 1 cm volume fused
// noise-free (as icp_refpose_contract).
//   A  scaleIntrinsics: identical at the same size, exact at 1/2 and 1/4
//   B  a 640x480 model still converges within 1 mm / 0.05 deg (unchanged path)
//   C  a 320x240 model, full solve: within 2 mm / 0.1 deg
//   D  a 160x120 model, coarsest level only, from 8 deg / 3 cm off: in the basin
//      (within 1 cm / 0.5 deg)
//   E  a 160x120 model raycast with the UNSCALED intrinsics is measurably wrong:
//      the fixture sees an intrinsics mismatch at all
#include "sensor/CameraIntrinsics.h"
#include "sensor/FrameData.h"
#include "support/SyntheticScene.h"
#include "tracking/ICPTracker.h"
#include "tsdf/TSDFVolume.h"

#include <cstdio>
#include <string>

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

namespace sensor   = kfusion::sensor;
namespace tracking = kfusion::tracking;
using azu_test::makePose;
using azu_test::poseError;

tracking::ModelFrame render(const kfusion::tsdf::TSDFVolume& vol, const Eigen::Matrix4f& pose,
                            int w, int h, const sensor::CameraIntrinsics& k) {
    tracking::ModelFrame m(w, h);
    vol.raycast(pose, k.fx, k.fy, k.cx, k.cy, w, h, m.vertices.data(), m.normals.data(),
                m.colors.data());
    m.pose = pose;
    return m;
}

}  // namespace

int main() {
    const azu_test::SyntheticScene scene;
    const azu_test::Intrinsics K;
    const sensor::CameraIntrinsics cam{K.fx, K.fy, K.cx, K.cy};

    // A
    const sensor::CameraIntrinsics same = sensor::scaleIntrinsics(cam, 640, 480, 640, 480);
    CHECK(same.fx == cam.fx && same.fy == cam.fy && same.cx == cam.cx && same.cy == cam.cy,
          "A: identical at the same size");
    const sensor::CameraIntrinsics half = sensor::scaleIntrinsics(cam, 640, 480, 320, 240);
    const sensor::CameraIntrinsics quarter = sensor::scaleIntrinsics(cam, 640, 480, 160, 120);
    CHECK(half.fx == cam.fx * 0.5f && half.cx == (cam.cx + 0.5f) * 0.5f - 0.5f &&
              half.cy == (cam.cy + 0.5f) * 0.5f - 0.5f,
          "A: half size");
    CHECK(quarter.fy == cam.fy * 0.25f && quarter.cx == (cam.cx + 0.5f) * 0.25f - 0.5f,
          "A: quarter size");

    kfusion::tsdf::TSDFParams p;
    p.min_depth = 0.3f;
    p.max_depth = 3.0f;
    kfusion::tsdf::TSDFVolume vol(p);
    const Eigen::Matrix4f P_model = makePose({0.0f, 0.05f, 0.0f}, {0.02f, 0.0f, 0.1f});
    const std::vector<float> d_model = scene.renderDepth(P_model, K);
    for (int i = 0; i < 5; ++i)
        vol.integrate(d_model.data(), nullptr, P_model, K.fx, K.fy, K.cx, K.cy, K.width, K.height,
                      0.3f, 3.0f);

    // B, C: one frame of motion from the model pose.
    const Eigen::Matrix4f P_live = P_model * makePose({0.0f, 0.02f, 0.004f}, {0.006f, 0.0f, 0.003f});
    sensor::FrameData live;
    azu_test::fillFrame(scene.renderDepth(P_live, K), live, K);
    sensor::FramePyramid pyr;
    sensor::buildFramePyramid(live, pyr);
    tracking::ICPTracker tracker;
    tracker.setIntrinsics(cam);

    const tracking::ModelFrame full = render(vol, P_model, 640, 480, cam);
    const tracking::ICPResult rb = tracker.track(pyr, full, P_model, full.pose);
    const auto eb = poseError(rb.pose, P_live);
    std::printf("  B 640x480: inliers %d err %.3f mm / %.4f deg\n", rb.inliers, eb.trans_m * 1e3,
                eb.rot_deg);
    CHECK(eb.trans_m < 1e-3f && eb.rot_deg < 0.05f, "B: 640x480 within 1 mm / 0.05 deg");

    const tracking::ModelFrame mid = render(vol, P_model, 320, 240, half);
    const tracking::ICPResult rc = tracker.track(pyr, mid, P_model, mid.pose);
    const auto ec = poseError(rc.pose, P_live);
    std::printf("  C 320x240: inliers %d err %.3f mm / %.4f deg\n", rc.inliers, ec.trans_m * 1e3,
                ec.rot_deg);
    CHECK(ec.trans_m < 2e-3f && ec.rot_deg < 0.1f, "C: 320x240 within 2 mm / 0.1 deg");

    // D, E: a relocalization-style coarse solve from 8 deg / 3 cm off.
    const Eigen::Matrix4f P_far = P_model * makePose({0.0f, 0.1396f, 0.0f}, {0.03f, 0.0f, 0.0f});
    sensor::FrameData live_far;
    azu_test::fillFrame(scene.renderDepth(P_far, K), live_far, K);
    sensor::FramePyramid pyr_far;
    sensor::buildFramePyramid(live_far, pyr_far);
    tracking::ICPParams coarse;
    coarse.max_iterations[0] = 0;
    coarse.max_iterations[1] = 0;
    coarse.max_iterations[2] = 30;
    coarse.dist_threshold = 0.25f;
    coarse.angle_threshold = 45.0f;
    tracker.setParams(coarse);

    const tracking::ModelFrame small = render(vol, P_model, 160, 120, quarter);
    const tracking::ICPResult rd = tracker.track(pyr_far, small, P_model, small.pose);
    const auto ed = poseError(rd.pose, P_far);
    std::printf("  D 160x120 coarse from 8 deg / 3 cm: inliers %d err %.2f mm / %.3f deg\n",
                rd.inliers, ed.trans_m * 1e3, ed.rot_deg);
    CHECK(ed.trans_m < 0.01f && ed.rot_deg < 0.5f, "D: 160x120 coarse lands in the basin");

    const tracking::ModelFrame wrong = render(vol, P_model, 160, 120, cam);   // unscaled K
    const tracking::ICPResult re = tracker.track(pyr_far, wrong, P_model, wrong.pose);
    const auto ee = poseError(re.pose, P_far);
    std::printf("  E 160x120 with unscaled intrinsics: inliers %d err %.2f mm / %.3f deg\n",
                re.inliers, ee.trans_m * 1e3, ee.rot_deg);
    CHECK(re.inliers < 100 || ee.trans_m > 0.02f || ee.rot_deg > 1.0f,
          "E: an intrinsics mismatch is measurably wrong (fixture is sensitive)");

    if (g_failures == 0) {
        std::printf("icp_lowres_model_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("icp_lowres_model_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
