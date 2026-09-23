// icp_refpose_contract (big-fix-two T0.11): ICP must project live points into
// the model image with the pose the model was RAYCAST from.
//
// The pipeline tracks frame N+1 against a model raycast after integrating some
// earlier frame, i.e. at an older pose. It used to pass the latest tracked
// pose as ICP's reference pose, so every live point was looked up at the wrong
// model pixel: a silent bias that grows with model lag and turns into tracking
// loss. ModelFrame now carries the pose it was raycast at.
//
// Fixture: tests/support/SyntheticScene.h room, 256^3 / 1 cm volume fused
// noise-free from P_model. The camera has since moved to P_prev (last tracked
// pose, the ICP initial estimate) and the live frame is at P_live.
//   A  ref = model.pose (what the pipeline now passes): converges to P_live
//      within 1 mm / 0.05 deg
//   B  ref = P_prev (the old bug): measurably wrong or rejected — proof the
//      fixture is sensitive to the reference pose at all
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

}  // namespace

int main() {
    const azu_test::SyntheticScene scene;
    const azu_test::Intrinsics K;

    kfusion::tsdf::TSDFParams p;
    p.min_depth = 0.3f;
    p.max_depth = 3.0f;
    kfusion::tsdf::TSDFVolume vol(p);

    const Eigen::Matrix4f P_model = makePose({0.0f, 0.05f, 0.0f}, {0.02f, 0.0f, 0.1f});
    const std::vector<float> d_model = scene.renderDepth(P_model, K);
    for (int i = 0; i < 5; ++i)
        vol.integrate(d_model.data(), nullptr, P_model, K.fx, K.fy, K.cx, K.cy, K.width, K.height,
                      0.3f, 3.0f);

    tracking::ModelFrame model;
    vol.raycast(P_model, K.fx, K.fy, K.cx, K.cy, K.width, K.height, model.vertices.data(),
                model.normals.data(), model.colors.data());
    model.pose = P_model;

    // 0.2 s of model lag at 0.5 rad/s + 0.15 m/s, then one more 30 Hz frame.
    const Eigen::Matrix4f P_prev = P_model * makePose({0.02f, 0.1f, 0.0f}, {0.03f, 0.0f, 0.01f});
    const Eigen::Matrix4f P_live = P_prev * makePose({0.0f, 0.015f, 0.003f}, {0.004f, 0.0f, 0.002f});

    sensor::FrameData live;
    azu_test::fillFrame(scene.renderDepth(P_live, K), live, K);
    sensor::FramePyramid pyr;
    sensor::buildFramePyramid(live, pyr);

    tracking::ICPTracker tracker;

    const tracking::ICPResult good = tracker.track(pyr, model, P_prev, model.pose);
    const auto eg = poseError(good.pose, P_live);
    std::printf("  A ref=model.pose: ok=%d inliers=%d err %.3f mm / %.4f deg\n", good.tracking_ok,
                good.inliers, eg.trans_m * 1e3, eg.rot_deg);
    CHECK(good.tracking_ok, "A: tracking succeeds with the model's raycast pose as reference");
    CHECK(eg.trans_m < 1e-3f && eg.rot_deg < 0.05f, "A: converges within 1 mm / 0.05 deg");

    const tracking::ICPResult bad = tracker.track(pyr, model, P_prev, P_prev);
    const auto eb = poseError(bad.pose, P_live);
    std::printf("  B ref=prev pose: ok=%d inliers=%d err %.3f mm / %.4f deg\n", bad.tracking_ok,
                bad.inliers, eb.trans_m * 1e3, eb.rot_deg);
    CHECK(!bad.tracking_ok || eb.trans_m > 5e-3f || eb.rot_deg > 0.25f,
          "B: the wrong reference pose is detectably worse (fixture is sensitive)");

    if (g_failures == 0) {
        std::printf("icp_refpose_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("icp_refpose_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
