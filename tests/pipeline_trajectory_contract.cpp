// pipeline_trajectory_contract (big-fix-two T0.9-T0.12): closed-loop tracking
// through the REAL PipelineController on a moving synthetic camera.
//
// tests/support/SyntheticScene.h renders the room from a known trajectory; each
// depth image is quantized to raw Kinect 11-bit codes (cpuDepthMetersToRaw) and
// injected through the seam, so it passes the production preprocessing,
// pyramid, ICP, tracking policy, TSDF integration and raycast. The first frame
// defines the world frame, which the trajectory starts at, so estimated and
// true poses are directly comparable.
//
//   A  90 frames, 2 mm + 0.25 deg per frame: never TrackingLost, every frame
//      graded (not Failed) and the trajectory RMS error (ATE) < 1 cm
//   B  final pose within 2 cm / 1 deg of the truth
#include "app/PipelineController.h"
#include "sensor/DepthValidity.h"
#include "sensor/KinectSensor.h"
#include "support/SyntheticScene.h"

#include <QCoreApplication>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "pipeline_trajectory_contract needs AZU_PIPELINE_TEST_SEAM"
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

using kfusion::app::PipelineController;
using kfusion::app::PipelineState;
using kfusion::sensor::RawFrame;
using namespace std::chrono_literals;

std::shared_ptr<RawFrame> rawFrom(const std::vector<float>& depth, uint64_t id) {
    auto f = std::make_shared<RawFrame>();
    f->frame_id    = id;
    f->depth_valid = true;
    f->rgb_valid   = true;
    for (size_t i = 0; i < depth.size(); ++i) {
        f->depth[i] = depth[i] > 0.0f ? kfusion::sensor::cpuDepthMetersToRaw(depth[i], 0.3f, 5.0f) : 0;
    }
    for (size_t i = 0; i < f->rgb.size(); ++i) f->rgb[i] = static_cast<uint8_t>(90 + (i % 7) * 20);
    return f;
}

}  // namespace

int main() {
    CHECK(qApp == nullptr, "seam precondition: no QApplication");
    const azu_test::SyntheticScene scene;
    PipelineController pc(kfusion::sensor::PreprocessBackend::CPU);
    CHECK(pc.startWithoutSensorForTests(), "seam start");

    constexpr int kFrames = 90;
    double sq_sum = 0.0;
    int lost = 0, failed = 0, timeouts = 0;
    Eigen::Matrix4f truth = Eigen::Matrix4f::Identity(), est = Eigen::Matrix4f::Identity();
    for (int i = 0; i < kFrames; ++i) {
        truth = azu_test::makePose({0.0f, 0.00436f * i, 0.0f}, {0.002f * i, 0.0f, 0.0f});
        const uint64_t before = pc.trackedFrameCountForTests();
        pc.injectRawFrameForTests(rawFrom(scene.renderDepth(truth), static_cast<uint64_t>(i + 1)));
        // Frame 1 is the world origin and is not graded; wait for the others.
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        while (i > 0 && pc.trackedFrameCountForTests() == before &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(2ms);
        }
        if (i > 0 && pc.trackedFrameCountForTests() == before) ++timeouts;
        std::this_thread::sleep_for(40ms);   // let integration + raycast refresh the model
        const auto m = pc.metricsSnapshot();
        if (m.state == PipelineState::TrackingLost) ++lost;
        if (i > 0 && m.tracking_quality == 2) ++failed;
        est = pc.currentPose();
        const auto e = azu_test::poseError(est, truth);
        sq_sum += static_cast<double>(e.trans_m) * e.trans_m;
    }
    pc.stop();

    const double ate = std::sqrt(sq_sum / kFrames);
    const auto fin = azu_test::poseError(est, truth);
    std::printf("  A: ATE %.2f mm, lost %d, failed %d, timeouts %d | final %.2f mm / %.3f deg\n",
                ate * 1e3, lost, failed, timeouts, fin.trans_m * 1e3, fin.rot_deg);
    CHECK(timeouts == 0, "A: every frame was tracked");
    CHECK(lost == 0, "A: tracking is never lost on a smooth 30 Hz trajectory");
    CHECK(failed == 0, "A: no frame is graded Failed");
    CHECK(ate < 0.01, "A: trajectory RMS error < 1 cm");
    CHECK(fin.trans_m < 0.02f && fin.rot_deg < 1.0f, "B: final pose within 2 cm / 1 deg");

    if (g_failures == 0) {
        std::printf("pipeline_trajectory_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("pipeline_trajectory_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
