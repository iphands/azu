// pipeline_relocalization_contract (big-fix-two T0.13): losing tracking must be
// recoverable and must not freeze the viewport.
//
// Real PipelineController through the seam, synthetic room (SyntheticScene):
//   frames  0..24  smooth motion (2 mm + 0.25 deg per frame)
//   frames 25..29  the camera faces a wall 5 cm away: no depth in band, every
//                  frame Failed, TrackingLost after three
//   frames 30..69  the camera returns 12 deg / 3 cm away from the last good pose
//                  and keeps moving smoothly
// Asserts:
//   A  TrackingLost is entered during the look-away
//   B  tracking recovers (state Running) within 10 frames of the return
//   C  after recovery the pose is within 2 cm / 1 deg of the truth
//   D  preview frames keep arriving while lost (no frozen viewport)
#include "app/PipelineController.h"
#include "sensor/DepthValidity.h"
#include "sensor/KinectSensor.h"
#include "support/SyntheticScene.h"

#include <QCoreApplication>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "pipeline_relocalization_contract needs AZU_PIPELINE_TEST_SEAM"
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
    return f;
}

}  // namespace

int main() {
    CHECK(qApp == nullptr, "seam precondition: no QApplication");
    std::atomic<int> previews{0};
    PipelineController::resetUiFrameTestStateForTests();
    PipelineController::registerUiFrameTestHookForTests(
        [&previews](const kfusion::sensor::FrameData&) { previews.fetch_add(1); });

    const azu_test::SyntheticScene scene;
    PipelineController pc(kfusion::sensor::PreprocessBackend::CPU);
    pc.setFrameReadyCallback([](const kfusion::sensor::FrameData&) {});
    CHECK(pc.startWithoutSensorForTests(), "seam start");

    auto smooth = [](int i) {
        return azu_test::makePose({0.0f, 0.00436f * i, 0.0f}, {0.002f * i, 0.0f, 0.0f});
    };
    const Eigen::Matrix4f last_good = smooth(24);
    const Eigen::Matrix4f away      = azu_test::makePose({0.0f, 3.14159f, 0.0f}, {0.0f, 0.0f, 0.0f});
    const Eigen::Matrix4f offset    = azu_test::makePose({0.0f, 0.2094f, 0.0f}, {0.03f, 0.0f, 0.0f});

    bool entered_lost = false;
    int recovered_at = -1;
    int previews_during_lost = 0;
    Eigen::Matrix4f truth = Eigen::Matrix4f::Identity();
    for (int i = 0; i < 70; ++i) {
        if (i < 25)      truth = smooth(i);
        else if (i < 30) truth = away;
        else             truth = last_good * offset * smooth(i - 30);
        const uint64_t before = pc.trackedFrameCountForTests();
        const int previews_before = previews.load();
        pc.injectRawFrameForTests(rawFrom(scene.renderDepth(truth), static_cast<uint64_t>(i + 1)));
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        while (i > 0 && pc.trackedFrameCountForTests() == before &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(2ms);
        }
        std::this_thread::sleep_for(110ms);   // > the 10 Hz preview throttle
        const PipelineState st = pc.metricsSnapshot().state;
        if (st == PipelineState::TrackingLost) {
            entered_lost = true;
            previews_during_lost += previews.load() - previews_before;
        }
        if (i >= 30 && recovered_at < 0 && st == PipelineState::Running) recovered_at = i;
    }
    const auto err = azu_test::poseError(pc.currentPose(), truth);
    pc.stop();
    PipelineController::resetUiFrameTestStateForTests();

    std::printf("  lost=%d recovered_at=%d previews_while_lost=%d final %.2f mm / %.3f deg\n",
                entered_lost, recovered_at, previews_during_lost, err.trans_m * 1e3, err.rot_deg);
    CHECK(entered_lost, "A: looking at a 5 cm wall enters TrackingLost");
    CHECK(recovered_at >= 30 && recovered_at < 40, "B: tracking recovers within 10 frames");
    CHECK(err.trans_m < 0.02f && err.rot_deg < 1.0f, "C: pose after recovery within 2 cm / 1 deg");
    CHECK(previews_during_lost > 0, "D: the viewport keeps getting frames while lost");

    if (g_failures == 0) {
        std::printf("pipeline_relocalization_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("pipeline_relocalization_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
