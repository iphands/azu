// pipeline_test_seam_smoke (big-fix todo 2): starts the REAL
// src/app/PipelineController.cpp through the AZU_PIPELINE_TEST_SEAM without a
// sensor and without a QApplication, drives the real onRawFrame()/queue path
// with deterministic synthetic RawFrames until the real TSDF integration
// counter moves, asserts the null-qApp seam delivered UI-frame callbacks, and
// shuts the controller down cleanly. Hard caps: 16 injected frames / 5 s.

#include "app/PipelineController.h"

#include <QCoreApplication>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "pipeline_test_seam_smoke must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

using namespace std::chrono;
using kfusion::app::PipelineController;
using kfusion::sensor::DEPTH_HEIGHT;
using kfusion::sensor::DEPTH_WIDTH;
using kfusion::sensor::FrameData;
using kfusion::sensor::RawFrame;

constexpr int kMaxInjectFrames = 16;
constexpr int kBudgetMs = 5000;

int g_failures = 0;

#define CHECK(cond, what)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL: %s  [%s:%d]\n", (what), __FILE__, __LINE__);    \
            ++g_failures;                                                      \
        }                                                                      \
    } while (false)

// Deterministic synthetic frame: a frontal plane at ~0.85 m (raw 700 with a
// sub-quantization tilt inside the valid band) plus an RGB gradient. No
// device, no GPU, no randomness.
std::shared_ptr<RawFrame> makeSyntheticFrame(uint64_t id) {
    auto f = std::make_shared<RawFrame>();
    f->frame_id = id;
    f->timestamp_depth = static_cast<double>(id) / 30.0;
    f->timestamp_rgb = f->timestamp_depth;
    f->depth_valid = true;
    f->rgb_valid = true;
    for (int y = 0; y < DEPTH_HEIGHT; ++y) {
        for (int x = 0; x < DEPTH_WIDTH; ++x) {
            const int idx = y * DEPTH_WIDTH + x;
            // raw in [700, 715] -> ~0.84-0.86 m, inside [0.3, 2.5] default range
            f->depth[idx] = static_cast<uint16_t>(700 + ((x ^ y) & 15));
            f->rgb[idx * 3 + 0] = static_cast<uint8_t>((x * 3 + id) & 0xFF);
            f->rgb[idx * 3 + 1] = static_cast<uint8_t>((y * 5) & 0xFF);
            f->rgb[idx * 3 + 2] = static_cast<uint8_t>((x + y) & 0xFF);
        }
    }
    return f;
}

} // namespace

int main() {
    CHECK(qApp == nullptr,
          "seam precondition: test runs with NO QApplication (null qApp)");

    PipelineController controller(kfusion::sensor::PreprocessBackend::CPU);
    PipelineController::resetUiFrameTestStateForTests();

    std::atomic<int> hook_calls{0};
    std::atomic<int> hook_width{-1};
    std::atomic<int> hook_height{-1};
    PipelineController::registerUiFrameTestHookForTests(
        [&](const FrameData& frame) {
            hook_width.store(frame.width);
            hook_height.store(frame.height);
            hook_calls.fetch_add(1);
        });

    // Production contract: the UI-preview builder only dispatches when a
    // frame-ready callback is registered. With qApp null the seam delivers
    // through the hook above and records the count.
    controller.setFrameReadyCallback([](const FrameData&) {});

    CHECK(controller.startWithoutSensorForTests(),
          "startWithoutSensorForTests() returns true");
    CHECK(controller.isRunning(), "controller reports Running after seam start");

    const auto t0 = steady_clock::now();
    int injected = 0;
    int integrated = 0;
    while (integrated < 1) {
        const auto elapsed_ms = duration_cast<milliseconds>(steady_clock::now() - t0).count();
        if (elapsed_ms >= kBudgetMs) {
            break;
        }
        if (injected < kMaxInjectFrames) {
            ++injected;
            controller.injectRawFrameForTests(makeSyntheticFrame(injected));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        integrated = controller.metricsSnapshot().integrated_frames;
    }

    const int deliveries_at_check = PipelineController::uiFrameDeliveryCountForTests();

    // Keep the raw queue non-empty when stop() notifies its workers: the
    // stop-path lost-wakeup is todo 10's fix, and a parked worker with an
    // empty queue is exactly its symptom. This keeps the smoke bounded.
    for (int i = 0; i < 4; ++i) {
        controller.injectRawFrameForTests(makeSyntheticFrame(1000 + i));
    }

    controller.stop();
    CHECK(!controller.isRunning(), "controller stopped cleanly (no hang, no sensor)");

    const int deliveries_final = PipelineController::uiFrameDeliveryCountForTests();

    CHECK(integrated >= 1, "metrics.integrated_frames >= 1 through the real queue path");
    CHECK(deliveries_at_check > 0, "null-qApp seam callback delivery count > 0");
    CHECK(hook_calls.load() > 0, "test hook observed at least one UI frame");
    CHECK(hook_width.load() > 0 && hook_height.load() > 0,
          "hook received a sized preview FrameData");

    std::printf(
        "SEAM-RESULT injected_frames=%d integrated_frames=%d seam_deliveries_at_check=%d "
        "seam_deliveries_final=%d hook_calls=%d hook_ui_size=%dx%d elapsed_ms=%lld\n",
        injected, integrated, deliveries_at_check, deliveries_final, hook_calls.load(),
        hook_width.load(), hook_height.load(),
        static_cast<long long>(
            duration_cast<milliseconds>(steady_clock::now() - t0).count()));

    PipelineController::resetUiFrameTestStateForTests();

    if (g_failures == 0) {
        std::printf("pipeline_test_seam_smoke: PASS (real controller, no sensor, null qApp)\n");
        return 0;
    }
    std::printf("pipeline_test_seam_smoke: FAIL (%d checks failed)\n", g_failures);
    return 1;
}
