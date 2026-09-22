// pipeline_hyperparams_contract (big-fix todo 24): bounded, deterministic,
// CPU-only propagation + callback-race test against the REAL
// src/app/PipelineController.cpp through the AZU_PIPELINE_TEST_SEAM. It proves,
// with no device, display, GPU, or filesystem access:
//   * each worker reads the depth band per processed frame, so a mid-run
//     setHyperparams() reaches tracking AND integration on the next frame;
//   * a live setHyperparams() reaches the preprocessor (SR scale) while running;
//   * setNumThreads() while running is parked and applied at a frame boundary,
//     and while stopped it applies immediately;
//   * the UI subscriber is invoked with callback_mutex_ released;
//   * stop() leaves every guarded lock free.
// Correctness is decided ONLY by those post-conditions, observed through
// generation counters the production loops themselves advance; CTest TIMEOUT is
// the hang detector and no sleep proves success.

#include "app/PipelineController.h"

#include <QCoreApplication>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <thread>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "pipeline_hyperparams_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

using namespace std::chrono;
using kfusion::app::FusionHyperparams;
using kfusion::app::PipelineController;
using kfusion::app::PipelineState;
using kfusion::sensor::DEPTH_HEIGHT;
using kfusion::sensor::DEPTH_WIDTH;
using kfusion::sensor::FrameData;
using kfusion::sensor::PreprocessBackend;
using kfusion::sensor::RawFrame;

// 16 frames is ~8s of pipeline time and is comfortably enough for a worker to
// record a band twice; every wait below is bounded by this count, never by time.
constexpr int kMaxPumpFrames = 16;
constexpr auto kFrameSpacing = std::chrono::milliseconds(500);

// A flat plane loses tracking after frame 1 (no depth gradient for ICP to lock
// onto), which stops integration and starves the observations this test needs.
// A paraboloid gives ICP curvature to track and keeps the pipeline running.
constexpr int kRawDepthCenter = 700;
constexpr int kRawDepthRelief = 90;

int g_failures = 0;

#define CHECK(cond, what)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL: %s  [%s:%d]\n", (what), __FILE__, __LINE__);    \
            ++g_failures;                                                      \
        }                                                                      \
    } while (false)

bool near(float a, float b) { return std::fabs(a - b) < 1e-6f; }

std::shared_ptr<RawFrame> makeSyntheticFrame(uint64_t id) {
    auto f = std::make_shared<RawFrame>();
    f->frame_id        = id;
    f->timestamp_depth = static_cast<double>(id) / 30.0;
    f->timestamp_rgb   = f->timestamp_depth;
    f->depth_valid     = true;
    f->rgb_valid       = true;
    for (int y = 0; y < DEPTH_HEIGHT; ++y) {
        for (int x = 0; x < DEPTH_WIDTH; ++x) {
            const int   idx = y * DEPTH_WIDTH + x;
            const float u   = (static_cast<float>(x) - 320.0f) / 320.0f;
            const float v   = (static_cast<float>(y) - 240.0f) / 240.0f;
            int raw = kRawDepthCenter + static_cast<int>(kRawDepthRelief * (u * u + v * v));
            if (raw < 1)   raw = 1;
            if (raw > 950) raw = 950;
            f->depth[idx]      = static_cast<uint16_t>(raw);
            f->rgb[idx * 3 + 0] = static_cast<uint8_t>((x * 3 + id) & 0xFF);
            f->rgb[idx * 3 + 1] = static_cast<uint8_t>((y * 5) & 0xFF);
            f->rgb[idx * 3 + 2] = static_cast<uint8_t>((x + y) & 0xFF);
        }
    }
    return f;
}

// Keeps the pipeline fed until `done` holds, bounded by frame count. A stuck
// pipeline exits the loop and the CHECK on the caller decides the failure.
bool pumpUntil(PipelineController &controller, uint64_t &next_id,
               const std::function<bool()> &done) {
    for (int injected = 0; injected < kMaxPumpFrames && !done(); ++injected) {
        controller.injectRawFrameForTests(makeSyntheticFrame(next_id++));
        std::this_thread::sleep_for(kFrameSpacing);
    }
    return done();
}

} // namespace

int main() {
    CHECK(qApp == nullptr,
          "seam precondition: test runs with NO QApplication (null qApp)");

    PipelineController controller(PreprocessBackend::CPU);
    PipelineController::resetUiFrameTestStateForTests();

    // A non-null production subscriber (the GUI contract) plus the seam hook that
    // observes delivery. The hook asserts the callback lock is NOT held while a
    // subscriber runs — the copy-then-invoke contract.
    std::atomic<bool> hook_saw_callback_lock{false};
    controller.setFrameReadyCallback([](const FrameData &) {});
    PipelineController::registerUiFrameTestHookForTests(
        [&controller, &hook_saw_callback_lock](const FrameData &) {
            if (!controller.callbackMutexIsFreeForTests())
                hook_saw_callback_lock.store(true);
        });
    CHECK(controller.callbackMutexIsFreeForTests(),
          "callback_mutex_ is free outside the setters");

    // Stopped path: applied immediately, before any worker exists.
    controller.setNumThreads(1);
    CHECK(controller.appliedThreadCountForTests() == 1,
          "setNumThreads() while stopped applies immediately");

    CHECK(controller.startWithoutSensorForTests(),
          "startWithoutSensorForTests() returns true");
    CHECK(controller.isRunning(), "controller reports Running after seam start");

    const FusionHyperparams defaults = controller.hyperparamsSnapshot();
    CHECK(near(defaults.min_depth, 0.30f) && near(defaults.max_depth, 2.50f),
          "default depth band is 0.30m..2.50m");
    CHECK(defaults.sr_scale == 2, "default SR scale is 2");
    CHECK(controller.appliedSrScaleForTests() == defaults.sr_scale,
          "start() pushed the requested SR scale into the preprocessor");

    uint64_t next_id = 1;

    // --- Phase A: both workers record the DEFAULT band, and UI delivery runs.
    CHECK(pumpUntil(controller, next_id, [&controller] {
              return controller.lastTrackingDepthBandForTests().generation >= 1 &&
                     controller.lastIntegrationDepthBandForTests().generation >= 1 &&
                     PipelineController::uiFrameDeliveryCountForTests() >= 1;
          }),
          "tracking, integration and UI delivery all happened on the default band");

    {
        const auto t = controller.lastTrackingDepthBandForTests();
        const auto i = controller.lastIntegrationDepthBandForTests();
        CHECK(t.valid && near(t.min_depth, 0.30f) && near(t.max_depth, 2.50f),
              "tracking worker used the default depth band");
        CHECK(i.valid && near(i.min_depth, 0.30f) && near(i.max_depth, 2.50f),
              "integration worker used the default depth band");
    }
    CHECK(!hook_saw_callback_lock.load(),
          "UI subscriber never ran while callback_mutex_ was held");

    // --- Phase B: change hyperparameters WHILE RUNNING.
    FusionHyperparams changed  = defaults;
    changed.min_depth          = 0.45f;
    changed.max_depth          = 1.60f;
    changed.sr_scale           = 4;
    const uint64_t track_gen_before = controller.lastTrackingDepthBandForTests().generation;
    const uint64_t integ_gen_before = controller.lastIntegrationDepthBandForTests().generation;

    controller.setHyperparams(changed);

    const FusionHyperparams applied = controller.hyperparamsSnapshot();
    CHECK(near(applied.min_depth, 0.45f) && near(applied.max_depth, 1.60f) &&
              applied.sr_scale == 4,
          "hyperparamsSnapshot() reflects the live setHyperparams()");
    CHECK(controller.appliedSrScaleForTests() == 4,
          "live setHyperparams() reached the preprocessor SR scale");

    // --- Phase C: both workers pick up the NEW band on later frames.
    CHECK(pumpUntil(controller, next_id, [&] {
              return controller.lastTrackingDepthBandForTests().generation >
                         track_gen_before &&
                     controller.lastIntegrationDepthBandForTests().generation >
                         integ_gen_before;
          }),
          "both workers processed another frame after the change");
    {
        const auto t = controller.lastTrackingDepthBandForTests();
        const auto i = controller.lastIntegrationDepthBandForTests();
        CHECK(t.valid && near(t.min_depth, 0.45f) && near(t.max_depth, 1.60f),
              "tracking worker picked up the new depth band while running");
        CHECK(i.valid && near(i.min_depth, 0.45f) && near(i.max_depth, 1.60f),
              "integration worker picked up the new depth band while running");
    }
    // No lock-freeness probe while the pipeline runs: a worker legitimately
    // holds tracker_mutex_ mid-track and tsdf_mutex_ mid-integrate, so a probe
    // here reports a busy lock as a leak (reproduced at OMP_NUM_THREADS=2).
    // Phase E probes the same locks after stop() joined every worker, which is
    // the only point where a held lock can only mean a leak.

    // --- Phase D: thread-count request while running is parked, then applied.
    controller.setNumThreads(2);
    const int right_after = controller.appliedThreadCountForTests();
    CHECK(right_after == 1 || right_after == 2,
          "running setNumThreads() only ever leaves a requested value applied");
    CHECK(pumpUntil(controller, next_id,
                    [&controller] { return controller.appliedThreadCountForTests() == 2; }),
          "parked thread count is applied at a frame boundary while running");

    // --- Phase E: shutdown leaves every guarded lock free.
    controller.stop();
    CHECK(!controller.isRunning(), "isRunning() false after stop() returns");
    CHECK(controller.state() == PipelineState::Stopped, "state() == Stopped after stop()");
    CHECK(controller.trackerMutexIsFreeForTests(), "tracker_mutex_ free after stop()");
    CHECK(controller.tsdfMutexIsFreeForTests(), "tsdf_mutex_ free after stop()");
    CHECK(controller.callbackMutexIsFreeForTests(), "callback_mutex_ free after stop()");

    const int delivery_count = PipelineController::uiFrameDeliveryCountForTests();
    PipelineController::resetUiFrameTestStateForTests();

    std::printf(
        "HYPERPARAMS-CONTRACT frames=%llu track_gen=%llu integ_gen=%llu "
        "deliveries=%d sr_scale=%d threads=%d\n",
        static_cast<unsigned long long>(next_id - 1),
        static_cast<unsigned long long>(
            controller.lastTrackingDepthBandForTests().generation),
        static_cast<unsigned long long>(
            controller.lastIntegrationDepthBandForTests().generation),
        delivery_count,
        controller.appliedSrScaleForTests(),
        controller.appliedThreadCountForTests());

    if (g_failures == 0) {
        std::printf(
            "pipeline_hyperparams_contract: PASS (per-frame depth-band "
            "propagation, live setHyperparams, parked setNumThreads, guarded "
            "callbacks, real controller, no sensor)\n");
        return 0;
    }
    std::printf("pipeline_hyperparams_contract: FAIL (%d checks failed)\n", g_failures);
    return 1;
}
