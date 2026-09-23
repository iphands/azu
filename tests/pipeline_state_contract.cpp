// pipeline_state_contract (big-fix todo 25): bounded, deterministic, CPU-only
// reset/start-state and queue-backpressure contract against the REAL
// src/app/PipelineController.cpp through the AZU_PIPELINE_TEST_SEAM. It proves,
// with no device, display, GPU, or filesystem access:
//   * start() clears the motion model, so the FIRST prediction of a session is
//     built from a cleared last_pose_ (never one retained across stop/start);
//   * reset() clears it too and zeroes the metrics it owns;
//   * the raw queue is retain-latest: under deterministic backpressure (the
//     tracking worker parked at a seam gate, holding no lock) the frames kept are
//     the NEWEST ones, the frames discarded are the oldest, and the eviction is
//     counted;
//   * the integration queue is retain-latest under the same gate discipline, and
//     its own drop counter moves;
//   * every drop counter is observable through metricsSnapshot(), and stop()
//     leaves metrics_mutex_ free.
// Every wait is decided by a counter the production loops advance (tracking /
// integration generations, popped frame ids, seam park flags). sleep_for only
// pumps work; CTest TIMEOUT is the hang detector.

#include "app/PipelineController.h"

#include <QCoreApplication>

#include <chrono>
#include <cstdio>
#include <cmath>
#include <functional>
#include <memory>
#include <thread>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "pipeline_state_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

using namespace std::chrono;
using kfusion::app::PipelineController;
using kfusion::app::PipelineMetrics;
using kfusion::app::PipelineState;
using kfusion::sensor::DEPTH_HEIGHT;
using kfusion::sensor::DEPTH_WIDTH;
using kfusion::sensor::PreprocessBackend;
using kfusion::sensor::RawFrame;

// Raw/integration queue capacity, mirrored from PipelineController's
// kRawQueueCapacity / kIntegrationQueueCapacity. A copy is required here: the
// constants are private, and the contract below is stated in terms of them.
constexpr size_t kQueueCapacity = 3;

constexpr int kMaxPumpFrames = 16;
constexpr auto kFrameSpacing = std::chrono::milliseconds(400);

// Paraboloid, not a plane: a flat field makes the real ICP lose tracking after
// frame 1, integration stops, and the observations this contract needs never
// arrive (see .omo/notepads/big-fix/learnings.md, todo 24).
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
            int raw = 700 + static_cast<int>(90.0f * (u * u + v * v));
            if (raw < 1)   raw = 1;
            if (raw > 950) raw = 950;
            f->depth[idx]       = static_cast<uint16_t>(raw);
            f->rgb[idx * 3 + 0] = static_cast<uint8_t>((x * 3 + id) & 0xFF);
            f->rgb[idx * 3 + 1] = static_cast<uint8_t>((y * 5) & 0xFF);
            f->rgb[idx * 3 + 2] = static_cast<uint8_t>((x + y) & 0xFF);
        }
    }
    return f;
}

int g_failures = 0;

#define CHECK(cond, what)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL: %s  [%s:%d]\n", (what), __FILE__, __LINE__);    \
            ++g_failures;                                                      \
        }                                                                      \
    } while (false)

bool near(float a, float b) { return std::fabs(a - b) < 1e-4f; }

bool isIdentity(const Eigen::Matrix4f& m) {
    const Eigen::Matrix4f id = Eigen::Matrix4f::Identity();
    return (m - id).cwiseAbs().maxCoeff() < 1e-6f;
}

// Keeps the pipeline fed until `done` holds, bounded by injected-frame count.
bool pumpUntil(PipelineController& c, uint64_t& next_id,
               const std::function<bool()>& done,
               int max_frames = kMaxPumpFrames) {
    for (int injected = 0; injected < max_frames && !done(); ++injected) {
        c.injectRawFrameForTests(makeSyntheticFrame(next_id++));
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

    uint64_t next_id = 1;
    Eigen::Matrix4f session_one_pose = Eigen::Matrix4f::Identity();

    // ============================================================
    // Phase A: a session that tracks, so a stale last_pose_ EXISTS to be
    // carried into the next session.
    // ============================================================
    CHECK(controller.startWithoutSensorForTests(), "session 1 seam start");
    CHECK(pumpUntil(controller, next_id, [&controller] {
              return controller.lastMotionModelForTests().generation >= 2 &&
                     controller.lastIntegrationDepthBandForTests().generation >= 2;
          }),
          "session 1 produced tracked predictions and integrations");
    session_one_pose = controller.currentPose();
    CHECK(!isIdentity(session_one_pose),
          "session 1 actually moved the pose (premise: a stale last_pose_ exists "
          "for a start() that fails to clear the motion model)");
    controller.stop();
    CHECK(controller.state() == PipelineState::Stopped, "state Stopped after stop()");

    // ============================================================
    // Phase B: start() alone (NO reset in between) must clear the motion model.
    // The tracking worker is parked BEFORE the start, so between start() and the
    // resume nothing can touch last_pose_ except startInternal() itself: the read
    // below is ordering-deterministic, not a race with a worker.
    // ============================================================
    controller.pauseTrackingWorkerForTests();
    CHECK(controller.startWithoutSensorForTests(), "session 2 seam start");
    CHECK(pumpUntil(controller, next_id,
                    [&controller] { return controller.trackingWorkerParkedForTests(); },
                    1),
          "tracking worker reached the seam gate");
    CHECK(isIdentity(controller.lastPoseForTests()),
          "start() cleared last_pose_ without requiring a prior reset()");
    CHECK(isIdentity(controller.currentPose()),
          "start() cleared current_pose_");

    const uint64_t motion_gen_before_restart =
        controller.lastMotionModelForTests().generation;
    controller.resumePipelineWorkersForTests();
    CHECK(pumpUntil(controller, next_id, [&controller, motion_gen_before_restart] {
              return controller.lastMotionModelForTests().generation >
                     motion_gen_before_restart;
          }),
          "session 2 produced its first prediction");
    {
        const auto m = controller.lastMotionModelForTests();
        CHECK(m.valid, "motion observation is valid");
        CHECK(isIdentity(m.last_pose_before),
              "the FIRST prediction of the new session saw a cleared motion "
              "model, not the pose retained across stop/start");
        CHECK(near((m.predicted_pose - m.prev_pose).cwiseAbs().maxCoeff(), 0.0f),
              "with a cleared model the first prediction has zero delta "
              "(predicted == prev), i.e. no jump across the restart gap");
    }
    controller.stop();

    // ============================================================
    // Phase C: reset() clears the pose pair and the metrics it owns.
    // ============================================================
    controller.reset();
    CHECK(controller.state() == PipelineState::Idle, "state Idle after reset()");
    CHECK(isIdentity(controller.lastPoseForTests()),
          "reset() cleared last_pose_");
    {
        const PipelineMetrics m = controller.metricsSnapshot();
        CHECK(m.frame_count == 0 && m.dropped_frames == 0 &&
                  m.dropped_integration_frames == 0 &&
                  m.dropped_pre_model_frames == 0,
              "reset() zeroed frame_count and every drop counter under "
              "metrics_mutex_");
    }

    // ============================================================
    // Phase D: raw queue is RETAIN-LATEST under deterministic backpressure.
    // The tracking worker is parked at the gate and holds no lock, so the queue
    // fills and the eviction policy is fully determined: capacity 3, six frames
    // in, exactly three evictions, and they are the three OLDEST ids.
    // ============================================================
    controller.pauseTrackingWorkerForTests();
    CHECK(controller.startWithoutSensorForTests(), "session 3 seam start");
    CHECK(pumpUntil(controller, next_id,
                    [&controller] { return controller.trackingWorkerParkedForTests(); },
                    1),
          "tracking worker parked before the burst");

    const PipelineMetrics before_burst = controller.metricsSnapshot();
    // Under load the pump above can inject one frame before the worker parks;
    // it then sits in the queue ahead of the burst. Count it instead of
    // assuming an empty queue (that assumption made this test flaky).
    const int queued_before = static_cast<int>(controller.rawQueueDepthForTests());
    constexpr int kBurst = 6;
    for (int i = 0; i < kBurst; ++i) {
        controller.injectRawFrameForTests(makeSyntheticFrame(next_id++));
    }
    const uint64_t last_id = next_id - 1;
    const int kExcess = queued_before + kBurst - static_cast<int>(kQueueCapacity);

    CHECK(controller.rawQueueDepthForTests() == kQueueCapacity,
          "the raw queue stayed bounded at capacity under a burst");
    {
        const PipelineMetrics m = controller.metricsSnapshot();
        CHECK(m.dropped_frames == before_burst.dropped_frames + kExcess,
              "exactly the over-capacity frames were dropped, and every drop is "
              "reported by metricsSnapshot()");
        CHECK(m.frame_count == before_burst.frame_count + kBurst,
              "every arriving frame was counted as received, dropped or not");
    }

    controller.resumePipelineWorkersForTests();
    const uint64_t oldest_retained = last_id - kQueueCapacity + 1;
    CHECK(pumpUntil(controller, next_id, [&controller, oldest_retained, last_id] {
              const uint64_t popped = controller.lastPoppedFrameIdForTests();
              return popped >= oldest_retained && popped >= last_id;
          }),
          "the newest injected frame was the one the pipeline went on to "
          "process, and no frame older than the retained window ever was");
    CHECK(controller.lastPoppedFrameIdForTests() >= oldest_retained,
          "no stale (evicted) frame was ever handed to the tracking worker");

    // ============================================================
    // Phase E: integration queue is retain-latest too, with its own counter.
    // Park the integration worker; then every frame the tracking loop accepts
    // piles into the integration queue. Even discounting a full queue plus a
    // full tracking-lost flush, the pushes left MUST overflow it, and the
    // integration drop counter must move.
    // ============================================================
    controller.pauseIntegrationWorkerForTests();
    CHECK(pumpUntil(controller, next_id,
                    [&controller] { return controller.integrationWorkerParkedForTests(); },
                    2),
          "integration worker parked");
    const PipelineMetrics before_park  = controller.metricsSnapshot();
    const size_t    depth_at_park      = controller.integrationQueueDepthForTests();
    const uint64_t  track_gen_at_park  = controller.lastTrackingDepthBandForTests().generation;

    CHECK(pumpUntil(controller, next_id, [&controller, track_gen_at_park] {
              return controller.lastTrackingDepthBandForTests().generation >=
                     track_gen_at_park + 10;
          }),
          "ten more frames went through the tracking loop while integration was "
          "parked");
    const uint64_t pushed_window = controller.lastTrackingDepthBandForTests().generation -
                                   track_gen_at_park;
    {
        const PipelineMetrics m    = controller.metricsSnapshot();
        const int             moved = m.dropped_integration_frames -
                                      before_park.dropped_integration_frames;
        const int             floor = int(pushed_window) - 2 * int(kQueueCapacity);
        CHECK(moved >= floor && moved > 0,
              "the integration queue overflowed under a parked consumer and the "
              "overflow is observable through metricsSnapshot()");
        CHECK(m.dropped_frames >= before_burst.dropped_frames,
              "raw-frame accounting never went backwards");
        std::printf(
            "STATE phase-E depth_at_park=%zu pushed=%llu integration_drops=+%d\n",
            depth_at_park, static_cast<unsigned long long>(pushed_window), moved);
    }
    controller.resumePipelineWorkersForTests();

    // ============================================================
    // Phase F: shutdown leaves the metrics lock free and the counters readable.
    // ============================================================
    controller.stop();
    CHECK(!controller.isRunning(), "isRunning() false after stop()");
    CHECK(controller.state() == PipelineState::Stopped, "state Stopped after stop()");
    CHECK(controller.metricsMutexIsFreeForTests(), "metrics_mutex_ free after stop()");
    CHECK(controller.callbackMutexIsFreeForTests(), "callback_mutex_ free after stop()");
    CHECK(controller.trackerMutexIsFreeForTests(), "tracker_mutex_ free after stop()");
    CHECK(controller.tsdfMutexIsFreeForTests(), "tsdf_mutex_ free after stop()");
    CHECK(controller.rawQueueDepthForTests() <= kQueueCapacity &&
              controller.integrationQueueDepthForTests() <= kQueueCapacity,
          "both queues stayed within capacity for the whole run");

    const PipelineMetrics final_metrics = controller.metricsSnapshot();
    PipelineController::resetUiFrameTestStateForTests();

    std::printf(
        "STATE-CONTRACT injected=%llu popped=%llu dropped_raw=%d "
        "dropped_integration=%d dropped_pre_model=%d pose_moved=%d\n",
        static_cast<unsigned long long>(next_id - 1),
        static_cast<unsigned long long>(controller.lastPoppedFrameIdForTests()),
        final_metrics.dropped_frames, final_metrics.dropped_integration_frames,
        final_metrics.dropped_pre_model_frames,
        isIdentity(session_one_pose) ? 0 : 1);

    if (g_failures == 0) {
        std::printf(
            "pipeline_state_contract: PASS (start/reset clear the motion model, "
            "retain-latest queues, counted drops through metricsSnapshot(), real "
            "controller, no sensor)\n");
        return 0;
    }
    std::printf("pipeline_state_contract: FAIL (%d checks failed)\n", g_failures);
    return 1;
}
