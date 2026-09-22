// pipeline_stop_contract (big-fix todo 9): bounded, deterministic, CPU-only
// start/stop stress test against the REAL src/app/PipelineController.cpp
// through the AZU_PIPELINE_TEST_SEAM. It drives 50 repeated start -> (optionally
// inject one synthetic frame) -> stop cycles on a SINGLE controller instance and
// asserts, after every stop(), that stop() returned (no hang), isRunning() is
// false, state() == Stopped, and the controller can be restarted. Correctness is
// decided ONLY by those post-conditions; a lost wakeup deadlocks stop() and the
// CTest TIMEOUT kills the process. No sleep/timing is used to prove success, and
// there is no device, display, GPU, or filesystem access.

#include "app/PipelineController.h"

#include <QCoreApplication>

#include <chrono>
#include <cstdio>
#include <memory>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "pipeline_stop_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

using namespace std::chrono;
using kfusion::app::PipelineController;
using kfusion::app::PipelineState;
using kfusion::sensor::DEPTH_HEIGHT;
using kfusion::sensor::DEPTH_WIDTH;
using kfusion::sensor::PreprocessBackend;
using kfusion::sensor::RawFrame;

// 50 bounded cycles. Cycles where injectThisCycle() is true exercise the
// non-empty-queue shutdown path (a frame is queued before stop()); the rest
// stop immediately after start() to exercise the empty-queue path, which is
// exactly where the lost-wakeup deadlock lived.
constexpr int kCycles = 50;

constexpr bool injectThisCycle(int cycle) { return (cycle % 7) == 3; }

int g_failures = 0;
int g_injected_cycles = 0;

#define CHECK(cond, what)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL: %s  [%s:%d]\n", (what), __FILE__, __LINE__);    \
            ++g_failures;                                                      \
        }                                                                      \
    } while (false)

// Deterministic synthetic frame: a frontal plane at ~0.85 m inside the default
// [0.3, 2.5] depth band, plus an RGB gradient. No device, GPU, or randomness.
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

    // One controller instance for all cycles: this is what proves the SAME
    // object survives repeated start()/stop() (worker predicate cleared and
    // threads relaunched by startInternal, not a fresh construction each time).
    PipelineController controller(PreprocessBackend::CPU);

    const auto run_t0 = steady_clock::now();
    long long slowest_cycle_ms = 0;

    for (int cycle = 0; cycle < kCycles; ++cycle) {
        const auto cycle_t0 = steady_clock::now();

        CHECK(controller.startWithoutSensorForTests(),
              "startWithoutSensorForTests() returns true on restart");
        CHECK(controller.isRunning(),
              "controller reports Running after seam start");

        if (injectThisCycle(cycle)) {
            controller.injectRawFrameForTests(makeSyntheticFrame(cycle + 1));
            ++g_injected_cycles;
        }

        // stop() must return. If the shutdown notification is lost, this call
        // blocks forever and the CTest TIMEOUT (not any sleep here) fails it.
        controller.stop();

        CHECK(!controller.isRunning(),
              "isRunning() false immediately after stop() returns");
        CHECK(controller.state() == PipelineState::Stopped,
              "state() == Stopped after stop()");

        const auto cycle_ms =
            duration_cast<milliseconds>(steady_clock::now() - cycle_t0).count();
        if (cycle_ms > slowest_cycle_ms) slowest_cycle_ms = cycle_ms;
    }

    const auto total_ms =
        duration_cast<milliseconds>(steady_clock::now() - run_t0).count();

    std::printf(
        "STOP-CONTRACT cycles=%d injected_cycles=%d empty_stop_cycles=%d "
        "slowest_cycle_ms=%lld total_ms=%lld\n",
        kCycles, g_injected_cycles, kCycles - g_injected_cycles,
        static_cast<long long>(slowest_cycle_ms),
        static_cast<long long>(total_ms));

    if (g_failures == 0) {
        std::printf(
            "pipeline_stop_contract: PASS (%d start/stop cycles, real controller, "
            "no hang, no sensor)\n",
            kCycles);
        return 0;
    }
    std::printf("pipeline_stop_contract: FAIL (%d checks failed)\n", g_failures);
    return 1;
}
