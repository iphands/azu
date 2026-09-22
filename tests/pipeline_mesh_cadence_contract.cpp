// pipeline_mesh_cadence_contract (big-fix todo 25): bounded, deterministic,
// CPU-only mesh-request contract against the REAL src/app/PipelineController.cpp
// through the AZU_PIPELINE_TEST_SEAM. The cadence interval is pinned through the
// seam so the ONLY automatic request per start is the bootstrap one, and every
// other request is issued explicitly and versioned. It proves, with no device,
// display, GPU, or filesystem access:
//   * the integration loop's time-based cadence issues a real versioned request
//     and the worker serves it, without any requester waiting;
//   * a request that arrives while an older extraction is in flight is neither
//     lost nor merged: both versions are extracted, in order, each publish
//     tagged with the version it answers;
//   * a result extracted against a superseded (reset) generation is dropped, not
//     published, while a post-reset request is still served;
//   * a request that is pending across reset() is drained and never executed
//     after the restart;
//   * capture/tracking keep advancing while a mesh extraction is in flight.
// The oracle is always the version counters, the hook's claimed-version list and
// the extraction/publish/drop counters. sleep_for only pumps work; CTest TIMEOUT
// is the hang detector.

#include "app/PipelineController.h"

#include <QCoreApplication>

#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "pipeline_mesh_cadence_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

using namespace std::chrono;
using kfusion::app::PipelineController;
using kfusion::app::PipelineState;
using kfusion::sensor::DEPTH_HEIGHT;
using kfusion::sensor::DEPTH_WIDTH;
using kfusion::sensor::PreprocessBackend;
using kfusion::sensor::RawFrame;

// Long enough that the cadence contributes exactly ONE request per start (the
// bootstrap), so every later version in this test was requested explicitly and
// the version arithmetic is exact. Correctness never depends on this interval;
// Phase A proves the automatic path fires at all.
constexpr auto kHugeCadence = std::chrono::microseconds(3600ll * 1000000ll);

constexpr int kMaxPumpFrames = 24;
constexpr auto kFrameSpacing = std::chrono::milliseconds(300);

// Paraboloid fixture: a flat plane loses tracking after frame 1 and integration
// stops, so no cadence request and no mesh would ever be produced (see
// .omo/notepads/big-fix/learnings.md, todo 24).
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

// Keeps the pipeline fed until `done` holds, bounded by injected-frame count.
bool pumpUntil(PipelineController& c, uint64_t& next_id,
               const std::function<bool()>& done) {
    for (int injected = 0; injected < kMaxPumpFrames && !done(); ++injected) {
        c.injectRawFrameForTests(makeSyntheticFrame(next_id++));
        std::this_thread::sleep_for(kFrameSpacing);
    }
    return done();
}

std::string versions(const std::vector<uint64_t>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) {
        s += std::to_string(v[i]) + (i + 1 == v.size() ? "]" : ",");
    }
    return s;
}

} // namespace

int main() {
    CHECK(qApp == nullptr,
          "seam precondition: test runs with NO QApplication (null qApp)");

    PipelineController controller(PreprocessBackend::CPU);
    PipelineController::resetUiFrameTestStateForTests();
    controller.setMeshCadenceIntervalForTests(kHugeCadence);

    uint64_t next_id = 1;

    // ============================================================
    // Phase A: the automatic cadence issues a versioned request and a real
    // extraction serves it. Nothing in this phase requested a mesh by hand.
    // ============================================================
    CHECK(controller.startWithoutSensorForTests(), "session 1 seam start");
    CHECK(pumpUntil(controller, next_id, [&controller] {
              return controller.meshStateForTests().served >= 1;
          }),
          "the integration loop's cadence requested a mesh and the worker "
          "published it");
    {
        const auto s = controller.meshStateForTests();
        CHECK(s.requested == 1 && s.claimed == 1 && s.served == 1,
              "exactly one (bootstrap) request existed and it was served — the "
              "cadence is time-gated, not per-frame");
        CHECK(s.generation == 0 && s.served_generation == 0,
              "the published mesh is tagged with the current generation");
        CHECK(s.extractions == 1 && s.publishes == 1 && s.stale_drops == 0,
              "one request caused exactly one extraction and one publish");
    }

    // ============================================================
    // Phase B: a request arriving while an older extraction is IN FLIGHT is
    // neither lost nor merged. The worker is parked inside meshingLoop() (having
    // claimed its version, holding no volume lock), so "in flight" is exact.
    // ============================================================
    // Two holds: the worker parks once for v2 and again for v3, so the claimed
    // version list is a direct, in-order record of which request each extraction
    // answered.
    controller.armMeshExtractionHoldForTests(2);
    const uint64_t v_inflight = controller.requestMeshForTests();
    CHECK(v_inflight == 2, "the second request was assigned version 2");
    CHECK(pumpUntil(controller, next_id, [&controller] {
              return controller.meshHookVersionsForTests().size() == 1;
          }),
          "the worker claimed version 2 and is parked mid-extraction");
    CHECK(controller.meshHookVersionsForTests() == std::vector<uint64_t>{2},
          "the in-flight extraction is the one answering request v2");

    const uint64_t track_gen_before =
        controller.lastTrackingDepthBandForTests().generation;
    const uint64_t v_queued = controller.requestMeshForTests();
    CHECK(v_queued == 3, "the request issued during the in-flight extraction got "
                         "its own version 3");
    CHECK(pumpUntil(controller, next_id, [&controller, track_gen_before] {
              return controller.lastTrackingDepthBandForTests().generation >=
                     track_gen_before + 3;
          }),
          "tracking kept processing frames while a mesh extraction was in "
          "flight: the cadence never blocks capture or tracking");
    {
        const auto s = controller.meshStateForTests();
        CHECK(s.requested == 3 && s.claimed == 2,
              "the queued request stayed visible as pending work instead of "
              "being cleared behind the worker's back");
    }

    controller.releaseMeshExtractionHoldForTests();
    CHECK(pumpUntil(controller, next_id, [&controller] {
              return controller.meshHookVersionsForTests().size() == 2;
          }),
          "after finishing v2 the worker claimed v3 as its own extraction rather "
          "than discarding or folding it into v2");
    CHECK(controller.meshHookVersionsForTests() == std::vector<uint64_t>({2, 3}),
          "the queued request was claimed in request order");
    controller.releaseMeshExtractionHoldForTests();
    CHECK(pumpUntil(controller, next_id, [&controller] {
              return controller.meshStateForTests().served >= 3;
          }),
          "both the in-flight and the queued request were served");
    {
        const auto s        = controller.meshStateForTests();
        const auto claimed  = controller.meshHookVersionsForTests();
        CHECK(claimed == std::vector<uint64_t>({2, 3}),
              "each version was extracted once, in request order");
        CHECK(s.extractions == 3 && s.publishes == 3,
              "two requests issued against one in-flight extraction produced two "
              "extractions and two publishes — nothing lost, nothing merged");
        CHECK(s.served == 3 && s.claimed == 3 && s.requested == 3,
              "the published mesh is tagged with the newest answered version");
        CHECK(s.stale_drops == 0, "nothing was dropped before any reset happened");
    }

    // ============================================================
    // Phase C: a result whose generation was invalidated while it ran cannot be
    // published — but a post-reset request is still served.
    // ============================================================
    const auto before_invalidate = controller.meshStateForTests();
    controller.armMeshExtractionHoldForTests(1);
    const uint64_t v_stale = controller.requestMeshForTests();
    CHECK(v_stale == 4, "the pre-reset request got version 4");
    CHECK(pumpUntil(controller, next_id, [&controller] {
              return controller.meshHookVersionsForTests().size() == 3;
          }),
          "the worker claimed v4 and is parked");
    controller.invalidateMeshStateForTests();
    {
        const auto s = controller.meshStateForTests();
        CHECK(s.generation == before_invalidate.generation + 1,
              "invalidation advanced the mesh generation");
        CHECK(s.requested == s.claimed,
              "invalidation drained the pending requests, so no pre-reset work "
              "survives into the next generation");
    }
    controller.releaseMeshExtractionHoldForTests();
    CHECK(pumpUntil(controller, next_id, [&controller, before_invalidate] {
              return controller.meshStateForTests().stale_drops >=
                     before_invalidate.stale_drops + 1;
          }),
          "the extraction that ran against the dead generation was counted as a "
          "stale drop");
    {
        const auto s = controller.meshStateForTests();
        CHECK(s.publishes == before_invalidate.publishes,
              "no mesh from a superseded generation reached SharedMesh");
        CHECK(s.served == before_invalidate.served &&
                  s.served_generation == before_invalidate.served_generation,
              "the served (version, generation) tag was not overwritten by the "
              "stale result");
    }
    const uint64_t v_after_reset = controller.requestMeshForTests();
    CHECK(pumpUntil(controller, next_id, [&controller, v_after_reset] {
              return controller.meshStateForTests().served >= v_after_reset;
          }),
          "a request issued after the reset is still served");
    {
        const auto s = controller.meshStateForTests();
        CHECK(s.served == v_after_reset && s.served_generation == 1,
              "the post-reset publish is tagged with the NEW generation");
    }
    controller.stop();

    // ============================================================
    // Phase D: a request pending across reset() is drained, not replayed.
    // ============================================================
    const uint64_t v_pending = controller.requestMeshForTests();
    CHECK(v_pending == 6, "the request issued while stopped got version 6");
    controller.reset();
    {
        const auto s = controller.meshStateForTests();
        CHECK(s.requested == s.claimed && s.requested == v_pending,
              "reset() drained the pending request");
    }
    const auto extractions_before_restart = controller.meshStateForTests();
    CHECK(controller.startWithoutSensorForTests(), "session 2 seam start");
    CHECK(pumpUntil(controller, next_id, [&controller, v_pending] {
              return controller.meshStateForTests().served > v_pending;
          }),
          "the restarted session's own cadence request was served");
    {
        const auto s = controller.meshStateForTests();
        CHECK(s.extractions == extractions_before_restart.extractions + 1,
              "the drained pre-reset request never executed after the restart: "
              "exactly one extraction ran, the new session's own");
        CHECK(s.generation == 2 && s.served_generation == 2,
              "the restarted session runs on the generation reset() created: "
              "generation 0 at start, 1 after the Phase C invalidation, 2 after "
              "reset() — and the new publish is tagged with it");
    }

    // ============================================================
    // Phase E: shutdown. No worker exists after stop(), so nothing can publish.
    // ============================================================
    const auto before_stop = controller.meshStateForTests();
    controller.stop();
    CHECK(!controller.isRunning(), "isRunning() false after stop()");
    CHECK(controller.state() == PipelineState::Stopped, "state Stopped after stop()");
    {
        const auto s = controller.meshStateForTests();
        CHECK(s.publishes == before_stop.publishes && s.extractions ==
                  before_stop.extractions,
              "stop() left no worker that could still extract or publish");
        std::printf("MESH-CONTRACT published=%s stale_drops=%llu\n",
                    versions(controller.meshHookVersionsForTests()).c_str(),
                    static_cast<unsigned long long>(s.stale_drops));
    }

    PipelineController::resetUiFrameTestStateForTests();

    if (g_failures == 0) {
        std::printf(
            "pipeline_mesh_cadence_contract: PASS (time-based cadence, versioned "
            "requests, per-version publish tags, reset generation guard, real "
            "controller, no sensor)\n");
        return 0;
    }
    std::printf("pipeline_mesh_cadence_contract: FAIL (%d checks failed)\n",
                g_failures);
    return 1;
}
