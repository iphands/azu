// pipeline_stall_visibility_contract (big-fix-two T0.3): a starved pipeline must
// LOOK starved. The "few frames then hangs" bug stayed invisible because
// capture_fps was an instantaneous 1/dt written only when a frame arrived, so
// it froze at ~30 fps after the last frame while the state still said Running.
//
// Real PipelineController through the seam (no device):
//   A  frames injected at ~30 Hz for 1.5 s read as 20..40 fps, not stalled
//   B  after 1.5 s with no frames: capture_fps == 0, sensor_stalled, and
//      seconds_since_frame >= 1
//   C  a session that never receives a frame is stalled after 1 s too
//   D  after stop(), nothing reports a stall
#include "app/PipelineController.h"
#include "sensor/KinectSensor.h"

#include <QCoreApplication>

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "pipeline_stall_visibility_contract needs AZU_PIPELINE_TEST_SEAM"
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
using kfusion::sensor::RawFrame;
using namespace std::chrono_literals;

std::shared_ptr<RawFrame> frame(uint64_t id) {
    auto f = std::make_shared<RawFrame>();
    f->frame_id    = id;
    f->depth_valid = true;
    f->rgb_valid   = true;
    for (auto& d : f->depth) d = 700;
    return f;
}

}  // namespace

int main() {
    CHECK(qApp == nullptr, "seam precondition: no QApplication");
    PipelineController pc(kfusion::sensor::PreprocessBackend::CPU);

    CHECK(pc.startWithoutSensorForTests(), "A: seam start");
    uint64_t id = 1;
    const auto t_end = std::chrono::steady_clock::now() + 1500ms;
    while (std::chrono::steady_clock::now() < t_end) {
        pc.injectRawFrameForTests(frame(id++));
        pc.metricsSnapshot();   // the GUI polls; rates are measured on read
        std::this_thread::sleep_for(33ms);
    }
    const auto a = pc.metricsSnapshot();
    std::printf("  A: capture %.1f fps, stalled %d\n", a.capture_fps, a.sensor_stalled);
    CHECK(a.capture_fps > 20.0f && a.capture_fps < 40.0f, "A: ~30 Hz input reads as ~30 fps");
    CHECK(!a.sensor_stalled, "A: a fed pipeline is not stalled");

    std::this_thread::sleep_for(1500ms);
    pc.metricsSnapshot();
    const auto b = pc.metricsSnapshot();
    std::printf("  B: capture %.1f fps, stalled %d, %.2f s since frame\n", b.capture_fps,
                b.sensor_stalled, b.seconds_since_frame);
    CHECK(b.capture_fps == 0.0f, "B: capture_fps decays to 0 when frames stop");
    CHECK(b.sensor_stalled, "B: the stall is flagged");
    CHECK(b.seconds_since_frame >= 1.0f, "B: the stall clock counts from the last frame");
    pc.stop();

    CHECK(pc.startWithoutSensorForTests(), "C: second seam start");
    std::this_thread::sleep_for(1200ms);
    const auto c = pc.metricsSnapshot();
    CHECK(c.sensor_stalled && c.capture_fps == 0.0f, "C: no frame at all since start() is a stall");
    pc.stop();

    const auto d = pc.metricsSnapshot();
    CHECK(!d.sensor_stalled, "D: a stopped pipeline is not reported as stalled");

    if (g_failures == 0) {
        std::printf("pipeline_stall_visibility_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("pipeline_stall_visibility_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
