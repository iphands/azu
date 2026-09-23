// kinect_pairing_realtrace_contract: regression test for the "captures a few
// frames then hangs" bug (c311c69). Real Kinect v1 clocks run depth and RGB at
// slightly different periods (2,002,155 vs 2,000,287 ticks of the 60 MHz
// counter), so their phase slips through a full cycle every ~35.7 s. A pairing
// policy has to publish every depth frame at EVERY phase, not just the phases a
// hand-written fixture happens to pick.
//
//   A  replay tests/data/kinect_ts_trace.txt (real device, crosses a wrap)
//   B  synthetic real-period streams, 1100 frames (> one slip cycle) at several
//      start phases, plus one run starting 2 s before the uint32 wrap
//   C  an RGB outage (samples 100..200 missing) still publishes every depth
//      frame; exactly the frames inside the outage go out depth-only
//
// Asserts per run: >= 99.9% of depth frames published, longest run of
// consecutive depth-only frames <= 1 outside an outage, every pair within the
// 17 ms window, ids strictly sequential.
#include "sensor/KinectSensor.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "kinect_pairing_realtrace_contract needs AZU_PIPELINE_TEST_SEAM"
#endif
#ifndef AZU_TEST_DATA_DIR
#error "AZU_TEST_DATA_DIR must point at tests/data"
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

using Sensor = kfusion::sensor::KinectSensor;
using Frame  = kfusion::sensor::RawFrame;

struct Event {
    bool     depth;
    uint32_t ticks;
};

struct Stats {
    size_t depth_in       = 0;
    size_t published      = 0;
    size_t paired         = 0;
    size_t depth_only     = 0;
    size_t max_only_run   = 0;
    double max_abs_ms     = 0.0;
    bool   ids_sequential = true;
    std::vector<uint32_t> depth_only_ticks;
};

Stats replay(const std::vector<Event>& events) {
    static const std::vector<uint16_t> depth(kfusion::sensor::DEPTH_WIDTH * kfusion::sensor::DEPTH_HEIGHT, 800);
    static const std::vector<uint8_t>  rgb(kfusion::sensor::RGB_WIDTH * kfusion::sensor::RGB_HEIGHT * 3, 128);

    Stats s;
    size_t only_run = 0;
    uint64_t last_id = 0;
    kfusion::sensor::KinectSensor sensor;
    sensor.setFrameCallback([&](std::shared_ptr<Frame> f) {
        ++s.published;
        if (f->frame_id != last_id + 1) s.ids_sequential = false;
        last_id = f->frame_id;
        if (f->rgb_valid) {
            ++s.paired;
            only_run = 0;
            s.max_abs_ms = std::max(s.max_abs_ms,
                                    std::fabs(Sensor::tickDeltaMs(f->depth_ticks, f->rgb_ticks)));
        } else {
            ++s.depth_only;
            s.max_only_run = std::max(s.max_only_run, ++only_run);
            s.depth_only_ticks.push_back(f->depth_ticks);
        }
    });
    for (const Event& e : events) {
        if (e.depth) {
            ++s.depth_in;
            sensor.injectDepthForTests(depth.data(), e.ticks);
        } else {
            sensor.injectRgbForTests(rgb.data(), e.ticks);
        }
    }
    sensor.setFrameCallback(nullptr);
    return s;
}

void expectHealthy(const Stats& s, const std::string& who, bool allow_outage = false) {
    // The final depth frame may legitimately still be waiting for its RGB.
    CHECK(s.published + 1 >= s.depth_in &&
              static_cast<double>(s.published) >= 0.999 * static_cast<double>(s.depth_in) - 1.0,
          who + ": >= 99.9% of depth frames published (" + std::to_string(s.published) + "/" +
              std::to_string(s.depth_in) + ")");
    if (!allow_outage) {
        CHECK(s.max_only_run <= 1, who + ": no run of consecutive depth-only frames (" +
                                       std::to_string(s.max_only_run) + ")");
        CHECK(s.paired >= s.published - s.published / 100,
              who + ": >= 99% of published frames carry RGB (" + std::to_string(s.paired) + ")");
    }
    CHECK(s.max_abs_ms <= Sensor::kMaxColorSkewMs + 1e-9,
          who + ": every pair is inside the window (max " + std::to_string(s.max_abs_ms) + " ms)");
    CHECK(s.ids_sequential, who + ": frame ids strictly sequential");
}

// Interleave two free-running clocks in true time order; emit the uint32 value
// libfreenect would report.
std::vector<Event> synth(uint64_t depth_start, uint64_t rgb_start, size_t frames,
                         size_t drop_rgb_from = 0, size_t drop_rgb_to = 0) {
    constexpr uint64_t kDepthPeriod = 2002155;
    constexpr uint64_t kRgbPeriod   = 2000287;
    std::vector<Event> ev;
    size_t di = 0, ri = 0;
    while (di < frames) {
        const uint64_t td = depth_start + di * kDepthPeriod;
        const uint64_t tr = rgb_start + ri * kRgbPeriod;
        if (tr <= td) {
            if (ri < drop_rgb_from || ri >= drop_rgb_to) ev.push_back({false, static_cast<uint32_t>(tr)});
            ++ri;
        } else {
            ev.push_back({true, static_cast<uint32_t>(td)});
            ++di;
        }
    }
    return ev;
}

void sectionRealTrace() {
    const std::string path = std::string(AZU_TEST_DATA_DIR) + "/kinect_ts_trace.txt";
    std::ifstream in(path);
    CHECK(in.good(), "trace fixture readable: " + path);
    std::vector<Event> ev;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        char kind = 0;
        unsigned long ticks = 0;
        if (ls >> kind >> ticks) ev.push_back({kind == 'D', static_cast<uint32_t>(ticks)});
    }
    CHECK(ev.size() == 1200, "trace fixture has 1200 events");
    const Stats s = replay(ev);
    std::printf("  real trace: %zu/%zu published, %zu paired, max |d| %.2f ms\n",
                s.published, s.depth_in, s.paired, s.max_abs_ms);
    expectHealthy(s, "real trace");
}

void sectionSlipCycle() {
    constexpr uint64_t kPhaseStep = 2002155 / 6;
    for (uint64_t p = 0; p < 6; ++p) {
        const uint64_t base = 1000000000ull;
        const Stats s = replay(synth(base + p * kPhaseStep, base, 1100));
        expectHealthy(s, "phase " + std::to_string(p));
    }
    // Start 2 s before the counter wraps.
    const uint64_t pre_wrap = (1ull << 32) - 120000000ull;
    const Stats w = replay(synth(pre_wrap, pre_wrap + 1188000, 1100));
    std::printf("  wrap run: %zu/%zu published, %zu paired\n", w.published, w.depth_in, w.paired);
    expectHealthy(w, "wrap run");
}

void sectionRgbOutage() {
    const uint64_t base = 1000000000ull;
    const Stats s = replay(synth(base + 700000, base, 400, 100, 201));
    expectHealthy(s, "rgb outage", /*allow_outage=*/true);
    // RGB samples 100..200 cover [base + 100*P_rgb, base + 200*P_rgb]; only depth
    // frames farther than 17 ms from every surviving RGB sample may go out bare.
    const uint64_t lo = base + 99 * 2000287ull + 17 * 60000ull;
    const uint64_t hi = base + 201 * 2000287ull - 17 * 60000ull;
    bool all_inside = true;
    for (uint32_t t : s.depth_only_ticks) {
        if (t < static_cast<uint32_t>(lo) || t > static_cast<uint32_t>(hi)) all_inside = false;
    }
    CHECK(all_inside, "rgb outage: depth-only frames lie inside the outage");
    CHECK(s.depth_only >= 95 && s.depth_only <= 101,
          "rgb outage: ~100 frames go out depth-only (" + std::to_string(s.depth_only) + ")");
}

}  // namespace

int main() {
    sectionRealTrace();
    sectionSlipCycle();
    sectionRgbOutage();
    if (g_failures == 0) {
        std::printf("kinect_pairing_realtrace_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("kinect_pairing_realtrace_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
