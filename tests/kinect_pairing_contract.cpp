// kinect_pairing_contract: device-free contract for KinectSensor depth/RGB
// pairing. A REAL kfusion::sensor::KinectSensor (default-constructed: no init(),
// no libfreenect, no capture thread) is fed through the AZU_PIPELINE_TEST_SEAM
// injectors, which call the private onDepth()/onRgb() path directly.
//
// Timestamps are libfreenect's native unit: 60 MHz hardware-counter ticks
// (uint32, wraps every ~71.6 s). The previous contract injected microseconds,
// which is how the "50 ms" window that was really 0.83 ms got locked in.
//
// Policy under test (docs/CANONICAL_SEMANTICS.md, "Sensor pairing"):
//   1  depth-led: a depth frame takes the held RGB sample when |d| <= 17 ms
//   2  if the held RGB is > 17 ms older, the depth frame waits for the next RGB
//   3  a depth frame that finds no RGB within 17 ms is published depth-only
//      (rgb_valid == false) — geometry never waits on or is dropped for colour
//   4  |d| == 17 ms is accepted, 17 ms + 1 tick is not
//   5  pairing is correct across the uint32 counter wrap, and the unwrapped
//      millisecond timestamps stay monotonic
//   6  frame ids are sequential in capture order, published frames carry the
//      exact bytes injected for them, and publication happens outside the lock
//   7  with no callback the ready slot holds only the newest frame
//   8  the pool returns in full once references drop
//   9  each depth frame carries the mean accelerometer reading since the
//      previous one (the previous mean when none arrived; invalid before the
//      first; a zero vector is not a reading)
#include "sensor/KinectSensor.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "kinect_pairing_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
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

using Frame  = kfusion::sensor::RawFrame;
using Sensor = kfusion::sensor::KinectSensor;

constexpr size_t kDepthBytes = static_cast<size_t>(kfusion::sensor::DEPTH_WIDTH) *
                               kfusion::sensor::DEPTH_HEIGHT * sizeof(uint16_t);
constexpr size_t kRgbBytes = static_cast<size_t>(kfusion::sensor::RGB_WIDTH) *
                             kfusion::sensor::RGB_HEIGHT * 3u;
constexpr size_t kPoolSize = 8; // KinectSensor::POOL_SIZE (private)

constexpr uint32_t kTicksPerMs = 60000u;
constexpr uint32_t ms(double v) { return static_cast<uint32_t>(v * kTicksPerMs); }

struct Received {
    uint64_t               frame_id  = 0;
    bool                   lock_free = false;
    std::shared_ptr<Frame> keep;
};

std::vector<uint16_t> makeDepth(uint32_t ts) {
    std::vector<uint16_t> d(kDepthBytes / sizeof(uint16_t));
    for (size_t i = 0; i < d.size(); ++i) d[i] = static_cast<uint16_t>(1u + ((i + ts) % 2046u));
    return d;
}

std::vector<uint8_t> makeRgb(uint32_t ts) {
    std::vector<uint8_t> rgb(kRgbBytes);
    for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = static_cast<uint8_t>((i * 31u + ts) & 0xffu);
    return rgb;
}

class Fixture {
public:
    Fixture() { subscribe(); }
    ~Fixture() { sensor_.setFrameCallback(nullptr); }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;

    void subscribe() {
        sensor_.setFrameCallback([this](std::shared_ptr<Frame> f) {
            Received r;
            r.lock_free = sensor_.syncMutexIsFreeForTests();
            r.frame_id  = f->frame_id;
            r.keep      = std::move(f);
            published_.push_back(std::move(r));
        });
    }
    void unsubscribe() { sensor_.setFrameCallback(nullptr); }

    void depth(uint32_t ts) { sensor_.injectDepthForTests(depthPayload(ts).data(), ts); }
    void rgb(uint32_t ts) { sensor_.injectRgbForTests(rgbPayload(ts).data(), ts); }
    void accel(double x, double y, double z) { sensor_.ingestAccel(x, y, z); }

    size_t publishedCount() const { return published_.size(); }
    size_t readyCount() const { return sensor_.readyFrameCountForTests(); }
    size_t freeCount() const { return sensor_.freeFrameCountForTests(); }
    std::shared_ptr<Frame> latest() { return sensor_.getLatestFrame(); }
    void release() { published_.clear(); }

    // The n-th published frame (0-based) must be the given depth sample, paired
    // with `rgb_ts` or depth-only when rgb_ts == kNone.
    static constexpr uint64_t kNone = ~0ull;
    void expect(size_t n, uint32_t depth_ts, uint64_t rgb_ts, const std::string& what) {
        CHECK(published_.size() > n, what + ": frame published");
        if (published_.size() <= n) return;
        const Received& r = published_[n];
        const Frame& f = *r.keep;
        CHECK(r.frame_id == n + 1, what + ": sequential frame id");
        CHECK(r.lock_free, what + ": callback ran outside sync_mutex_");
        CHECK(f.depth_valid, what + ": depth valid");
        CHECK(f.depth_ticks == depth_ts, what + ": carries its depth tick stamp");
        CHECK(std::memcmp(f.depth.data(), depthPayload(depth_ts).data(), kDepthBytes) == 0,
              what + ": carries its own depth bytes");
        if (rgb_ts == kNone) {
            CHECK(!f.rgb_valid, what + ": published depth-only");
        } else {
            const uint32_t rts = static_cast<uint32_t>(rgb_ts);
            CHECK(f.rgb_valid, what + ": paired with RGB");
            CHECK(f.rgb_ticks == rts, what + ": paired with the expected RGB sample");
            CHECK(std::memcmp(f.rgb.data(), rgbPayload(rts).data(), kRgbBytes) == 0,
                  what + ": carries that RGB sample's bytes");
            CHECK(std::fabs(f.timestamp_depth - f.timestamp_rgb) <= Sensor::kMaxColorSkewMs,
                  what + ": unwrapped ms timestamps agree with the tick delta");
        }
    }
    const Frame& frame(size_t n) const { return *published_[n].keep; }

private:
    const std::vector<uint16_t>& depthPayload(uint32_t ts) {
        auto it = depth_cache_.find(ts);
        return it != depth_cache_.end() ? it->second : depth_cache_.emplace(ts, makeDepth(ts)).first->second;
    }
    const std::vector<uint8_t>& rgbPayload(uint32_t ts) {
        auto it = rgb_cache_.find(ts);
        return it != rgb_cache_.end() ? it->second : rgb_cache_.emplace(ts, makeRgb(ts)).first->second;
    }

    kfusion::sensor::KinectSensor             sensor_;
    std::vector<Received>                     published_;
    std::map<uint32_t, std::vector<uint16_t>> depth_cache_;
    std::map<uint32_t, std::vector<uint8_t>>  rgb_cache_;
};

void sectionUnits() {
    CHECK(Sensor::kTicksPerMs == 60000.0, "libfreenect clock is 60 MHz");
    CHECK(Sensor::kMaxColorSkewMs == 17.0, "pairing window is half a 30 Hz period");
    CHECK(Sensor::tickDeltaMs(ms(40), ms(10)) == 30.0, "tick delta in ms");
    CHECK(Sensor::tickDeltaMs(ms(10), ms(40)) == -30.0, "signed tick delta");
    CHECK(std::fabs(Sensor::tickDeltaMs(5u * kTicksPerMs, 0xFFFFFFFFu - 5u * kTicksPerMs + 1u) - 10.0) < 1e-9,
          "delta is wrap-safe");
}

void sectionPolicy() {
    Fixture fx;
    const uint32_t T = 1000u * kTicksPerMs;

    // 1. depth-led immediate pair.
    fx.rgb(T);
    CHECK(fx.publishedCount() == 0, "an RGB sample alone publishes nothing");
    fx.depth(T + ms(10));
    fx.expect(0, T + ms(10), T, "1 depth 10 ms after RGB");

    // 2. held RGB 25 ms old: depth waits, then takes the next (nearer) RGB.
    fx.rgb(T + ms(100));
    fx.depth(T + ms(125));
    CHECK(fx.publishedCount() == 1, "2 depth 25 ms after RGB waits");
    fx.rgb(T + ms(133.3));
    fx.expect(1, T + ms(125), T + ms(133.3), "2 waiting depth pairs with the next RGB");

    // 3. RGB stream stalls: a waiting depth frame is flushed depth-only by the next depth.
    fx.depth(T + ms(200));
    fx.depth(T + ms(233));
    fx.expect(2, T + ms(200), Fixture::kNone, "3 depth with no RGB goes out depth-only");
    // The next RGB is 16 ms after the waiting depth: pairs.
    fx.rgb(T + ms(249));
    fx.expect(3, T + ms(233), T + ms(249), "3 RGB resumes");

    // 4. boundary: exactly 17 ms is accepted, one more tick is not.
    fx.rgb(T + ms(300));
    fx.depth(T + ms(317));
    fx.expect(4, T + ms(317), T + ms(300), "4 |d| == 17 ms accepted");
    fx.rgb(T + ms(400));
    fx.depth(T + ms(417) + 1u);
    CHECK(fx.publishedCount() == 5, "4 17 ms + 1 tick waits for the next RGB");
    fx.rgb(T + ms(450));
    fx.expect(5, T + ms(417) + 1u, Fixture::kNone, "4 next RGB 33 ms later is too far: depth-only");

    // 5. a waiting depth frame met by an RGB > 17 ms away goes out depth-only,
    //    and that RGB is kept for the following depth frame.
    fx.depth(T + ms(460));
    fx.expect(6, T + ms(460), T + ms(450), "5 held RGB 10 ms before depth");

    CHECK(fx.publishedCount() == 7, "exactly seven frames published");
    fx.release();
}

void sectionWrap() {
    Fixture fx;
    const uint32_t near_wrap = 0xFFFFFFFFu - ms(20) + 1u; // wraps 20 ms later
    fx.rgb(near_wrap);
    fx.depth(near_wrap + ms(5));
    fx.expect(0, near_wrap + ms(5), near_wrap, "wrap: pair before the wrap");
    fx.rgb(near_wrap + ms(33));                 // past the wrap
    fx.depth(near_wrap + ms(38));
    fx.expect(1, near_wrap + ms(38), near_wrap + ms(33), "wrap: pair across the wrap");
    fx.rgb(near_wrap + ms(66));
    fx.depth(near_wrap + ms(52));               // RGB held is 14 ms newer
    fx.expect(2, near_wrap + ms(52), near_wrap + ms(66), "wrap: RGB newer than depth pairs");
    if (fx.publishedCount() == 3) {
        CHECK(fx.frame(1).timestamp_depth > fx.frame(0).timestamp_depth &&
              fx.frame(2).timestamp_depth > fx.frame(1).timestamp_depth,
              "wrap: unwrapped depth ms stay monotonic");
        CHECK(std::fabs(fx.frame(1).timestamp_depth - fx.frame(0).timestamp_depth - 33.0) < 1e-6,
              "wrap: unwrapped delta across the wrap is the real 33 ms");
    }
    fx.release();
    CHECK(fx.freeCount() >= kPoolSize - 1, "wrap: only the held RGB stays checked out");
}

void sectionLatestSlot() {
    Fixture fx;
    fx.unsubscribe();
    const uint32_t T = 5000u * kTicksPerMs;
    for (uint32_t i = 0; i < 3; ++i) {
        fx.rgb(T + ms(33.0 * i));
        fx.depth(T + ms(33.0 * i + 4));
        CHECK(fx.readyCount() == 1, "latest: the ready slot holds exactly one frame");
    }
    std::shared_ptr<Frame> newest = fx.latest();
    CHECK(newest != nullptr, "latest: getLatestFrame returns a frame");
    if (newest) {
        CHECK(newest->frame_id == 3, "latest: the survivor is the newest frame");
        CHECK(newest->depth_ticks == T + ms(70), "latest: newest depth stamp");
        CHECK(newest->rgb_valid, "latest: newest frame is paired");
    }
    CHECK(fx.latest() == nullptr, "latest: a second getLatestFrame returns nothing");
    newest.reset();
    CHECK(fx.freeCount() == kPoolSize, "latest: pool fully restored");
}

void sectionPoolUnderLoad() {
    Fixture fx;
    const uint32_t T = 9000u * kTicksPerMs;
    for (uint32_t i = 0; i < 64; ++i) {
        fx.rgb(T + ms(33.3 * i));
        fx.depth(T + ms(33.3 * i + 3));
        CHECK(fx.publishedCount() == 1, "pool: every depth frame publishes");
        fx.release();
        CHECK(fx.freeCount() == kPoolSize, "pool: released frames return to the pool");
    }
}

void sectionAccel() {
    Fixture fx;
    const uint32_t T = 12000u * kTicksPerMs;
    auto pair = [&](uint32_t i) {
        fx.rgb(T + ms(33.0 * i));
        fx.depth(T + ms(33.0 * i + 4));
    };
    auto near = [](float a, float b) { return std::fabs(a - b) < 1e-5f; };
    pair(0);
    CHECK(fx.publishedCount() == 1 && !fx.frame(0).accel_valid, "accel: invalid before the first sample");
    fx.accel(0.0, 9.0, 0.0);
    fx.accel(0.2, 9.2, 0.4);
    pair(1);
    CHECK(fx.publishedCount() == 2 && fx.frame(1).accel_valid, "accel: valid after samples");
    CHECK(near(fx.frame(1).accel[0], 0.1f) && near(fx.frame(1).accel[1], 9.1f) &&
              near(fx.frame(1).accel[2], 0.2f),
          "accel: the frame carries the mean of the samples since the previous frame");
    pair(2);
    CHECK(fx.frame(2).accel_valid && near(fx.frame(2).accel[1], 9.1f),
          "accel: no new sample -> the previous mean");
    fx.accel(0.0, 0.0, 0.0);
    fx.accel(1.0, 8.0, 1.0);
    pair(3);
    CHECK(near(fx.frame(3).accel[0], 1.0f) && near(fx.frame(3).accel[1], 8.0f),
          "accel: a zero vector (no reading yet) is ignored");
}

}  // namespace

int main() {
    sectionUnits();
    sectionPolicy();
    sectionWrap();
    sectionLatestSlot();
    sectionPoolUnderLoad();
    sectionAccel();

    if (g_failures == 0) {
        std::printf("kinect_pairing_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("kinect_pairing_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
