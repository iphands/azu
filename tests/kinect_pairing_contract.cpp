// kinect_pairing_contract (big-fix Todo 23): CPU-only, device-free contract for
// KinectSensor depth/RGB pairing and latest-frame semantics. The fixture is a
// REAL kfusion::sensor::KinectSensor built by its default constructor only: no
// init(), no start(), no libfreenect call, no device, no capture thread, no
// sleep. AZU_PIPELINE_TEST_SEAM injects synthetic samples into the real private
// onDepth()/onRgb() path and reads the real pending/ready/free state.
//
// The narrative section walks ONE sensor through the canonical fixture, so the
// frame-id ledger stays continuous across rejects and across both delivery modes:
//
//   1  a 51 ms depth/RGB delta is stale: nothing is published, no frame id is
//      consumed, and exactly one buffer stays retained — the newer-timestamped
//      side. The acceptance of the NEXT sample on the stale stream is itself the
//      witness that the retained side was the depth frame and the recycled one
//      the RGB frame; no pending-timestamp accessor is needed.
//   2  accepted pairs are ID-sequential across both ledgers (depth-led and
//      RGB-led), so a rejected pair cannot silently burn an id.
//   3  a delta of exactly kMaxFrameSyncDeltaMs is stale: the comparison is strict
//      `<`. The fixture pins its literal 50 ms boundary against the product
//      constant, so retuning the threshold fails here instead of quietly turning
//      the boundary case into an ordinary one.
//   4  every callback observes syncMutexIsFreeForTests() == true, so product
//      publication happens outside sync_mutex_; a callback invoked under the
//      pairing lock fails every accepted case at once.
//   5  a published frame carries the depth bytes of its own depth sample and the
//      RGB bytes of the RGB sample it actually paired with, plus millisecond
//      timestamps derived from libfreenect microseconds. A frame assembled from
//      the wrong pair is a byte mismatch, not a guess.
//   6  with no callback registered the ready slot holds ONE frame: three accepted
//      pairs leave the newest in the slot and recycle the older two (the count
//      stays 1 and the surviving id proves 6 and 7 were consumed and dropped).
//      getLatestFrame() moves the slot out, so the second call returns nullptr
//      instead of a stale frame.
//   7  the pool is asserted after every step (the exact free count names how many
//      buffers are held) and returns in full once references drop; a stranded
//      pending or ready frame shows up as a permanent shortfall.
#include "sensor/KinectSensor.h"

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

// POOL_SIZE in include/sensor/KinectSensor.h is private; the contract pins the
// same number so a leaked pending/ready buffer is observable as a shortfall.
constexpr size_t kPoolSize = 8;

// The product conversion: libfreenect microseconds -> RawFrame milliseconds.
constexpr double msOf(uint32_t us) { return static_cast<double>(us) / 1000.0; }

// One accepted publication, as observed from inside the product callback.
struct Received {
    uint64_t           frame_id        = 0;
    bool               lock_free       = false;
    bool               depth_valid     = false;
    bool               rgb_valid       = false;
    double             timestamp_depth = -1.0;
    double             timestamp_rgb   = -1.0;
    std::shared_ptr<Frame> keep;
};

// Per-sample payloads, cached by microsecond timestamp, so an accepted frame is
// compared against the exact bytes injected for it.
std::vector<uint16_t> makeDepth(uint32_t ts_us) {
    std::vector<uint16_t> d(kDepthBytes / sizeof(uint16_t));
    for (size_t i = 0; i < d.size(); ++i) {
        d[i] = static_cast<uint16_t>(1u + ((i + ts_us) % 2046u));
    }
    return d;
}

std::vector<uint8_t> makeRgb(uint32_t ts_us) {
    std::vector<uint8_t> rgb(kRgbBytes);
    for (size_t i = 0; i < rgb.size(); ++i) {
        rgb[i] = static_cast<uint8_t>((i * 31u + ts_us) & 0xffu);
    }
    return rgb;
}

class Fixture {
public:
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;

    Fixture() { subscribe(); }
    ~Fixture() { sensor_.setFrameCallback(nullptr); }

    void subscribe() {
        sensor_.setFrameCallback(
            [this](std::shared_ptr<Frame> frame) { onPublished(std::move(frame)); });
    }
    void unsubscribe() { sensor_.setFrameCallback(nullptr); }

    void injectDepth(uint32_t ts_us) {
        const std::vector<uint16_t>& payload = depthPayload(ts_us);
        sensor_.injectDepthForTests(payload.data(), ts_us);
    }

    void injectRgb(uint32_t ts_us) {
        const std::vector<uint8_t>& payload = rgbPayload(ts_us);
        sensor_.injectRgbForTests(payload.data(), ts_us);
    }

    uint64_t lastAcceptedId() const {
        return published_.empty() ? 0u : published_.back().frame_id;
    }
    size_t readyCount() const { return sensor_.readyFrameCountForTests(); }
    size_t freeCount() const { return sensor_.freeFrameCountForTests(); }
    bool syncLockIsFree() { return sensor_.syncMutexIsFreeForTests(); }
    std::shared_ptr<Frame> latest() { return sensor_.getLatestFrame(); }

    // Drop every reference the fixture holds to a published frame so the pooled
    // deleter recycles it.
    void releasePublished() { published_.clear(); }

    void expectNothing(const std::string& what) {
        CHECK(published_.size() == expected_accepted_, what + ": no callback fired");
        CHECK(sensor_.readyFrameCountForTests() == 0, what + ": no ready slot was filled");
        CHECK(sensor_.syncMutexIsFreeForTests(), what + ": the pairing lock was released");
    }

    void expectAccepted(uint64_t id, uint32_t depth_us, uint32_t rgb_us) {
        ++expected_accepted_;
        const std::string who = "accepted id " + std::to_string(id);
        CHECK(published_.size() == expected_accepted_, who + ": published exactly once");
        if (published_.size() != expected_accepted_) {
            return;
        }
        const Received& r = published_.back();
        CHECK(r.frame_id == id, who + ": takes the next sequential frame id");
        CHECK(r.lock_free, who + ": callback ran outside sync_mutex_");
        CHECK(r.depth_valid && r.rgb_valid, who + ": both halves marked valid");
        CHECK(r.timestamp_depth == msOf(depth_us), who + ": depth ms from libfreenect us");
        CHECK(r.timestamp_rgb == msOf(rgb_us), who + ": rgb ms from libfreenect us");
        CHECK(depthMatches(r, depth_us), who + ": carries its own depth sample");
        CHECK(rgbMatches(r, rgb_us), who + ": carries the RGB sample it paired with");
        CHECK(sensor_.readyFrameCountForTests() == 0,
              who + ": callback delivery does not also fill the ready slot");
    }

    // `held` buffers are checked out of the pool: published frames the fixture
    // still references, plus any retained pending side.
    void expectHeld(size_t held, const std::string& what) {
        const size_t want   = kPoolSize - held;
        const size_t actual = sensor_.freeFrameCountForTests();
        CHECK(actual == want,
              what + ": " + std::to_string(want) + " free expected, got " + std::to_string(actual));
    }

private:
    void onPublished(std::shared_ptr<Frame> frame) {
        Received r;
        // The mechanical proof publication is outside the lock: try_lock() on the
        // mutex the publisher would still be holding.
        r.lock_free       = sensor_.syncMutexIsFreeForTests();
        r.frame_id        = frame->frame_id;
        r.depth_valid     = frame->depth_valid;
        r.rgb_valid       = frame->rgb_valid;
        r.timestamp_depth = frame->timestamp_depth;
        r.timestamp_rgb   = frame->timestamp_rgb;
        r.keep            = frame;
        published_.push_back(std::move(r));
    }

    bool depthMatches(const Received& r, uint32_t ts_us) {
        if (!r.keep || r.keep->depth.size() * sizeof(uint16_t) != kDepthBytes) {
            return false;
        }
        const std::vector<uint16_t>& want = depthPayload(ts_us);
        return std::memcmp(r.keep->depth.data(), want.data(), kDepthBytes) == 0;
    }

    bool rgbMatches(const Received& r, uint32_t ts_us) {
        if (!r.keep || r.keep->rgb.size() != kRgbBytes) {
            return false;
        }
        const std::vector<uint8_t>& want = rgbPayload(ts_us);
        return std::memcmp(r.keep->rgb.data(), want.data(), kRgbBytes) == 0;
    }

    const std::vector<uint16_t>& depthPayload(uint32_t ts_us) {
        std::map<uint32_t, std::vector<uint16_t>>::iterator it = depth_cache_.find(ts_us);
        if (it != depth_cache_.end()) {
            return it->second;
        }
        return depth_cache_.emplace(ts_us, makeDepth(ts_us)).first->second;
    }

    const std::vector<uint8_t>& rgbPayload(uint32_t ts_us) {
        std::map<uint32_t, std::vector<uint8_t>>::iterator it = rgb_cache_.find(ts_us);
        if (it != rgb_cache_.end()) {
            return it->second;
        }
        return rgb_cache_.emplace(ts_us, makeRgb(ts_us)).first->second;
    }

    kfusion::sensor::KinectSensor             sensor_;
    std::vector<Received>                     published_;
    std::map<uint32_t, std::vector<uint16_t>> depth_cache_;
    std::map<uint32_t, std::vector<uint8_t>>  rgb_cache_;
    size_t                                    expected_accepted_ = 0;
};

// Steps 1-8: the pairing ledger of one continuously running sensor.
void sectionPairingLedger(Fixture& fx) {
    CHECK(fx.freeCount() == kPoolSize, "a fresh sensor has an idle pool");
    CHECK(fx.readyCount() == 0, "a fresh sensor has no ready frame");
    CHECK(fx.syncLockIsFree(), "a fresh sensor holds no pairing lock");

    // 1. depth 1000.000 ms + RGB 949.000 ms: a 51 ms delta, above the window.
    fx.injectDepth(1000000u);
    fx.injectRgb(949000u);
    fx.expectNothing("stale RGB 51 ms behind a newer depth");
    fx.expectHeld(1, "stale RGB recycled, newer depth retained");

    // 2. RGB 950.001 ms pairs with the RETAINED depth sample.
    fx.injectRgb(950001u);
    fx.expectAccepted(1, 1000000u, 950001u);
    fx.expectHeld(1, "one published frame checked out");

    // 3. depth-led accept: depth 2000.000 ms + RGB 1990.000 ms.
    fx.injectDepth(2000000u);
    fx.injectRgb(1990000u);
    fx.expectAccepted(2, 2000000u, 1990000u);
    fx.expectHeld(2, "two published frames checked out");

    // 4. RGB-led accept: RGB 3000.000 ms + depth 3040.000 ms.
    fx.injectRgb(3000000u);
    fx.expectNothing("an unmatched RGB sample publishes nothing");
    fx.injectDepth(3040000u);
    fx.expectAccepted(3, 3040000u, 3000000u);
    fx.expectHeld(3, "three published frames checked out");

    // 5. depth 4000.000 ms + RGB 3950.000 ms: exactly the window, so stale.
    CHECK(msOf(4000000u) - msOf(3950000u) == Sensor::kMaxFrameSyncDeltaMs,
          "the boundary fixture delta is exactly the documented window");
    fx.injectDepth(4000000u);
    fx.injectRgb(3950000u);
    fx.expectNothing("exactly-50 ms RGB delta");
    fx.expectHeld(4, "exact-window RGB recycled, depth retained");

    // 6. RGB 3950.001 ms is 49.999 ms, just inside.
    fx.injectRgb(3950001u);
    fx.expectAccepted(4, 4000000u, 3950001u);
    fx.expectHeld(4, "four published frames checked out");

    // 7. RGB 5000.000 ms is newer; depth 4949.000 ms is 51 ms stale.
    fx.injectRgb(5000000u);
    fx.injectDepth(4949000u);
    fx.expectNothing("stale depth 51 ms behind a newer RGB");
    fx.expectHeld(5, "stale depth recycled, newer RGB retained");

    // 8. depth 4950.001 ms pairs with the retained RGB sample.
    fx.injectDepth(4950001u);
    fx.expectAccepted(5, 4950001u, 5000000u);
    fx.expectHeld(5, "five published frames checked out");
    std::printf("  ledger: 5 accepts, 3 rejects, ids 1..5 sequential\n");
}

// Steps 9-13 on the same sensor: latest-slot semantics with no callback.
void sectionLatestSlotDropsOlder(Fixture& fx) {
    fx.releasePublished();
    CHECK(fx.freeCount() == kPoolSize,
          "dropping the fixture references recycles every published buffer");

    fx.unsubscribe();

    struct Pair {
        uint32_t depth_us;
        uint32_t rgb_us;
    };
    const Pair pairs[] = {
        {6000000u, 5990000u},
        {7000000u, 6990000u},
        {8000000u, 7990000u},
    };
    for (size_t i = 0; i < 3; ++i) {
        fx.injectDepth(pairs[i].depth_us);
        fx.injectRgb(pairs[i].rgb_us);
        CHECK(fx.readyCount() == 1, "the ready slot holds exactly one frame");
        fx.expectHeld(1, "each new pair recycled the frame it displaced");
    }

    std::shared_ptr<Frame> newest = fx.latest();
    CHECK(newest != nullptr, "getLatestFrame returns the newest frame");
    if (newest) {
        // id 8 is the witness that ids 6 and 7 were consumed AND dropped: FIFO
        // would have returned 6, a retaining queue would still hold a second one.
        CHECK(newest->frame_id == 8, "the surviving frame is the newest accepted pair");
        CHECK(newest->timestamp_depth == msOf(pairs[2].depth_us),
              "the newest frame carries the newest depth timestamp");
        CHECK(newest->timestamp_rgb == msOf(pairs[2].rgb_us),
              "the newest frame carries the newest RGB timestamp");
        CHECK(newest->depth_valid && newest->rgb_valid, "the newest frame is complete");
    }
    CHECK(fx.readyCount() == 0, "getLatestFrame emptied the ready slot");
    CHECK(fx.latest() == nullptr, "a second getLatestFrame returns nothing, not a stale frame");

    newest.reset();
    CHECK(fx.freeCount() == kPoolSize, "releasing the consumed frame restores the whole pool");
    std::printf("  latest slot: ids 6/7 dropped, id 8 consumed once\n");
}

// A consumer that releases each frame keeps the pool in full circulation: this is
// what a retained ready queue or a leaked pending side would break.
void sectionPoolStaysHealthyUnderLoad() {
    Fixture fx;
    for (uint32_t step = 0; step < 8; ++step) {
        const uint32_t depth_us = 1000000u * (step + 1u);
        const uint32_t rgb_us   = depth_us - 1000u;
        fx.injectDepth(depth_us);
        fx.injectRgb(rgb_us);
        CHECK(fx.lastAcceptedId() == step + 1u, "pairs publish under sequential ids");
        CHECK(fx.readyCount() == 0, "callback delivery leaves the ready slot empty");
        fx.releasePublished();
        CHECK(fx.freeCount() == kPoolSize, "each released frame returns to the pool");
    }
}

}  // namespace

int main() {
    Fixture fixture;
    sectionPairingLedger(fixture);
    sectionLatestSlotDropsOlder(fixture);
    sectionPoolStaysHealthyUnderLoad();

    if (g_failures == 0) {
        std::printf("kinect_pairing_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("kinect_pairing_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
