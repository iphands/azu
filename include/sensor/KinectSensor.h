#pragma once

// The canonical CPU depth-domain boundary lives in its own header so that the
// sensor, the frame builder and the signal conditioner all consume ONE
// validity rule (big-fix Todo 20). It is included here because every existing
// consumer of rawDepthToMeters() already includes this header.
#include "sensor/DepthValidity.h"

#include <atomic>
#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <memory>
#include <functional>

struct _freenect_context;
typedef struct _freenect_context freenect_context;
struct _freenect_device;
typedef struct _freenect_device freenect_device;

namespace kfusion {
namespace sensor {

// Raw Kinect v1 depth/rgb frame dimensions
static constexpr int DEPTH_WIDTH  = 640;
static constexpr int DEPTH_HEIGHT = 480;
static constexpr int RGB_WIDTH    = 640;
static constexpr int RGB_HEIGHT   = 480;

// Convert raw 11-bit depth to meters: rawDepthToMeters() and the full
// configured-band boundary cpuDepthMeters() come from sensor/DepthValidity.h,
// which is included above. big-fix Todo 20 moved them there so the sensor,
// FrameData and the signal conditioner share one validity predicate instead of
// three ad hoc range checks. Do not redefine the curve locally.

struct RawFrame {
    std::vector<uint16_t> depth;
    std::vector<uint8_t>  rgb;
    // Milliseconds on the Kinect's own 60 MHz clock, unwrapped per stream
    // (see kTicksPerMs). Only differences between them are meaningful.
    double timestamp_depth = 0.0;
    double timestamp_rgb   = 0.0;
    // Raw libfreenect timestamps (60 MHz ticks, uint32, wraps every ~71.6 s).
    uint32_t depth_ticks   = 0;
    uint32_t rgb_ticks     = 0;
    bool   depth_valid     = false;
    // false for a depth-only frame: no RGB sample was close enough in time.
    // Geometry never waits on colour.
    bool   rgb_valid       = false;
    uint64_t frame_id      = 0;

    RawFrame() {
        depth.resize(DEPTH_WIDTH * DEPTH_HEIGHT);
        rgb.resize(RGB_WIDTH * RGB_HEIGHT * 3);
    }
};

using FrameCallback = std::function<void(std::shared_ptr<RawFrame>)>;

class KinectSensor {
public:
    // libfreenect stamps frames with the Kinect's 60 MHz hardware counter, NOT
    // microseconds (OpenNI2-FreenectDriver VideoStream.hpp; measured 2,002,155
    // ticks per 33.37 ms depth frame). The uint32 counter wraps every 71.58 s, so
    // deltas are always taken as int32_t(a - b).
    static constexpr double kTicksPerMs = 60000.0;

    // Depth-led pairing: each depth frame takes the nearest RGB sample whose
    // |delta| <= this (half a 30 Hz frame period), else it is published
    // depth-only with rgb_valid == false. See docs/CANONICAL_SEMANTICS.md.
    static constexpr double kMaxColorSkewMs = 17.0;

    // Signed depth - rgb delta in ms, correct across a counter wrap.
    static double tickDeltaMs(uint32_t depth_ticks, uint32_t rgb_ticks) {
        return static_cast<int32_t>(depth_ticks - rgb_ticks) / kTicksPerMs;
    }

    KinectSensor();
    ~KinectSensor();

    // Non-copyable
    KinectSensor(const KinectSensor&) = delete;
    KinectSensor& operator=(const KinectSensor&) = delete;

    bool init();
    bool start();
    void stop();
    bool isRunning() const { return running_.load(); }
    bool isConnected() const { return device_ != nullptr; }

    // Register callback invoked on UI/pipeline thread (posted from capture thread)
    void setFrameCallback(FrameCallback cb) { frame_callback_ = std::move(cb); }

    // Returns latest synchronized frame (zero-copy)
    std::shared_ptr<RawFrame> getLatestFrame();

    // Internal helper to get/release frames from pool
    void releaseFrame(std::shared_ptr<RawFrame> frame);

private:
    // libfreenect state
    freenect_context* ctx_    = nullptr;
    freenect_device*  device_ = nullptr;

    // Thread-safe pipeline state
    static constexpr size_t POOL_SIZE = 8;
    
    struct PoolState {
        std::vector<std::shared_ptr<RawFrame>> pool;
        std::mutex                             mutex;
        // Single latest ready slot (not a queue): getLatestFrame() is a
        // newest-wins consumer, so a queue could only ever hand out stale
        // frames. Destructed/recycled OUTSIDE `mutex` — see publishLatest().
        std::shared_ptr<RawFrame>              ready_frame;
        std::queue<std::shared_ptr<RawFrame>>  free_queue;
    };
    std::shared_ptr<PoolState> pool_state_;
    
    // Pairing state (guarded by sync_mutex_). rgb_held_ is the newest RGB
    // sample not yet consumed; depth_waiting_ is a depth frame whose newest RGB
    // was too old, parked until the next RGB arrives (or the next depth frame
    // flushes it depth-only).
    std::mutex              sync_mutex_;
    std::shared_ptr<RawFrame> rgb_held_;
    std::shared_ptr<RawFrame> depth_waiting_;
    // Per-stream unwrap state for RawFrame::timestamp_* (60 MHz tick counters).
    uint32_t last_depth_ticks_ = 0;
    uint32_t last_rgb_ticks_   = 0;
    uint64_t depth_wraps_      = 0;
    uint64_t rgb_wraps_        = 0;
    bool     have_depth_ticks_ = false;
    bool     have_rgb_ticks_   = false;

    uint64_t frame_counter_ = 0;
    uint64_t pair_log_      = 0;

    std::atomic<bool> running_{false};
    std::thread       capture_thread_;

    FrameCallback frame_callback_;

    void captureLoop();

    // libfreenect callbacks (static, forwarded via user data)
    static void depthCallback(freenect_device* dev, void* depth, uint32_t timestamp);
    static void rgbCallback(freenect_device* dev, void* rgb, uint32_t timestamp);

    void onDepth(const void* data, uint32_t timestamp);
    void onRgb(const void* data, uint32_t timestamp);

    // Move the RGB sample `rgb` into depth frame `depth` (buffer swap, no copy),
    // stamp it, and assign the next frame id. Caller holds sync_mutex_.
    void attachRgbLocked(RawFrame& depth, RawFrame& rgb);
    // Assign the next frame id to a depth-only frame. Caller holds sync_mutex_.
    void markDepthOnlyLocked(RawFrame& depth);

    // Deliver a finished frame outside sync_mutex_ (callback, else latest slot).
    void deliver(std::shared_ptr<RawFrame> frame, FrameCallback& callback);

    // Install `frame` as the single ready frame, dropping whatever it replaces.
    // The displaced frame is destructed outside pool_state_->mutex: its pooled
    // deleter locks that same mutex, so destroying it inside would deadlock.
    void publishLatest(std::shared_ptr<RawFrame> frame);
    
    // Internal helper to get/release frames from pool
    std::shared_ptr<RawFrame> acquireFreeFrame();

#ifdef AZU_PIPELINE_TEST_SEAM
public:
    // ---- Device-free pairing test seam (big-fix Todo 23) ----
    // Same compile-time discipline as the PipelineController / SignalConditioner
    // seams: AZU_PIPELINE_TEST_SEAM is defined only for the azu_test_core
    // targets, so production never sees these member functions. They add no
    // layout change and no libfreenect or device call — a default-constructed
    // KinectSensor is the whole fixture. inject*ForTests() feed the REAL private
    // onDepth()/onRgb() pairing path, the counts read the real queues, and
    // syncMutexIsFreeForTests() is the mechanical witness that product
    // publication happens outside sync_mutex_.
    void injectDepthForTests(const void* data, uint32_t timestamp);
    void injectRgbForTests(const void* data, uint32_t timestamp);
    size_t readyFrameCountForTests() const;
    size_t freeFrameCountForTests() const;
    bool syncMutexIsFreeForTests();
#endif
};

} // namespace sensor
} // namespace kfusion
