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
    double timestamp_depth = 0.0;
    double timestamp_rgb   = 0.0;
    bool   depth_valid     = false;
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
        std::queue<std::shared_ptr<RawFrame>>  ready_queue;
        std::queue<std::shared_ptr<RawFrame>>  free_queue;
    };
    std::shared_ptr<PoolState> pool_state_;
    
    // Pairing state
    std::mutex              sync_mutex_;
    std::shared_ptr<RawFrame> depth_pending_;
    std::shared_ptr<RawFrame> rgb_pending_;

    uint64_t frame_counter_ = 0;
    uint64_t pair_log_      = 0;

    std::atomic<bool> running_{false};
    std::thread       capture_thread_;

    FrameCallback frame_callback_;

    void captureLoop();

    // libfreenect callbacks (static, forwarded via user data)
    static void depthCallback(freenect_device* dev, void* depth, uint32_t timestamp);
    static void rgbCallback(freenect_device* dev, void* rgb, uint32_t timestamp);

    void onDepth(void* data, uint32_t timestamp);
    void onRgb(void* data, uint32_t timestamp);
    
    // Internal helper to get/release frames from pool
    std::shared_ptr<RawFrame> acquireFreeFrame();
};

} // namespace sensor
} // namespace kfusion
