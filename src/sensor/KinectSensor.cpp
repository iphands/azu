#include "sensor/KinectSensor.h"
#include "utils/Logger.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <sys/time.h>

#ifdef HAVE_FREENECT
#include <libfreenect/libfreenect.h>
#endif

namespace kfusion {
namespace sensor {

KinectSensor::KinectSensor() {
  pool_state_ = std::make_shared<PoolState>();
  for (size_t i = 0; i < POOL_SIZE; ++i) {
    pool_state_->pool.push_back(std::make_shared<RawFrame>());
    pool_state_->free_queue.push(pool_state_->pool.back());
  }
}

KinectSensor::~KinectSensor() {
    stop();
#ifdef HAVE_FREENECT
    if (device_) {
        freenect_close_device(device_);
        device_ = nullptr;
    }
    if (ctx_) {
        freenect_shutdown(ctx_);
        ctx_ = nullptr;
    }
#endif
}

bool KinectSensor::init() {
#ifndef HAVE_FREENECT
    KFLOG_ERROR("Sensor", "Kinect support is unavailable in this build (libfreenect not found at configure time).");
    return false;
#else
    // After stop(), device/context stay open so the pipeline can start again (e.g. Reset scan).
    if (device_) return true;

    if (freenect_init(&ctx_, nullptr) < 0) {
        KFLOG_ERROR("Sensor", "freenect_init FAILED: Could not initialize libfreenect.");
        ctx_ = nullptr;
        return false;
    }
    // Every failure below releases the context, so a retry starts clean
    // instead of leaking one libusb context per attempt.
    auto fail = [this](const char* msg) {
        KFLOG_ERROR("Sensor", msg);
        freenect_shutdown(ctx_);
        ctx_ = nullptr;
        return false;
    };
    freenect_set_log_level(ctx_, FREENECT_LOG_ERROR);
    freenect_select_subdevices(ctx_,
        static_cast<freenect_device_flags>(FREENECT_DEVICE_MOTOR | FREENECT_DEVICE_CAMERA));

    int num_devices = freenect_num_devices(ctx_);
    if (num_devices < 1) {
        return fail("No Kinect devices detected. Please check USB and power.");
    }

    if (freenect_open_device(ctx_, &device_, 0) < 0) {
        device_ = nullptr;
        return fail("freenect_open_device FAILED: Could not open Kinect index 0.");
    }

    freenect_set_user(device_, this);
    freenect_set_depth_callback(device_, depthCallback);
    freenect_set_video_callback(device_, rgbCallback);
    freenect_set_depth_mode(device_,
        freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_11BIT));
    freenect_set_video_mode(device_,
        freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB));

    return true;
#endif
}

bool KinectSensor::start() {
#ifndef HAVE_FREENECT
    return false;
#else
    if (!device_) return false;
    if (running_.load()) return true;

    if (freenect_start_depth(device_) < 0) {
        KFLOG_ERROR("Sensor", "freenect_start_depth FAILED.");
        return false;
    }
    if (freenect_start_video(device_) < 0) {
        KFLOG_ERROR("Sensor", "freenect_start_video FAILED.");
        freenect_stop_depth(device_);
        return false;
    }

    running_.store(true);
    capture_thread_ = std::thread(&KinectSensor::captureLoop, this);
    return true;
#endif
}

void KinectSensor::stop() {
#ifndef HAVE_FREENECT
    running_.store(false);
    return;
#else
    if (!running_.load()) return;
    running_.store(false);

    // Join the event pump FIRST. freenect_stop_*() cancels the iso transfers
    // and then pumps libusb events itself until they are reaped; doing that
    // while the capture thread is still pumping means two threads handling
    // events (and callbacks) at once. After the join there is one pumper.
    if (capture_thread_.joinable())
        capture_thread_.join();

    if (device_) {
        freenect_stop_depth(device_);
        freenect_stop_video(device_);
    }
#endif
    // A restart must not pair against samples from before the stop.
    std::shared_ptr<RawFrame> rgb, depth;
    {
        std::lock_guard<std::mutex> lk(sync_mutex_);
        rgb   = std::move(rgb_held_);
        depth = std::move(depth_waiting_);
        have_depth_ticks_ = have_rgb_ticks_ = false;
    }
}

void KinectSensor::captureLoop() {
#ifdef HAVE_FREENECT
    while (running_.load()) {
        struct timeval timeout;
        timeout.tv_sec  = 0;
        timeout.tv_usec = 10000; // 10ms poll
        int ret = freenect_process_events_timeout(ctx_, &timeout);
        if (ret < 0 && running_.load()) {
            KFLOGF_ERROR("Sensor", "libfreenect event processing error: %d", ret);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
#endif
}

#ifdef HAVE_FREENECT
void KinectSensor::depthCallback(freenect_device* dev, void* depth, uint32_t timestamp) {
    static_cast<KinectSensor*>(freenect_get_user(dev))->onDepth(depth, timestamp);
}

void KinectSensor::rgbCallback(freenect_device *dev, void *rgb,
                               uint32_t timestamp) {
  static_cast<KinectSensor *>(freenect_get_user(dev))->onRgb(rgb, timestamp);
}
#else
void KinectSensor::depthCallback(freenect_device*, void*, uint32_t) {}
void KinectSensor::rgbCallback(freenect_device*, void*, uint32_t) {}
#endif

namespace {
// Unwrap one stream's uint32 60 MHz counter into milliseconds. A backwards step
// of more than half the range is a wrap; anything smaller is jitter/reordering.
double unwrapTicksMs(uint32_t ts, uint32_t& last, uint64_t& wraps, bool& have) {
    if (have && ts < last && (last - ts) > 0x80000000u) {
        ++wraps;
    }
    last = ts;
    have = true;
    return static_cast<double>((wraps << 32) + ts) / KinectSensor::kTicksPerMs;
}
} // namespace

void KinectSensor::attachRgbLocked(RawFrame& depth, RawFrame& rgb) {
    // Swap, not memcpy: both buffers are pool-owned and equally sized, and the
    // RGB frame goes straight back to the pool with depth's stale buffer.
    std::swap(depth.rgb, rgb.rgb);
    depth.rgb_ticks     = rgb.rgb_ticks;
    depth.timestamp_rgb = rgb.timestamp_rgb;
    depth.rgb_valid     = true;
    depth.frame_id      = ++frame_counter_;
}

void KinectSensor::markDepthOnlyLocked(RawFrame& depth) {
    depth.rgb_valid = false;
    depth.frame_id  = ++frame_counter_;
}

void KinectSensor::deliver(std::shared_ptr<RawFrame> frame, FrameCallback& callback) {
    (frame->rgb_valid ? paired_ : depth_only_).fetch_add(1, std::memory_order_relaxed);
    if (++pair_log_ % 150 == 0) {
        KFLOGF_DEBUG("Sensor", "Frame %lu published (rgb %s, depth-rgb %.2f ms)",
                     frame->frame_id, frame->rgb_valid ? "paired" : "missing",
                     frame->rgb_valid ? tickDeltaMs(frame->depth_ticks, frame->rgb_ticks) : 0.0);
    }
    if (callback) {
        callback(std::move(frame));
    } else {
        publishLatest(std::move(frame));
    }
}

void KinectSensor::onDepth(const void* data, uint32_t timestamp) {
    FrameCallback callback;
    std::shared_ptr<RawFrame> overdue;
    std::shared_ptr<RawFrame> publish;

    {
        std::lock_guard<std::mutex> lk(sync_mutex_);

        depth_callbacks_.fetch_add(1, std::memory_order_relaxed);
        std::shared_ptr<RawFrame> frame = acquireFreeFrame();
        if (!frame) {   // consumers are holding every pooled buffer
            pool_exhausted_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        std::memcpy(frame->depth.data(), data, DEPTH_WIDTH * DEPTH_HEIGHT * 2);
        frame->depth_ticks     = timestamp;
        frame->timestamp_depth = unwrapTicksMs(timestamp, last_depth_ticks_, depth_wraps_,
                                               have_depth_ticks_);
        frame->depth_valid     = true;

        // A parked depth frame that never got a closer RGB sample goes out now,
        // depth-only, ahead of this one so ids stay in capture order.
        if (depth_waiting_) {
            markDepthOnlyLocked(*depth_waiting_);
            overdue = std::move(depth_waiting_);
        }

        if (!rgb_held_) {
            depth_waiting_ = std::move(frame);
        } else {
            const double d = tickDeltaMs(timestamp, rgb_held_->rgb_ticks);
            if (std::fabs(d) <= kMaxColorSkewMs) {
                attachRgbLocked(*frame, *rgb_held_);
                rgb_held_.reset();
                publish = std::move(frame);
            } else if (d > 0.0) {
                // Newest RGB is too old; the next one will be nearer.
                depth_waiting_ = std::move(frame);
            } else {
                markDepthOnlyLocked(*frame);
                publish = std::move(frame);
            }
        }
        if (overdue || publish) callback = frame_callback_;
    }

    // Publication runs with sync_mutex_ released: the callback is pipeline code.
    if (overdue) deliver(std::move(overdue), callback);
    if (publish) deliver(std::move(publish), callback);
}

void KinectSensor::onRgb(const void* data, uint32_t timestamp) {
    FrameCallback callback;
    std::shared_ptr<RawFrame> publish;
    std::shared_ptr<RawFrame> spent;
    std::shared_ptr<RawFrame> superseded;

    {
        std::lock_guard<std::mutex> lk(sync_mutex_);

        rgb_callbacks_.fetch_add(1, std::memory_order_relaxed);
        std::shared_ptr<RawFrame> frame = acquireFreeFrame();
        if (!frame) {
            pool_exhausted_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        std::memcpy(frame->rgb.data(), data, RGB_WIDTH * RGB_HEIGHT * 3);
        frame->rgb_ticks     = timestamp;
        frame->timestamp_rgb = unwrapTicksMs(timestamp, last_rgb_ticks_, rgb_wraps_,
                                             have_rgb_ticks_);
        frame->rgb_valid     = true;

        if (depth_waiting_) {
            const double d = tickDeltaMs(depth_waiting_->depth_ticks, timestamp);
            if (std::fabs(d) <= kMaxColorSkewMs) {
                attachRgbLocked(*depth_waiting_, *frame);
                spent = std::move(frame);
            } else {
                markDepthOnlyLocked(*depth_waiting_);
            }
            publish = std::move(depth_waiting_);
            callback = frame_callback_;
        }
        // Any older held sample is now useless: this one is nearer to every
        // future depth frame.
        superseded = std::move(rgb_held_);
        if (frame) rgb_held_ = std::move(frame);
    }

    // Consumed/superseded RGB buffers recycle here, past the lock.
    spent.reset();
    superseded.reset();
    if (publish) deliver(std::move(publish), callback);
}

void KinectSensor::publishLatest(std::shared_ptr<RawFrame> frame) {
    std::shared_ptr<RawFrame> displaced;
    {
        std::lock_guard<std::mutex> lk(pool_state_->mutex);
        displaced = std::move(pool_state_->ready_frame);
        pool_state_->ready_frame = std::move(frame);
    }
    // `displaced` (the older frame getLatestFrame() would otherwise have handed
    // out) recycles here, past the lock: its pooled deleter locks the mutex
    // this scope just released.
}

std::shared_ptr<RawFrame> KinectSensor::getLatestFrame() {
    std::lock_guard<std::mutex> lk(pool_state_->mutex);
    // Move-out leaves the slot empty: one consumer, newest frame only.
    return std::move(pool_state_->ready_frame);
}

std::shared_ptr<RawFrame> KinectSensor::acquireFreeFrame() {
    std::lock_guard<std::mutex> lk(pool_state_->mutex);
    if (pool_state_->free_queue.empty())
        return nullptr;

    auto base_frame = pool_state_->free_queue.front();
    pool_state_->free_queue.pop();

    base_frame->depth_valid = false;
    base_frame->rgb_valid = false;

    auto state = pool_state_;
    return std::shared_ptr<RawFrame>(
        base_frame.get(), [state, base_frame](RawFrame *) {
            std::lock_guard<std::mutex> lk_inner(state->mutex);
            state->free_queue.push(base_frame);
        });
}

void KinectSensor::releaseFrame(std::shared_ptr<RawFrame>) {
    // No-op: pooled frames recycle through their deleter.
}

#ifdef AZU_PIPELINE_TEST_SEAM
void KinectSensor::injectDepthForTests(const void* data, uint32_t timestamp) {
    onDepth(data, timestamp);
}

void KinectSensor::injectRgbForTests(const void* data, uint32_t timestamp) {
    onRgb(data, timestamp);
}

size_t KinectSensor::readyFrameCountForTests() const {
    std::lock_guard<std::mutex> lk(pool_state_->mutex);
    return pool_state_->ready_frame ? 1u : 0u;
}

size_t KinectSensor::freeFrameCountForTests() const {
    std::lock_guard<std::mutex> lk(pool_state_->mutex);
    return pool_state_->free_queue.size();
}

bool KinectSensor::syncMutexIsFreeForTests() {
    if (sync_mutex_.try_lock()) {
        sync_mutex_.unlock();
        return true;
    }
    return false;
}
#endif

} // namespace sensor
} // namespace kfusion
