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
        return false;
    }
    freenect_set_log_level(ctx_, FREENECT_LOG_ERROR);
    freenect_select_subdevices(ctx_,
        static_cast<freenect_device_flags>(FREENECT_DEVICE_MOTOR | FREENECT_DEVICE_CAMERA));

    int num_devices = freenect_num_devices(ctx_);
    if (num_devices < 1) {
        KFLOG_ERROR("Sensor", "No Kinect devices detected. Please check USB and power.");
        return false;
    }

    if (freenect_open_device(ctx_, &device_, 0) < 0) {
        KFLOG_ERROR("Sensor", "freenect_open_device FAILED: Could not open Kinect index 0.");
        return false;
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

  freenect_start_depth(device_);
  freenect_start_video(device_);

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

  if (device_) {
    freenect_stop_depth(device_);
    freenect_stop_video(device_);
  }

    if (capture_thread_.joinable())
        capture_thread_.join();
#endif
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

void KinectSensor::onDepth(const void* data, uint32_t timestamp) {
    FrameCallback callback;
    std::shared_ptr<RawFrame> publish;
    bool log_pair = false;

    {
        std::lock_guard<std::mutex> lk(sync_mutex_);

        if (!depth_pending_) {
            depth_pending_ = acquireFreeFrame();
            if (!depth_pending_) return; // Pool exhausted
        }

        std::memcpy(depth_pending_->depth.data(), data, DEPTH_WIDTH * DEPTH_HEIGHT * 2);
        depth_pending_->timestamp_depth = timestamp / 1000.0;
        depth_pending_->depth_valid = true;

        publish = pairPendingLocked();
        if (publish) {
            callback = frame_callback_;
            log_pair = (++pair_log_ % 150 == 0);
        }
    }

    if (!publish) return;

    // Throttled synchronization telemetry and product publication both run with
    // sync_mutex_ released: the callback is pipeline code, and calling it under
    // the pairing lock lets a slow consumer block the capture thread's sibling.
    if (log_pair) {
        KFLOGF_DEBUG("Sensor", "Frames synchronized (depth-led): ID=%lu", publish->frame_id);
    }

    if (callback) {
        callback(std::move(publish));
    } else {
        publishLatest(std::move(publish));
    }
}

void KinectSensor::onRgb(const void* data, uint32_t timestamp) {
    FrameCallback callback;
    std::shared_ptr<RawFrame> publish;
    bool log_pair = false;

    {
        std::lock_guard<std::mutex> lk(sync_mutex_);

        if (!rgb_pending_) {
            rgb_pending_ = acquireFreeFrame();
            if (!rgb_pending_) return;
        }

        std::memcpy(rgb_pending_->rgb.data(), data, RGB_WIDTH * RGB_HEIGHT * 3);
        rgb_pending_->timestamp_rgb = timestamp / 1000.0;
        rgb_pending_->rgb_valid = true;

        publish = pairPendingLocked();
        if (publish) {
            callback = frame_callback_;
            log_pair = (++pair_log_ % 150 == 0);
        }
    }

    if (!publish) return;

    if (log_pair) {
        KFLOGF_DEBUG("Sensor", "Frames synchronized (rgb-led): ID=%lu", publish->frame_id);
    }

    if (callback) {
        callback(std::move(publish));
    } else {
        publishLatest(std::move(publish));
    }
}

std::shared_ptr<RawFrame> KinectSensor::pairPendingLocked() {
    if (!depth_pending_ || !depth_pending_->depth_valid ||
        !rgb_pending_ || !rgb_pending_->rgb_valid) {
        return nullptr;
    }

    const double depth_ms = depth_pending_->timestamp_depth;
    const double rgb_ms   = rgb_pending_->timestamp_rgb;

    // A non-finite timestamp can only come from a corrupt conversion, and it
    // compares false everywhere, so it lands in the stale branch and the
    // non-finite side is the one dropped.
    const bool fresh = std::isfinite(depth_ms) && std::isfinite(rgb_ms) &&
                       std::fabs(depth_ms - rgb_ms) < kMaxFrameSyncDeltaMs;

    if (!fresh) {
        // Stale pair: nothing is published, no frame id is consumed and no RGB
        // is copied. The older-timestamped side is recycled; the newer side is
        // retained so the next sample on the stale stream can pair with it.
        const bool depth_is_stale = !(depth_ms > rgb_ms);
        if (depth_is_stale) {
            depth_pending_.reset();
        } else {
            rgb_pending_.reset();
        }
        return nullptr;
    }

    std::memcpy(depth_pending_->rgb.data(), rgb_pending_->rgb.data(), RGB_WIDTH * RGB_HEIGHT * 3);
    depth_pending_->timestamp_rgb = rgb_ms;
    depth_pending_->rgb_valid = true;
    depth_pending_->frame_id = ++frame_counter_;

    std::shared_ptr<RawFrame> combined = std::move(depth_pending_);
    rgb_pending_.reset();
    return combined;
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
    // No-op manually; handled by custom deleter now
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
