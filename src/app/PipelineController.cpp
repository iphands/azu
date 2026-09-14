#include "app/PipelineController.h"
#include "export/GLBExporter.h"
#include "export/PLYExporter.h"
#include "utils/Logger.h"
#include "utils/Timer.h"
#include "sensor/SuperResolution.h"
#include <QCoreApplication>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifdef CUDA_ENABLED
#include <cuda_runtime.h>
#elif defined(HIP_ENABLED)
#include <hip/hip_runtime.h>
#endif

namespace kfusion {
namespace app {

using namespace std::chrono;

PipelineController::PipelineController(sensor::PreprocessBackend preferred_backend)
    : preferred_backend_(preferred_backend),
      current_pose_(Eigen::Matrix4f::Identity())
{
    syncIcpDepthFromRange(hyperparams_);
    sensor_  = std::make_unique<sensor::KinectSensor>();
    tracker_ = std::make_unique<tracking::ICPTracker>(hyperparams_.icp);
    tsdf_    = std::make_unique<tsdf::TSDFVolume>(hyperparams_.tsdf);
    cubes_   = std::make_unique<meshing::MarchingCubes>();
    
    // Initialize data pool for automated recycling
    data_pool_state_ = std::make_shared<DataPool>();
    for (size_t i = 0; i < DATA_POOL_SIZE; ++i) {
        data_pool_state_->data_pool.push_back(std::make_shared<sensor::FrameData>());
        data_pool_state_->free_data_queue.push(data_pool_state_->data_pool.back().get());
    }

    // Initialize ping-pong buffers
    for (int i = 0; i < 3; ++i) {
        model_buffers_.buffers[i] = std::make_shared<tracking::ModelFrame>();
    }
}

PipelineController::~PipelineController() { stop(); }

bool PipelineController::start() {
  std::lock_guard<std::mutex> ctrl_lk(control_mutex_);
  if (running_.load())
    return true;

    running_.store(true);
    state_.store(PipelineState::Running);
    first_frame_          = true;
    model_ready_.store(false);
    frame_count_          = 0;
    ui_skip_counter_      = 0;
    lost_log_counter_     = 0;
    success_log_counter_  = 0;
    hip_ui_skip_          = 0;
    {
        std::lock_guard<std::mutex> lk(pose_mutex_);
        current_pose_  = Eigen::Matrix4f::Identity();
    }
    // Set frame callback before starting capture
    sensor_->setFrameCallback([this](std::shared_ptr<sensor::RawFrame> raw) {
        onRawFrame(std::move(raw));
    });

    if (!sensor_->init()) {
        KFLOG_ERROR("Pipeline", "Sensor initialization FAILED. Is the Kinect connected?");
        running_.store(false);
        state_.store(PipelineState::Error);
        return false;
    }

    if (!sensor_->start()) {
        KFLOG_ERROR("Pipeline", "Sensor start FAILED. Stream could not be opened.");
        running_.store(false);
        state_.store(PipelineState::Error);
        return false;
    }

#ifdef CUDA_ENABLED
    if (preferred_backend_ == sensor::PreprocessBackend::CPU || preferred_backend_ == sensor::PreprocessBackend::HIP) {
        use_gpu_ = false;
        tsdf_->setGPUEnabled(false);
        KFLOG_INFO("Pipeline", "Using CPU-only path (User explicitly requested or incompatible backend).");
    } else {
        try {
            tsdf_->initGPU();
            tracker_->initGPU();
            cudaError_t stream_err = cudaStreamCreate(&cuda_stream_);
            if (stream_err != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(stream_err));
            }
            for (int i = 0; i < 3; ++i) {
                model_buffers_.buffers[i]->d_vertices = utils::make_cuda_unique<float3>(sensor::DEPTH_WIDTH * sensor::DEPTH_HEIGHT);
                model_buffers_.buffers[i]->d_normals = utils::make_cuda_unique<float3>(sensor::DEPTH_WIDTH * sensor::DEPTH_HEIGHT);
                model_buffers_.buffers[i]->d_colors = utils::make_cuda_unique<uchar3>(sensor::DEPTH_WIDTH * sensor::DEPTH_HEIGHT);
            }
            use_gpu_ = true;
            tsdf_->setGPUEnabled(true);
            KFLOG_INFO("Pipeline", "CUDA hardware acceleration ENABLED (NVIDIA GPU detected).");
        } catch (const std::exception& e) {
            if (cuda_stream_) {
                cudaStreamDestroy(cuda_stream_);
                cuda_stream_ = nullptr;
            }
            use_gpu_ = false;
            tsdf_->setGPUEnabled(false);
            if (preferred_backend_ == sensor::PreprocessBackend::CUDA) {
                KFLOGF_WARN("Pipeline", "CUDA initialization FAILED: %s. Falling back to CPU mode.", e.what());
            } else {
                KFLOGF_WARN("Pipeline", "CUDA initialization FAILED: %s. Falling back to CPU mode.", e.what());
            }
        }
    }
#elif defined(HIP_ENABLED)
    if (preferred_backend_ == sensor::PreprocessBackend::CPU || preferred_backend_ == sensor::PreprocessBackend::CUDA) {
        use_gpu_ = false;
        tsdf_->setGPUEnabled(false);
        KFLOG_INFO("Pipeline", "Using CPU-only path (User explicitly requested or incompatible backend).");
    } else {
        try {
            tsdf_->initGPU();
            tracker_->initGPU();
            hipError_t stream_err = hipStreamCreate(&cuda_stream_);
            if (stream_err != hipSuccess) {
                throw std::runtime_error(hipGetErrorString(stream_err));
            }
            for (int i = 0; i < 3; ++i) {
                model_buffers_.buffers[i]->d_vertices = utils::make_hip_unique<float3>(sensor::DEPTH_WIDTH * sensor::DEPTH_HEIGHT);
                model_buffers_.buffers[i]->d_normals = utils::make_hip_unique<float3>(sensor::DEPTH_WIDTH * sensor::DEPTH_HEIGHT);
                model_buffers_.buffers[i]->d_colors = utils::make_hip_unique<uchar3>(sensor::DEPTH_WIDTH * sensor::DEPTH_HEIGHT);
            }
            use_gpu_ = true;
            tsdf_->setGPUEnabled(true);
            KFLOG_INFO("Pipeline", "HIP/ROCm hardware acceleration ENABLED (AMD iGPU detected).");
        } catch (const std::exception& e) {
            if (cuda_stream_) {
                (void)hipStreamDestroy(cuda_stream_);
                cuda_stream_ = nullptr;
            }
            use_gpu_ = false;
            tsdf_->setGPUEnabled(false);
            KFLOGF_WARN("Pipeline", "HIP initialization FAILED: %s. Falling back to CPU mode.", e.what());
        }
    }
#else
    use_gpu_ = false;
    tsdf_->setGPUEnabled(false);
    KFLOG_INFO("Pipeline", "Using CPU-only path (GPU not enabled in build).");
#endif

    configurePreprocessor();
    if (preprocessor_) {
        preprocessor_->reset();
    }

    // Launch pipeline threads
    tracking_thread_    = std::thread(&PipelineController::trackingLoop, this);
    integration_thread_ = std::thread(&PipelineController::integrationLoop, this);
    meshing_thread_     = std::thread(&PipelineController::meshingLoop, this);

  last_capture_time_ = steady_clock::now();
  last_tracking_time_ = steady_clock::now();

  KFLOG_INFO("Pipeline",
             "Worker threads launched (Tracking, Integration, Meshing).");
  return true;
}

PipelineMetrics PipelineController::metricsSnapshot() const {
  std::lock_guard<std::mutex> lk(metrics_mutex_);
  PipelineMetrics m = metrics_;
  m.state = state_.load();
  return m;
}

Eigen::Matrix4f PipelineController::currentPose() const {
  std::lock_guard<std::mutex> lk(pose_mutex_);
  return current_pose_;
}

FusionHyperparams PipelineController::hyperparamsSnapshot() const {
  std::lock_guard<std::mutex> lk(hyper_mutex_);
  return hyperparams_;
}

void PipelineController::setHyperparams(const FusionHyperparams &h) {
  FusionHyperparams hp = h;
  syncIcpDepthFromRange(hp);
  {
    std::lock_guard<std::mutex> lk(hyper_mutex_);
    hyperparams_ = hp;
  }
  tracker_->setParams(hp.icp);
  tsdf_->setParams(hp.tsdf);
  if (preprocessor_) {
    preprocessor_->setSrScale(hp.sr_scale);
  }
  KFLOGF_INFO(
      "Pipeline",
      "New Hyperparameters Applied: D_min=%.2fm, D_max=%.2fm, TSDF_res=%.1fmm, SR_scale=%dx",
      hp.min_depth, hp.max_depth, hp.tsdf.voxel_size * 1000.0f, hp.sr_scale);
}

void PipelineController::stop() {
  std::lock_guard<std::mutex> ctrl_lk(control_mutex_);
  if (!running_.load())
    return;
  running_.store(false);
  state_.store(PipelineState::Stopped);

  KFLOG_INFO("Pipeline",
             "Stopping pipeline... Waiting for worker threads to join.");
  sensor_->stop();
  sensor_->setFrameCallback(nullptr); // Unset to avoid late callbacks

  // Wake up threads blocked on condition variables
  tracking_queue_cv_.notify_all();
  integration_queue_cv_.notify_all();

  if (tracking_thread_.joinable())
    tracking_thread_.join();
  if (integration_thread_.joinable())
    integration_thread_.join();
  if (meshing_thread_.joinable())
    meshing_thread_.join();

#ifdef CUDA_ENABLED
  if (use_gpu_.load()) {
      cudaDeviceSynchronize();
  }
#elif defined(HIP_ENABLED)
  if (use_gpu_.load()) {
      (void)hipDeviceSynchronize();
  }
#endif

  // FINAL EXTRACTION: Collect all voxels for a "Full Model" point cloud view
  KFLOG_INFO("Pipeline", "Extracting full model point cloud for final view...");
  auto global_frame = std::make_shared<sensor::FrameData>();
  tsdf_->extractGlobalPointCloud(global_frame->vertices, global_frame->rgb);
  global_frame->width = 1;
  global_frame->height = static_cast<int>(global_frame->vertices.size());
  global_frame->depth_meters.assign(global_frame->height, 1.0f);
  global_frame->pose = Eigen::Matrix4f::Identity();

  if (frame_ready_cb_) {
    frame_ready_cb_(*global_frame);
  }

#ifdef CUDA_ENABLED
    if (cuda_stream_) {
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }
    tsdf_->freeGPU();
    tracker_->freeGPU();
    for (int i = 0; i < 3; ++i) {
        model_buffers_.buffers[i]->d_vertices.reset();
        model_buffers_.buffers[i]->d_normals.reset();
        model_buffers_.buffers[i]->d_colors.reset();
    }
#elif defined(HIP_ENABLED)
    if (cuda_stream_) {
        (void)hipStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }
    tsdf_->freeGPU();
    tracker_->freeGPU();
    for (int i = 0; i < 3; ++i) {
        model_buffers_.buffers[i]->d_vertices.reset();
        model_buffers_.buffers[i]->d_normals.reset();
        model_buffers_.buffers[i]->d_colors.reset();
    }
#endif
  KFLOG_INFO("Pipeline", "Pipeline shutdown complete.");
}

void PipelineController::reset() {
    // stop() acquires and releases control_mutex_ internally.
    // There is an intentional TOCTOU window between stop() returning and the
    // re-lock below where another thread could call start(). This is acceptable
    // because the UI is expected to serialize control operations. If concurrent
    // control is ever needed, merge stop() body inline here under one lock.
    stop();
    std::lock_guard<std::mutex> ctrl_lk(control_mutex_);

    // Clear queues
    {
        std::lock_guard<std::mutex> lk(tracking_queue_mutex_);
        while (!raw_queue_.empty()) raw_queue_.pop();
    }
    {
        std::lock_guard<std::mutex> lk(integration_queue_mutex_);
        while (!integration_queue_.empty()) integration_queue_.pop();
    }

    mesh_extraction_requested_.store(false);

    {
        std::lock_guard<std::mutex> lk(model_buffers_.mtx);
        for (int i = 0; i < 3; ++i) {
            if (!model_buffers_.buffers[i]) continue;
            auto& mf = *model_buffers_.buffers[i];
            std::fill(mf.vertices.begin(), mf.vertices.end(), Eigen::Vector3f::Zero());
            std::fill(mf.normals.begin(), mf.normals.end(), Eigen::Vector3f::Zero());
        }
        model_buffers_.front_idx.store(0);
        model_buffers_.back_idx.store(1);
        model_buffers_.ready_idx.store(2);
    }

    tsdf_->reset();
    {
        std::lock_guard<std::mutex> lk(pose_mutex_);
        current_pose_ = Eigen::Matrix4f::Identity();
        last_pose_ = Eigen::Matrix4f::Identity();
    }
    first_frame_          = true;
    frame_count_          = 0;
    ui_skip_counter_      = 0;
    lost_log_counter_     = 0;
    success_log_counter_  = 0;
    hip_ui_skip_          = 0;
    metrics_      = PipelineMetrics{};
    if (preprocessor_) {
        preprocessor_->reset();
    }
    shared_mesh_.update(std::make_shared<meshing::MeshData>());
    state_.store(PipelineState::Idle);
    KFLOG_INFO("Pipeline", "System RESET: Volume cleared, pose re-centered, queues drained.");
}

void PipelineController::configurePreprocessor() {
    sensor::PreprocessBackend resolved = sensor::PreprocessBackend::CPU;
    preprocessor_ = sensor::makePreprocessor(preferred_backend_,
                                             use_gpu_.load(),
                                             cuda_stream_,
                                             &resolved);
    active_backend_.store(resolved);
    KFLOGF_INFO("Pipeline", "Preprocessor backend: requested=%s active=%s",
                sensor::backendName(preferred_backend_),
                sensor::backendName(resolved));
}

// Called from sensor capture thread — must be lightweight
void PipelineController::onRawFrame(std::shared_ptr<sensor::RawFrame> raw) {
  if (!running_.load())
    return;

  {
    std::lock_guard<std::mutex> lk(tracking_queue_mutex_);
    if (raw_queue_.size() < 3) {
      // Store raw frame — processing happens in tracking thread
      raw_queue_.push(std::move(raw));
    }
  }
  tracking_queue_cv_.notify_one();

  // Update capture FPS
  auto now = steady_clock::now();
  float dt = duration<float>(now - last_capture_time_).count();
  last_capture_time_ = now;

  int fc = 0;
  float inst_fps = 0.0f;
  {
    std::lock_guard<std::mutex> lk(metrics_mutex_);
    metrics_.capture_fps = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
    metrics_.frame_count = ++frame_count_;
    fc = metrics_.frame_count;
    inst_fps = metrics_.capture_fps;
  }
  if (fc % 150 == 0) {
    KFLOGF_DEBUG("Pipeline", "Sensor throughput: %d frames received @ %.2f FPS",
                 fc, inst_fps);
  }
}

void PipelineController::trackingLoop() {
  while (running_.load()) {
    std::shared_ptr<sensor::RawFrame> raw;
    {
      std::unique_lock<std::mutex> lk(tracking_queue_mutex_);
      tracking_queue_cv_.wait(
          lk, [&] { return !running_.load() || !raw_queue_.empty(); });
      if (!running_.load())
        break;
      if (raw_queue_.empty())
        continue;
      raw = raw_queue_.front();
      raw_queue_.pop();
    }

    // Build processed frame here (using pool)
    auto frame = acquireFreeData();
    if (!frame) {
        KFLOG_WARN("Pipeline", "FrameData pool exhausted — dropping raw frame (increase pool or slow capture)");
        sensor_->releaseFrame(std::move(raw)); // Don't forget to recycle raw
        continue;
    }
    
    frame->frame_id = raw->frame_id;
    float d_min = 0.3f, d_max = 5.0f;
    {
        std::lock_guard<std::mutex> lk(hyper_mutex_);
        d_min = hyperparams_.min_depth;
        d_max = hyperparams_.max_depth;
    }

    if (preprocessor_) {
        preprocessor_->process(*raw, d_min, d_max);
    }
    // TODO: Use upscaled RGB for better texture quality in TSDF integration (future scope)
    // Currently disabled as it causes black textures - needs further investigation
    // const auto& upscaled_rgb = preprocessor_->getSrRgbUpscaled();
    // sensor::buildFrameData(raw->depth.data(), upscaled_rgb.data(), *frame, d_min, d_max);
    sensor::buildFrameData(raw->depth.data(), raw->rgb.data(), *frame, d_min, d_max);
    sensor::computeNormals(*frame);
    
    // We can release 'raw' immediately after buildFrameData copies it
    sensor_->releaseFrame(std::move(raw));

    // Notify UI (throttled, safe shared_ptr copy)
    // REMOVED: To prevent 30FPS UI Flickering & Callback Fighting with integrationLoop.
    // The UI should only render the stable raycasted model from the integration thread.

    // First frame: initialize pose; skip tracking
    if (first_frame_) {
        first_frame_ = false;
        frame->pose = Eigen::Matrix4f::Identity();
        // Enqueue for integration
        {
            std::lock_guard<std::mutex> lk(integration_queue_mutex_);
            if (integration_queue_.size() < 3)
                integration_queue_.push(frame);
            else
                releaseData(std::move(frame));
        }
        integration_queue_cv_.notify_one();

        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            metrics_.tracking_ok = true;
            metrics_.icp_error   = 0.0f;
        }
        KFLOGF_INFO("Pipeline", "First frame accepted (ID: %lu). Initializing world origin.", frame->frame_id);
        continue;
    }

    // Race-condition guard: wait for the integration thread to finish the first raycast.
    // Without this, frame 2 would ICP against an empty model and immediately declare
    // tracking lost, locking the pipeline out of ever recovering.
    if (!model_ready_.load()) {
        // Re-enqueue frame as if it just arrived and spin-wait (effectively drops it)
        releaseData(std::move(frame));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        continue;
    }

    // Get model frame safely from TripleBuffer storage (ZERO-COPY)
    std::shared_ptr<tracking::ModelFrame> model_ref;
    model_ref = model_buffers_.buffers[model_buffers_.acquireFront()];

    // Run ICP
    Eigen::Matrix4f prev_pose;
    Eigen::Matrix4f predicted_pose;
    {
      std::lock_guard<std::mutex> lk(pose_mutex_);
      prev_pose = current_pose_;

      // Motion Model: predicted = current * (last_delta)
      // last_delta = last_pose.inv * current_pose
      Eigen::Matrix4f delta = last_pose_.inverse() * current_pose_;
      predicted_pose = current_pose_ * delta;
      last_pose_ = current_pose_;
    }

    tracking::ICPResult icp_result;
    bool is_lost = (state_.load() == PipelineState::TrackingLost);

    if (is_lost) {
      // RELOCALIZATION MODE: Multi-hypothesis search
      tracking::ICPParams recovery_params = hyperparams_.icp;
      recovery_params.dist_threshold *= 2.5f; 
      recovery_params.angle_threshold = 45.0f;
      
      // Increase iterations for recovery
      for (int l = 0; l < sensor::FramePyramid::LEVELS; ++l) {
          recovery_params.max_iterations[l] *= 2;
      }

      tracking::ICPParams original_params = tracker_->params();
      tracker_->setParams(recovery_params);

      // Hypothesis 1: Last known pose
      Eigen::Matrix4f h1_pose = prev_pose;
      
      // Hypothesis 2: Small forward motion (assuming user is moving towards object)
      Eigen::Matrix4f h2_pose = prev_pose;
      h2_pose(2,3) += 0.05f; 

      // Hypothesis 3: Zero velocity (if predicted_pose was used but failed)
      Eigen::Matrix4f h3_pose = prev_pose;

      std::vector<Eigen::Matrix4f> hypotheses = {h1_pose, h2_pose, h3_pose};
      tracking::ICPResult best_result;
      best_result.tracking_ok = false;
      best_result.inliers = 0;

      for (const auto& h_pose : hypotheses) {
          tracking::ICPResult res;
          if (use_gpu_.load()) {
            std::lock_guard<std::mutex> gpu_lk(gpu_mutex_);
#ifdef CUDA_ENABLED
            res = tracker_->trackGPU(
                preprocessor_->getGPUDepthMeters(),
                preprocessor_->getGPURgb(),
                frame->width, frame->height,
                *model_ref, h_pose, prev_pose
            );
#elif defined(HIP_ENABLED)
            res = tracker_->trackGPU(
                preprocessor_->getGPUDepthMeters(),
                preprocessor_->getGPURgb(),
                frame->width, frame->height,
                *model_ref, h_pose, prev_pose
            );
#endif
          } else {
            sensor::FramePyramid pyramid;
            sensor::buildFramePyramid(*frame, pyramid);
            res = tracker_->track(pyramid, *model_ref, h_pose, prev_pose);
          }

          // Always track the best-inlier hypothesis so diagnostics are
          // meaningful even when every hypothesis fails (tracking_ok==false).
          if (res.inliers > best_result.inliers) {
              best_result = res;
          }
          if (best_result.tracking_ok && best_result.inliers > 5000) break; // Good enough
      }
      
      icp_result = best_result;
      tracker_->setParams(original_params);
    } else {
      if (use_gpu_.load()) {
        std::lock_guard<std::mutex> gpu_lk(gpu_mutex_);
#ifdef CUDA_ENABLED
        icp_result = tracker_->trackGPU(
            preprocessor_->getGPUDepthMeters(),
            preprocessor_->getGPURgb(),
            frame->width, frame->height,
            *model_ref, predicted_pose, prev_pose
        );
        if (!icp_result.tracking_ok) {
          icp_result = tracker_->trackGPU(
              preprocessor_->getGPUDepthMeters(),
              preprocessor_->getGPURgb(),
              frame->width, frame->height,
              *model_ref, prev_pose, prev_pose
          );
        }
#elif defined(HIP_ENABLED)
        icp_result = tracker_->trackGPU(
            preprocessor_->getGPUDepthMeters(),
            preprocessor_->getGPURgb(),
            frame->width, frame->height,
            *model_ref, predicted_pose, prev_pose
        );
        if (!icp_result.tracking_ok) {
          icp_result = tracker_->trackGPU(
              preprocessor_->getGPUDepthMeters(),
              preprocessor_->getGPURgb(),
              frame->width, frame->height,
              *model_ref, prev_pose, prev_pose
          );
        }
#endif
      } else {
        sensor::FramePyramid pyramid;
        sensor::buildFramePyramid(*frame, pyramid);
        icp_result = tracker_->track(pyramid, *model_ref, predicted_pose, prev_pose);
        if (!icp_result.tracking_ok) {
          icp_result = tracker_->track(pyramid, *model_ref, prev_pose, prev_pose);
        }
      }
    }

    // Robust Camera Path Detection: Sanity check the pose update
    if (icp_result.tracking_ok) {
        Eigen::Matrix4f diff = prev_pose.inverse() * icp_result.pose;
        float trans_dist = diff.block<3,1>(0,3).norm();
        
        // Rodrigues rotation magnitude
        Eigen::Matrix3f R_diff = diff.block<3,3>(0,0);
        float trace = R_diff.trace();
        float angle = std::acos(std::max(-1.0f, std::min(1.0f, (trace - 1.0f) / 2.0f)));

        // If camera "teleported" more than 15cm or rotated more than 30 degrees in 33ms, 
        // it's almost certainly a tracking artifact/mismatch.
        if (trans_dist > 0.15f || angle > 0.52f) {
            KFLOGF_WARN("Pipeline", "Pose Sanity Check FAILED: dist=%.3fm, angle=%.1f deg. Rejecting pose.", 
                       trans_dist, angle * 180.0f / M_PI);
            icp_result.tracking_ok = false;
        }
    }

    // Update tracking fps
    auto now = steady_clock::now();
    float dt = duration<float>(now - last_tracking_time_).count();
    last_tracking_time_ = now;

    {
      std::lock_guard<std::mutex> lk(metrics_mutex_);
      metrics_.tracking_fps = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
      metrics_.icp_error = icp_result.error;
      metrics_.tracking_ok = icp_result.tracking_ok;
      if (icp_result.valid_live_points > 0) {
          metrics_.icp_overlap_pct = 100.0f * static_cast<float>(icp_result.valid_model_points)
                                           / static_cast<float>(icp_result.valid_live_points);
          metrics_.icp_valid_model = icp_result.valid_model_points;
      }
    }

        if (!icp_result.tracking_ok) {
            // lost_log_counter_ is a member (not static local) to avoid UB on thread restart.
            if (++lost_log_counter_ % 30 == 1) {
                const char* advice = "Move device slowly or improve scene geometry.";
                if (icp_result.inliers < 1000) advice = "Insufficient geometry overlap (point the sensor at a known surface).";
                else if (icp_result.error > 1e-3) advice = "High residual error (fast motion or dynamic objects).";
                else if (icp_result.dist_filtered > 10000) advice = "Too many points filtered by distance (too close/far).";

                KFLOGF_WARN("Pipeline", "%s ICP Failed: inliers=%d, residual=%f, valid_live=%d, valid_model=%d, dist_filt=%d, angle_filt=%d. Advice: %s",
                           (is_lost ? "[RELOCALIZING]" : "[TRACKING LOST]"),
                           icp_result.inliers, icp_result.error, 
                           icp_result.valid_live_points, icp_result.valid_model_points,
                           icp_result.dist_filtered, icp_result.angle_filtered, advice);
            }
        } else {
            // success_log_counter_ is a member (not static local) to avoid UB on thread restart.
            if (++success_log_counter_ % 300 == 0) {
                KFLOGF_INFO("Pipeline", "Tracking STABLE: avg_inliers=%d, avg_residual=%.6f", 
                           icp_result.inliers, icp_result.error);
            }
        }

        if (icp_result.tracking_ok) {
            {
                std::lock_guard<std::mutex> lk(pose_mutex_);
                current_pose_ = icp_result.pose;
            }
            frame->pose = icp_result.pose; 
            state_.store(PipelineState::Running);

            // Enqueue for integration
            {
                std::lock_guard<std::mutex> lk(integration_queue_mutex_);
                if (integration_queue_.size() < 3)
                    integration_queue_.push(frame);
                else
                    releaseData(std::move(frame)); 
            }
            integration_queue_cv_.notify_one();
        } else {
            // RELOCALIZATION: If tracking lost, don't update state or pose, but don't stop.
            // We just don't integrate the frame. This keeps the model "clean".
            if (!is_lost) {
                 KFLOG_WARN("Pipeline", "Tracking lost! Suspension of TSDF integration. Entering relocalization mode...");
                 if (preprocessor_) {
                     preprocessor_->resetTemporalState();
                 }
                 // Clear the integration queue to prevent "garbage" poses from being integrated
                 std::lock_guard<std::mutex> lk(integration_queue_mutex_);
                 std::queue<std::shared_ptr<sensor::FrameData>> empty;
                 std::swap(integration_queue_, empty);
            }
            state_.store(PipelineState::TrackingLost);
            releaseData(std::move(frame));
        }
    }
}

void PipelineController::integrationLoop() {
  static constexpr int MESH_TRIGGER_FRAMES =
      5; // trigger mesh update every N integrated frames
  int frames_since_mesh = 0;
  
  float d_min = 0.3f, d_max = 5.0f;
  {
      std::lock_guard<std::mutex> lk(hyper_mutex_);
      d_min = hyperparams_.min_depth;
      d_max = hyperparams_.max_depth;
  }

  while (running_.load()) {
    std::shared_ptr<sensor::FrameData> frame;
    {
      std::unique_lock<std::mutex> lk(integration_queue_mutex_);
      integration_queue_cv_.wait(
          lk, [&] { return !running_.load() || !integration_queue_.empty(); });
      if (!running_.load())
        break;
      if (integration_queue_.empty())
        continue;
      frame = integration_queue_.front();
      integration_queue_.pop();
    }

    {
      utils::ScopedTimer t("TSDF Integration");
      // gpu_mutex_ MUST be acquired before tsdf_mutex_ (consistent lock order).
      // Always acquire gpu_mutex_ first to prevent deadlock with the meshing thread.
      std::unique_lock<std::mutex> gpu_lk(gpu_mutex_);
      std::unique_lock<std::shared_mutex> lk_tsdf(tsdf_mutex_);
#ifdef CUDA_ENABLED
      if (use_gpu_.load()) {
          tsdf_->integrate(
              frame->depth_meters.data(),
              frame->rgb.data(),
              frame->pose,
              static_cast<float>(sensor::FX), static_cast<float>(sensor::FY),
              static_cast<float>(sensor::CX), static_cast<float>(sensor::CY),
              frame->width, frame->height, d_min, d_max);
      } else {
          gpu_lk.unlock(); // CPU path doesn't need GPU serialization
          tsdf_->integrate(
              frame->depth_meters.data(), frame->rgb.data(), frame->pose,
              static_cast<float>(sensor::FX), static_cast<float>(sensor::FY),
              static_cast<float>(sensor::CX), static_cast<float>(sensor::CY),
              frame->width, frame->height, d_min, d_max);
      }
#elif defined(HIP_ENABLED)
      if (use_gpu_.load()) {
          tsdf_->integrate(
              frame->depth_meters.data(),
              frame->rgb.data(),
              frame->pose,
              static_cast<float>(sensor::FX), static_cast<float>(sensor::FY),
              static_cast<float>(sensor::CX), static_cast<float>(sensor::CY),
              frame->width, frame->height, d_min, d_max);
      } else {
          gpu_lk.unlock(); // CPU path doesn't need GPU serialization
          tsdf_->integrate(
              frame->depth_meters.data(), frame->rgb.data(), frame->pose,
              static_cast<float>(sensor::FX), static_cast<float>(sensor::FY),
              static_cast<float>(sensor::CX), static_cast<float>(sensor::CY),
              frame->width, frame->height, d_min, d_max);
      }
#else
      gpu_lk.unlock(); // CPU-only build: no GPU serialization needed
      tsdf_->integrate(
          frame->depth_meters.data(), frame->rgb.data(), frame->pose,
          static_cast<float>(sensor::FX), static_cast<float>(sensor::FY),
          static_cast<float>(sensor::CX), static_cast<float>(sensor::CY),
          frame->width, frame->height, d_min, d_max);
#endif
    }

    // 2. Generate model frame for tracking (Ping-Pong back buffer)
    int integrated_count = tsdf_->integratedFrames();
    if (integrated_count % 1 == 0) { // Raycast every frame for better tracking
      int back = model_buffers_.back_idx.load();
      auto &model_back = *(model_buffers_.buffers[back]);

#ifndef HIP_ENABLED
      // KIN-FORK(cpu-preview): Upstream defines this block only in the non-GPU
      // #else branch and reaches it from the CUDA branch via `goto cpu_raycast`,
      // whose label is compiled out in CUDA builds (CUDA never compiled).
      // Extracted so CUDA (GPU toggled off at runtime) and CPU share one preview
      // path. HIP keeps upstream shape to minimize merge conflicts.
      auto emitCpuPreview = [&]() {
          // Optimization for CPU: Downsample raycast for UI preview to reduce
          // jitter/lag
          {
              std::shared_lock<std::shared_mutex> lk_tsdf(tsdf_mutex_);
              tsdf_->raycast(frame->pose, static_cast<float>(sensor::FX),
                             static_cast<float>(sensor::FY),
                             static_cast<float>(sensor::CX),
                             static_cast<float>(sensor::CY), sensor::DEPTH_WIDTH,
                             sensor::DEPTH_HEIGHT, model_back.vertices.data(),
                             model_back.normals.data(), model_back.colors.data());
          }

          if (frame_ready_cb_) {
            size_t n = sensor::DEPTH_WIDTH * sensor::DEPTH_HEIGHT;
            auto ui_frame = std::make_shared<sensor::FrameData>();

            // For CPU, we use a 2x downsample for the UI preview to keep it
            // responsive (8fps -> 15fps feel)
            constexpr int step = 2;
            ui_frame->width = sensor::DEPTH_WIDTH / step;
            ui_frame->height = sensor::DEPTH_HEIGHT / step;
            int n_ui = ui_frame->width * ui_frame->height;
            ui_frame->vertices.resize(n_ui);
            ui_frame->rgb.resize(n_ui * 3);
            ui_frame->depth_meters.resize(n_ui);

            for (int y = 0; y < ui_frame->height; ++y) {
              for (int x = 0; x < ui_frame->width; ++x) {
                int idx_full = (y * step) * sensor::DEPTH_WIDTH + (x * step);
                int idx_ui = y * ui_frame->width + x;
                auto& v = model_back.vertices[idx_full];
                ui_frame->vertices[idx_ui] = v;
                ui_frame->rgb[idx_ui * 3 + 0] = model_back.colors[idx_full * 3 + 0];
                ui_frame->rgb[idx_ui * 3 + 1] = model_back.colors[idx_full * 3 + 1];
                ui_frame->rgb[idx_ui * 3 + 2] = model_back.colors[idx_full * 3 + 2];
                ui_frame->depth_meters[idx_ui] = (v.z() != 0.0f || v.x() != 0.0f || v.y() != 0.0f) ? 1.0f : 0.0f;
              }
            }
            ui_frame->pose = Eigen::Matrix4f::Identity();

            QMetaObject::invokeMethod(
                qApp,
                [this, ui_frame]() {
                  if (frame_ready_cb_)
                    frame_ready_cb_(*ui_frame);
                },
                Qt::QueuedConnection);
          }
      };
#endif

#ifdef CUDA_ENABLED
      if (use_gpu_.load()) {
          {
              // Raycast must be serialized: it launches a GPU kernel over the same
              // TSDF volume that integration just wrote. Use gpu_mutex_ not shared_lock.
              std::lock_guard<std::mutex> gpu_lk(gpu_mutex_);
              std::shared_lock<std::shared_mutex> lk_tsdf(tsdf_mutex_);
              tsdf_->raycastGPU(frame->pose, static_cast<float>(sensor::FX),
                                static_cast<float>(sensor::FY),
                                static_cast<float>(sensor::CX),
                                static_cast<float>(sensor::CY), sensor::DEPTH_WIDTH,
                                sensor::DEPTH_HEIGHT, model_back.d_vertices.get(),
                                model_back.d_normals.get(), model_back.d_colors.get());
          }
          // Sync to CPU for UI preview
          if (frame_ready_cb_) {
            size_t n = sensor::DEPTH_WIDTH * sensor::DEPTH_HEIGHT;
            std::vector<float3> h_v(n);
            std::vector<float3> h_n(n);
            std::vector<uchar3> h_c(n);

            cudaMemcpy(h_v.data(), model_back.d_vertices.get(), n * sizeof(float3), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_n.data(), model_back.d_normals.get(), n * sizeof(float3), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_c.data(), model_back.d_colors.get(), n * sizeof(uchar3), cudaMemcpyDeviceToHost);

            auto ui_frame = std::make_shared<sensor::FrameData>();
            ui_frame->width = sensor::DEPTH_WIDTH;
            ui_frame->height = sensor::DEPTH_HEIGHT;
            ui_frame->vertices.resize(n);
            ui_frame->rgb.resize(n * 3);
            ui_frame->depth_meters.resize(n);
            ui_frame->pose = Eigen::Matrix4f::Identity();

            for (size_t i = 0; i < n; ++i) {
              ui_frame->vertices[i] = Eigen::Vector3f(h_v[i].x, h_v[i].y, h_v[i].z);
              ui_frame->rgb[i*3+0] = h_c[i].x;
              ui_frame->rgb[i*3+1] = h_c[i].y;
              ui_frame->rgb[i*3+2] = h_c[i].z;
              ui_frame->depth_meters[i] = (h_v[i].z != 0.0f || h_v[i].x != 0.0f || h_v[i].y != 0.0f) ? 1.0f : 0.0f;
            }

            QMetaObject::invokeMethod(
                qApp,
                [this, ui_frame]() {
                  if (frame_ready_cb_)
                    frame_ready_cb_(*ui_frame);
                },
                Qt::QueuedConnection);
          }
      } else {
          emitCpuPreview();
      }
#elif defined(HIP_ENABLED)
      if (use_gpu_.load()) {
          {
              // Raycast serialized via gpu_mutex_ — prevents the meshing thread from
              // simultaneously running a heavy marching cubes kernel on the 5650u.
              std::lock_guard<std::mutex> gpu_lk(gpu_mutex_);
              std::shared_lock<std::shared_mutex> lk_tsdf(tsdf_mutex_);
              tsdf_->raycastGPU(frame->pose, static_cast<float>(sensor::FX),
                                static_cast<float>(sensor::FY),
                                static_cast<float>(sensor::CX),
                                static_cast<float>(sensor::CY), sensor::DEPTH_WIDTH,
                                sensor::DEPTH_HEIGHT, model_back.d_vertices.get(),
                                model_back.d_normals.get(), model_back.d_colors.get());
          }
          // Sync to CPU for UI preview.
          // Throttle: on the shared AMD iGPU (5650u) every hipMemcpy stalls the GPU
          // pipeline and starves the GNOME compositor. Only send a UI update every
          // HIP_UI_PREVIEW_INTERVAL integrated frames to keep the display responsive.
          static constexpr int HIP_UI_PREVIEW_INTERVAL = 3;
          if (frame_ready_cb_ && (++hip_ui_skip_ % HIP_UI_PREVIEW_INTERVAL == 0)) {
            size_t n = sensor::DEPTH_WIDTH * sensor::DEPTH_HEIGHT;
            std::vector<float3> h_v(n);
            std::vector<uchar3> h_c(n);

            // raycastGPU already called hipDeviceSynchronize internally.
            // Only copy what the UI actually needs (vertices + colors, skip normals).
            (void)hipMemcpy(h_v.data(), model_back.d_vertices.get(), n * sizeof(float3), hipMemcpyDeviceToHost);
            (void)hipMemcpy(h_c.data(), model_back.d_colors.get(), n * sizeof(uchar3), hipMemcpyDeviceToHost);

            auto ui_frame = std::make_shared<sensor::FrameData>();
            ui_frame->width  = sensor::DEPTH_WIDTH;
            ui_frame->height = sensor::DEPTH_HEIGHT;
            // Use resize (not reserve) so rgb is properly sized for index access in uploadPointCloud.
            ui_frame->vertices.resize(n);
            ui_frame->rgb.resize(n * 3);
            // Populate depth_meters so uploadPointCloud uses the depth-based validity path.
            ui_frame->depth_meters.resize(n);
            ui_frame->pose = Eigen::Matrix4f::Identity();

            for (size_t i = 0; i < n; ++i) {
              ui_frame->vertices[i] = Eigen::Vector3f(h_v[i].x, h_v[i].y, h_v[i].z);
              ui_frame->rgb[i*3+0]  = h_c[i].x;
              ui_frame->rgb[i*3+1]  = h_c[i].y;
              ui_frame->rgb[i*3+2]  = h_c[i].z;
              // A non-zero sentinel depth marks valid raycasted hits; zero means miss.
              ui_frame->depth_meters[i] = (h_v[i].z != 0.0f || h_v[i].x != 0.0f || h_v[i].y != 0.0f) ? 1.0f : 0.0f;
            }

            QMetaObject::invokeMethod(
                qApp,
                [this, ui_frame]() {
                  if (frame_ready_cb_)
                    frame_ready_cb_(*ui_frame);
                },
                Qt::QueuedConnection);
          }
      } else {
          // Fall through to CPU raycast
      }
#else
      emitCpuPreview();  // KIN-FORK(cpu-preview): was `goto cpu_raycast` + label
#endif
      model_buffers_.swap();
      // Signal trackingLoop that at least one valid model frame exists.
      model_ready_.store(true);
    }

    {
      std::lock_guard<std::mutex> lk(metrics_mutex_);
      metrics_.integrated_frames = integrated_count;
      // GPU path: usageFraction() iterates 2M CPU voxels that are stale/empty
      // when the volume lives on the GPU. Skip this expensive CPU loop.
      if (!use_gpu_.load()) {
          metrics_.volume_usage_pct = tsdf_->usageFraction() * 100.0f;
      }
    }

    if (integrated_count % 50 == 0 && integrated_count > 0) {
      KFLOGF_INFO("Pipeline",
                  "TSDF Status: %d frames integrated",
                  integrated_count);
    }

    // Return frame to pool!
    releaseData(std::move(frame));

    ++frames_since_mesh;
    if (frames_since_mesh >= MESH_TRIGGER_FRAMES) {
      frames_since_mesh = 0;
      mesh_extraction_requested_.store(true);
    }
  }
}

void PipelineController::meshingLoop() {
  while (running_.load()) {
    // Poll for mesh extraction request
    if (!mesh_extraction_requested_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }
    mesh_extraction_requested_.store(false);
    is_meshing_.store(true);

    mesh_extract_progress_.store(0.0f);
    KFLOG_INFO("Pipeline",
               "Marching Cubes: Extracting mesh from TSDF volume...");
    std::shared_ptr<meshing::MeshData> mesh;
    {
      utils::ScopedTimer t("Mesh Extraction");
      // gpu_mutex_ first to match lock ordering in integrationLoop.
      // This is the heaviest GPU job (~400ms): holding gpu_mutex_ prevents
      // tracking and integration from hammering the ROCm ring concurrently.
      std::unique_lock<std::mutex> gpu_lk(gpu_mutex_);
      std::shared_lock<std::shared_mutex> lk_tsdf(tsdf_mutex_);
#ifdef CUDA_ENABLED
      if (use_gpu_.load()) {
          mesh = cubes_->extractGPU(*tsdf_);
      } else {
          gpu_lk.unlock(); // CPU mesh extraction doesn't need GPU serialization
          mesh = cubes_->extract(*tsdf_, [this](float p) {
            mesh_extract_progress_.store(p);
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            metrics_.mesh_extract_pct = p * 100.0f;
          });
      }
#elif defined(HIP_ENABLED)
      if (use_gpu_.load()) {
          mesh = cubes_->extractGPU(*tsdf_);
      } else {
          gpu_lk.unlock(); // CPU mesh extraction doesn't need GPU serialization
          mesh = cubes_->extract(*tsdf_, [this](float p) {
            mesh_extract_progress_.store(p);
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            metrics_.mesh_extract_pct = p * 100.0f;
          });
      }
#else
      gpu_lk.unlock(); // CPU mesh extraction doesn't need GPU serialization
      mesh = cubes_->extract(*tsdf_, [this](float p) {
        mesh_extract_progress_.store(p);
        std::lock_guard<std::mutex> lk(metrics_mutex_);
        metrics_.mesh_extract_pct = p * 100.0f;
      });
#endif
    }

    {
      std::lock_guard<std::mutex> lk(metrics_mutex_);
      metrics_.mesh_triangles = mesh ? mesh->triangleCount() : 0;
      metrics_.mesh_extract_pct = 100.0f;
    }

    if (mesh && !mesh->empty()) {
      KFLOGF_INFO("Pipeline",
                  "Mesh Extraction SUCCESS: %zu triangles, %zu vertices ready "
                  "for rendering.",
                  mesh->triangleCount(), mesh->positions.size());
      shared_mesh_.update(std::move(mesh));
      if (mesh_ready_cb_)
        mesh_ready_cb_();
    } else {
      KFLOG_WARN("Pipeline", "Mesh Extraction EMPTY: Volume might be too "
                             "sparse or clipping values too aggressive.");
      if (mesh_ready_cb_)
        mesh_ready_cb_();
    }
    is_meshing_.store(false);
  }
}

bool PipelineController::exportPLY(const std::string &path) {
  uint64_t ver;
  auto mesh = shared_mesh_.snapshot(ver);
  if (!mesh || mesh->empty()) {
    std::cout << "[Pipeline] No mesh yet, extracting via meshingLoop...\n";
    mesh_extraction_requested_.store(true);
    int waits = 0;
    while (waits < 100 &&
           (mesh_extraction_requested_.load() || is_meshing_.load())) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      waits++;
    }
    mesh = shared_mesh_.snapshot(ver);
  }
  if (!mesh || mesh->empty()) {
    std::cerr << "[Pipeline] No mesh to export — scan more frames first.\n";
    return false;
  }
  {
    std::lock_guard<std::mutex> lk(metrics_mutex_);
    metrics_.export_pct = 50.0f;
  }
  bool ok = export_io::PLYExporter::writeBinary(*mesh, path);
  {
    std::lock_guard<std::mutex> lk(metrics_mutex_);
    metrics_.export_pct = ok ? 100.0f : 0.0f;
  }
  return ok;
}

bool PipelineController::exportGLB(const std::string &path) {
  uint64_t ver;
  auto mesh = shared_mesh_.snapshot(ver);
  if (!mesh || mesh->empty()) {
    std::cout << "[Pipeline] No mesh yet, extracting via meshingLoop...\n";
    mesh_extraction_requested_.store(true);
    int waits = 0;
    while (waits < 100 &&
           (mesh_extraction_requested_.load() || is_meshing_.load())) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      waits++;
    }
    mesh = shared_mesh_.snapshot(ver);
  }
  if (!mesh || mesh->empty()) {
    std::cerr << "[Pipeline] No mesh to export — scan more frames first.\n";
    return false;
  }
  {
    std::lock_guard<std::mutex> lk(metrics_mutex_);
    metrics_.export_pct = 50.0f;
  }
  bool ok = export_io::GLBExporter::write(*mesh, path);
  {
    std::lock_guard<std::mutex> lk(metrics_mutex_);
    metrics_.export_pct = ok ? 100.0f : 0.0f;
  }
  return ok;
}

std::shared_ptr<sensor::FrameData> PipelineController::acquireFreeData() {
  std::lock_guard<std::mutex> lk(data_pool_state_->mutex);
  if (data_pool_state_->free_data_queue.empty())
    return nullptr;

  auto *raw_ptr = data_pool_state_->free_data_queue.front();
  data_pool_state_->free_data_queue.pop();

  // Zero-out or reset frame data if needed
  // raw_ptr->reset();

  auto state = data_pool_state_;
  return std::shared_ptr<sensor::FrameData>(
      raw_ptr, [state, raw_ptr](sensor::FrameData *) {
        std::lock_guard<std::mutex> lk_inner(state->mutex);
        state->free_data_queue.push(raw_ptr);
      });
}

void PipelineController::releaseData(std::shared_ptr<sensor::FrameData>) {
  // Managed by custom deleter
}

} // namespace app
} // namespace kfusion
