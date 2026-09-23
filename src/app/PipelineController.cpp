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
    syncTsdfDepthFromRange(hyperparams_);
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

PipelineController::~PipelineController() {
    stop();
    std::lock_guard<std::mutex> ctrl_lk(control_mutex_);
    releaseGpuResources();
}

bool PipelineController::start() {
    return startInternal(true);
}

#ifdef AZU_PIPELINE_TEST_SEAM
// Test seam: identical startup EXCEPT sensor_->init()/sensor_->start() are
// bypassed, so headless tests never open a Kinect device. Everything else —
// state reset, back-end selection, preprocessor, worker threads — is the real
// production code path.
bool PipelineController::startWithoutSensorForTests() {
    return startInternal(false);
}
#endif

bool PipelineController::startInternal(bool engage_sensor) {
  std::lock_guard<std::mutex> ctrl_lk(control_mutex_);
  if (running_.load())
    return true;

    running_.store(true);
    state_.store(PipelineState::Running);
    first_frame_          = true;
    model_ready_.store(false);
    frame_count_          = 0;
    lost_log_counter_     = 0;
    success_log_counter_  = 0;
    hip_ui_skip_          = 0;
    {
        // The motion model belongs to ONE session. last_pose_ is the previous
        // frame's pose, and the pose-to-pose predictor forms
        // delta = last_pose_.inverse() * current_pose_; leaving a pose from an
        // earlier session here makes the FIRST prediction of the new session a
        // jump across the stop/start gap (and across any reset in between),
        // against a model that has not seen that motion. start() therefore clears
        // the whole model: the first tracked frame is predicted from identity by
        // identity, i.e. no stale first prediction is possible, and start() never
        // requires a prior reset to get that.
        std::lock_guard<std::mutex> lk(pose_mutex_);
        current_pose_ = Eigen::Matrix4f::Identity();
        last_pose_    = Eigen::Matrix4f::Identity();
    }
    // Set frame callback before starting capture
    sensor_->setFrameCallback([this](std::shared_ptr<sensor::RawFrame> raw) {
        onRawFrame(std::move(raw));
    });

    if (!engage_sensor) {
        KFLOG_WARN("Pipeline",
                   "TEST SEAM active: sensor init/start bypassed; RawFrames must "
                   "arrive via injectRawFrameForTests().");
    } else if (!sensor_->init()) {
        KFLOG_ERROR("Pipeline", "Sensor initialization FAILED. Is the Kinect connected?");
        running_.store(false);
        state_.store(PipelineState::Error);
        return false;
    }

    if (engage_sensor && !sensor_->start()) {
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
            // initGPU() is idempotent; it re-allocates only after freeGPU()
            // (reset or a volume-resolution change).
            tsdf_->initGPU();
            if (!gpu_resources_ready_) {
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
                gpu_resources_ready_ = true;
            }
            use_gpu_ = true;
            tsdf_->setGPUEnabled(true);
            KFLOG_INFO("Pipeline", "CUDA hardware acceleration ENABLED (NVIDIA GPU detected).");
        } catch (const std::exception& e) {
            releaseGpuResources();
            use_gpu_ = false;
            tsdf_->setGPUEnabled(false);
            KFLOGF_WARN("Pipeline", "CUDA initialization FAILED: %s. Falling back to CPU mode.", e.what());
        }
    }
#elif defined(HIP_ENABLED)
    if (preferred_backend_ == sensor::PreprocessBackend::CPU || preferred_backend_ == sensor::PreprocessBackend::CUDA) {
        use_gpu_ = false;
        tsdf_->setGPUEnabled(false);
        KFLOG_INFO("Pipeline", "Using CPU-only path (User explicitly requested or incompatible backend).");
    } else {
        try {
            // initGPU() is idempotent; it re-allocates only after freeGPU()
            // (reset or a volume-resolution change).
            tsdf_->initGPU();
            if (!gpu_resources_ready_) {
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
                gpu_resources_ready_ = true;
            }
            use_gpu_ = true;
            tsdf_->setGPUEnabled(true);
            KFLOG_INFO("Pipeline", "HIP/ROCm hardware acceleration ENABLED (AMD iGPU detected).");
        } catch (const std::exception& e) {
            releaseGpuResources();
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

    // A freshly built preprocessor carries its own default SR scale, so the
    // owner's value is pushed here; otherwise a scale set before start() would
    // silently apply only on the next setHyperparams().
    const int start_sr_scale = hyperparamsSnapshot().sr_scale;
    configurePreprocessor();
    {
        std::lock_guard<std::mutex> pp_lk(preprocessor_mutex_);
        if (preprocessor_) {
            preprocessor_->reset();
            preprocessor_->setSrScale(start_sr_scale);
        }
    }
#ifdef AZU_PIPELINE_TEST_SEAM
    {
        std::lock_guard<std::mutex> obs_lk(seam_obs_.mtx);
        seam_obs_.applied_sr_scale = preprocessor_ ? start_sr_scale : 0;
    }
#endif

    // The requested worker count is pushed here (workers not yet launched,
    // control_mutex_ held) so a request parked by a previous running session,
    // or one that raced with stop(), cannot be lost.
    threads_pending_.store(false);
    applyThreadCount(num_threads_.load());

    // Clear residual work and reset each queue's guarded shutdown predicate
    // before launching workers, so start/stop is idempotent: a previous stop()
    // may have left frames queued and left *_shutdown_ latched true. A worker
    // must observe a clean (false) predicate at startup or it would exit on its
    // first wait. Lock order: control_mutex_ (held) -> tracking_queue_mutex_
    // (released) -> integration_queue_mutex_ (released); never both queue
    // mutexes at once, so this cannot invert the order stop() uses.
    {
        std::lock_guard<std::mutex> lk(tracking_queue_mutex_);
        tracking_shutdown_ = false;
        while (!raw_queue_.empty()) raw_queue_.pop();
    }
    {
        std::lock_guard<std::mutex> lk(integration_queue_mutex_);
        integration_shutdown_ = false;
        while (!integration_queue_.empty()) integration_queue_.pop();
    }
    // The mesh worker's shutdown predicate follows the same guarded-queue rule,
    // and its version counters stay monotonic across restarts: bumping nothing
    // here keeps (served <= claimed <= requested) meaningful over a stop/start,
    // while clearing shutdown lets the fresh worker wait instead of exiting.
    // The cadence clock is primed to the epoch so the first integrated frame of
    // this session is immediately due for a mesh request.
    {
        std::lock_guard<std::mutex> lk(mesh_requests_.mtx);
        mesh_requests_.shutdown = false;
        mesh_requests_.cv.notify_all();
    }
    last_mesh_request_ = std::chrono::microseconds::zero();

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
  syncTsdfDepthFromRange(hp);

  // control_mutex_ serialises this whole application against start()/stop()/
  // reset(), and each component lock is taken ALONE (never nested), so no order
  // between tracker/tsdf/preprocessor can ever invert. hyper_mutex_ stays a leaf.
  std::lock_guard<std::mutex> ctrl_lk(control_mutex_);
  {
    std::lock_guard<std::mutex> lk(hyper_mutex_);
    hyperparams_ = hp;
  }
  {
    std::lock_guard<std::mutex> tr_lk(tracker_mutex_);
    tracker_->setParams(hp.icp);
  }
  {
    // Volume parameter replacement reallocates and clears voxels_, so it must be
    // exclusive against the integration/meshing/raycast readers that already
    // take this lock. gpu_mutex_ is NOT needed here: this branch never touches
    // the device (the backends are deferred for this plan).
    std::unique_lock<std::shared_mutex> tsdf_lk(tsdf_mutex_);
    tsdf_->setParams(hp.tsdf);
  }
  {
    std::lock_guard<std::mutex> pp_lk(preprocessor_mutex_);
    if (preprocessor_) {
      preprocessor_->setSrScale(hp.sr_scale);
    }
#ifdef AZU_PIPELINE_TEST_SEAM
    std::lock_guard<std::mutex> obs_lk(seam_obs_.mtx);
    seam_obs_.applied_sr_scale = preprocessor_ ? hp.sr_scale : 0;
#endif
  }
  KFLOGF_INFO(
      "Pipeline",
      "New Hyperparameters Applied: D_min=%.2fm, D_max=%.2fm, TSDF_res=%.1fmm, SR_scale=%dx",
      hp.min_depth, hp.max_depth, hp.tsdf.voxel_size * 1000.0f, hp.sr_scale);
}

void PipelineController::setFrameReadyCallback(FrameReadyCallback cb) {
  std::lock_guard<std::mutex> lk(callback_mutex_);
  frame_ready_cb_ = std::move(cb);
}

void PipelineController::setMeshReadyCallback(MeshReadyCallback cb) {
  std::lock_guard<std::mutex> lk(callback_mutex_);
  mesh_ready_cb_ = std::move(cb);
}

FrameReadyCallback PipelineController::frameReadyCallbackCopy() const {
  std::lock_guard<std::mutex> lk(callback_mutex_);
  return frame_ready_cb_;
}

MeshReadyCallback PipelineController::meshReadyCallbackCopy() const {
  std::lock_guard<std::mutex> lk(callback_mutex_);
  return mesh_ready_cb_;
}

bool PipelineController::hasFrameReadyCallback() const {
  std::lock_guard<std::mutex> lk(callback_mutex_);
  return static_cast<bool>(frame_ready_cb_);
}

void PipelineController::applyThreadCount(int n) {
  {
    std::lock_guard<std::mutex> tr_lk(tracker_mutex_);
    tracker_->setNumThreads(n);
  }
#ifdef AZU_PIPELINE_TEST_SEAM
  std::lock_guard<std::mutex> obs_lk(seam_obs_.mtx);
  seam_obs_.applied_threads = n;
#endif
}

void PipelineController::applyPendingThreadCount() {
  if (!threads_pending_.exchange(false)) {
    return;
  }
  applyThreadCount(pending_num_threads_.load());
}

void PipelineController::setNumThreads(int n) {
  // Publish the request first; threads_pending_ is the release/acquire edge that
  // makes the parked value visible to the worker that consumes it.
  num_threads_.store(n);
  const auto park = [this, n]() {
    pending_num_threads_.store(n);
    threads_pending_.store(true);
  };
  if (running_.load()) {
    park();
    return;
  }
  // Stopped: apply now, but re-check under control_mutex_ so a start() that won
  // the race cannot have a worker already tracking with the old count.
  std::lock_guard<std::mutex> ctrl_lk(control_mutex_);
  if (running_.load()) {
    park();
    return;
  }
  applyThreadCount(n);
}

void PipelineController::stop() {
  std::lock_guard<std::mutex> ctrl_lk(control_mutex_);
  if (!running_.load())
    return;
  running_.store(false);

  KFLOG_INFO("Pipeline",
             "Stopping pipeline... Waiting for worker threads to join.");
  sensor_->stop();
  sensor_->setFrameCallback(nullptr); // Unset to avoid late callbacks

  // Wake workers with NO lost wakeup: flip each queue's guarded shutdown
  // predicate and notify while STILL HOLDING that queue's mutex. A worker
  // evaluates its predicate and registers on the cv under the same mutex, so
  // stop() cannot slip a state change into the "checked-but-not-yet-waiting"
  // window: either stop() takes the mutex first (the worker then observes
  // shutdown=true and never blocks) or the worker registers first (this
  // notify_all wakes it). running_ alone cannot do this because it is stored
  // outside the queue mutex.
  //
  // Lock order: control_mutex_ (held) -> tracking_queue_mutex_ (released) ->
  // integration_queue_mutex_ (released). The two queue mutexes are never held
  // at once, and no worker path acquires control_mutex_ while holding a queue
  // mutex, so there is no ABBA inversion in either direction.
  {
      std::lock_guard<std::mutex> lk(tracking_queue_mutex_);
      tracking_shutdown_ = true;
      tracking_queue_cv_.notify_all();
  }
  {
      std::lock_guard<std::mutex> lk(integration_queue_mutex_);
      integration_shutdown_ = true;
      integration_queue_cv_.notify_all();
  }
  // Same guarded-predicate rule for the meshing worker: flip and notify while
  // holding mesh_requests_.mtx, so a worker that is between "checked the
  // predicate" and "registered on the cv" cannot miss the shutdown.
  {
      std::lock_guard<std::mutex> lk(mesh_requests_.mtx);
      mesh_requests_.shutdown = true;
      mesh_requests_.cv.notify_all();
  }
#ifdef AZU_PIPELINE_TEST_SEAM
  // A seam test may stop() while a worker is parked at a test gate or inside the
  // mesh hook; releasing them here keeps join() from waiting on a park that only
  // the test could lift.
  releaseWorkerGate(tracking_gate_);
  releaseWorkerGate(integration_gate_);
  {
      std::lock_guard<std::mutex> lk(mesh_hook_.mtx);
      mesh_hook_.holds = 0;
      mesh_hook_.released = true;
      mesh_hook_.cv.notify_all();
  }
#endif

  if (tracking_thread_.joinable())
    tracking_thread_.join();
  if (integration_thread_.joinable())
    integration_thread_.join();
  if (meshing_thread_.joinable())
    meshing_thread_.join();

  // Terminal state is published only once every writer is gone: trackingLoop
  // stores Running/TrackingLost as it finishes a frame, so storing Stopped
  // before the join let a mid-frame worker overwrite it afterwards.
  state_.store(PipelineState::Stopped);

#ifdef CUDA_ENABLED
  if (use_gpu_.load()) {
      (void)cudaDeviceSynchronize();
  }
#elif defined(HIP_ENABLED)
  if (use_gpu_.load()) {
      (void)hipDeviceSynchronize();
  }
#endif
  // No stop-time full-volume point-cloud extraction: it cost O(volume) on the
  // caller's thread, delivered the frame callback synchronously (GL off the GUI
  // thread when stop() came from a worker), and on CUDA its cudaMalloc could
  // throw out of stop() and abort the process. The last published mesh and
  // preview stay on screen; GPU resources stay allocated so a stopped scan can
  // still be meshed, exported, or resumed.
  KFLOG_INFO("Pipeline", "Pipeline shutdown complete.");
}

void PipelineController::releaseGpuResources() noexcept {
#ifdef CUDA_ENABLED
    if (cuda_stream_) {
        (void)cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }
#elif defined(HIP_ENABLED)
    if (cuda_stream_) {
        (void)hipStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }
#endif
#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
    tsdf_->freeGPU();
    tracker_->freeGPU();
    for (int i = 0; i < 3; ++i) {
        model_buffers_.buffers[i]->d_vertices.reset();
        model_buffers_.buffers[i]->d_normals.reset();
        model_buffers_.buffers[i]->d_colors.reset();
    }
#endif
    gpu_resources_ready_ = false;
}

void PipelineController::onWorkerFault(const char* worker, const char* what) noexcept {
    KFLOGF_ERROR("Pipeline", "%s worker FAILED: %s. Pipeline halted; press Stop, then Reset.",
                 worker, what);
    state_.store(PipelineState::Error);
}

void PipelineController::reset() {
    // stop() acquires and releases control_mutex_ internally.
    // There is an intentional TOCTOU window between stop() returning and the
    // re-lock below where another thread could call start(). This is acceptable
    // because the UI is expected to serialize control operations. If concurrent
    // control is ever needed, merge stop() body inline here under one lock.
    stop();
    std::lock_guard<std::mutex> ctrl_lk(control_mutex_);
    // The GPU volume is the scan on GPU builds; reset discards it. start()
    // re-allocates and uploads the cleared CPU volume.
    releaseGpuResources();

    // Clear queues
    {
        std::lock_guard<std::mutex> lk(tracking_queue_mutex_);
        while (!raw_queue_.empty()) raw_queue_.pop();
    }
    {
        std::lock_guard<std::mutex> lk(integration_queue_mutex_);
        while (!integration_queue_.empty()) integration_queue_.pop();
    }

    invalidateMeshState();

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

    {
        std::unique_lock<std::shared_mutex> tsdf_lk(tsdf_mutex_);
        tsdf_->reset();
    }
    {
        std::lock_guard<std::mutex> lk(pose_mutex_);
        current_pose_ = Eigen::Matrix4f::Identity();
        last_pose_ = Eigen::Matrix4f::Identity();
    }
    first_frame_          = true;
    lost_log_counter_     = 0;
    success_log_counter_  = 0;
    hip_ui_skip_          = 0;
    // frame_count_ and metrics_ are a pair guarded by metrics_mutex_ everywhere
    // else (onRawFrame writes both under it), so the reset that zeroes them takes
    // the same lock; control_mutex_ does not, because it serialises lifecycle and
    // this reset previously raced a metricsSnapshot() reader.
    {
        std::lock_guard<std::mutex> lk(metrics_mutex_);
        frame_count_ = 0;
        metrics_     = PipelineMetrics{};
    }
    {
        std::lock_guard<std::mutex> pp_lk(preprocessor_mutex_);
        if (preprocessor_) {
            preprocessor_->reset();
        }
    }
    shared_mesh_.update(std::make_shared<meshing::MeshData>());
    state_.store(PipelineState::Idle);
    KFLOG_INFO("Pipeline", "System RESET: Volume cleared, pose re-centered, queues drained.");
}

void PipelineController::configurePreprocessor() {
    std::lock_guard<std::mutex> pp_lk(preprocessor_mutex_);
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

#ifdef AZU_PIPELINE_TEST_SEAM
namespace {
struct UiFrameTestState {
    std::mutex mtx;
    std::function<void(const sensor::FrameData&)> hook;
};
UiFrameTestState& ui_frame_test_state() {
    static UiFrameTestState state;
    return state;
}
std::atomic<int> g_seam_ui_deliveries{0};
} // namespace
#endif

void PipelineController::dispatchUiFrame(std::shared_ptr<sensor::FrameData> ui_frame) {
    // Snapshot the subscriber once. The queued lambda captures THAT copy and the
    // shared frame, never `this`, so a later setFrameReadyCallback() (or
    // controller teardown ordering) cannot change what this dispatch invokes.
    FrameReadyCallback on_frame = frameReadyCallbackCopy();
    if (qApp) {
        QMetaObject::invokeMethod(
            qApp,
            [on_frame = std::move(on_frame), ui_frame]() {
                if (on_frame)
                    on_frame(*ui_frame);
            },
            Qt::QueuedConnection);
        return;
    }
#ifdef AZU_PIPELINE_TEST_SEAM
    std::function<void(const sensor::FrameData&)> hook;
    {
        std::lock_guard<std::mutex> lk(ui_frame_test_state().mtx);
        hook = ui_frame_test_state().hook;
    }
    if (hook) {
        g_seam_ui_deliveries.fetch_add(1, std::memory_order_relaxed);
        hook(*ui_frame);
        return;
    }
    if (on_frame) {
        g_seam_ui_deliveries.fetch_add(1, std::memory_order_relaxed);
        on_frame(*ui_frame);
    }
    // No hook and no callback: deterministic no-op; null qApp is never touched.
#endif
}

#ifdef AZU_PIPELINE_TEST_SEAM
void PipelineController::registerUiFrameTestHookForTests(
    std::function<void(const sensor::FrameData&)> hook) {
    std::lock_guard<std::mutex> lk(ui_frame_test_state().mtx);
    ui_frame_test_state().hook = std::move(hook);
}

int PipelineController::uiFrameDeliveryCountForTests() {
    return g_seam_ui_deliveries.load(std::memory_order_relaxed);
}

void PipelineController::resetUiFrameTestStateForTests() {
    {
        std::lock_guard<std::mutex> lk(ui_frame_test_state().mtx);
        ui_frame_test_state().hook = nullptr;
    }
    g_seam_ui_deliveries.store(0, std::memory_order_relaxed);
}
#endif

// Called from sensor capture thread — must be lightweight
void PipelineController::onRawFrame(std::shared_ptr<sensor::RawFrame> raw) {
  if (!running_.load())
    return;

  // Retain-latest, bounded: at capacity the OLDEST queued frame is displaced.
  // The old policy rejected the arriving frame instead, so a burst while the
  // tracking worker was busy threw away the freshest geometry and kept the stale
  // frames — the exact inversion of what a tracking pipeline needs, and silent.
  std::shared_ptr<sensor::RawFrame> displaced;
  bool displaced_a_frame = false;
  {
    std::lock_guard<std::mutex> lk(tracking_queue_mutex_);
    if (raw_queue_.size() >= kRawQueueCapacity) {
      displaced = std::move(raw_queue_.front());
      raw_queue_.pop();
      displaced_a_frame = true;
    }
    raw_queue_.push(std::move(raw));
  }
  tracking_queue_cv_.notify_one();
  // The displaced frame's pooled deleter takes the sensor pool mutex, so it is
  // destructed here — past the queue lock — exactly like KinectSensor's
  // publishLatest() does for the frame it replaces.
  displaced.reset();

  // Update capture FPS
  auto now = steady_clock::now();
  float dt = duration<float>(now - last_capture_time_).count();
  last_capture_time_ = now;

  int fc = 0;
  float inst_fps = 0.0f;
  int dropped = 0;
  {
    // metrics_mutex_ is a leaf: taken after the queue lock is released, never
    // nested inside it.
    std::lock_guard<std::mutex> lk(metrics_mutex_);
    metrics_.capture_fps = (dt > 0.0f) ? (1.0f / dt) : 0.0f;
    metrics_.frame_count = ++frame_count_;
    if (displaced_a_frame) ++metrics_.dropped_frames;
    fc = metrics_.frame_count;
    inst_fps = metrics_.capture_fps;
    dropped = metrics_.dropped_frames;
  }
  if (fc % 150 == 0) {
    KFLOGF_DEBUG("Pipeline",
                 "Sensor throughput: %d frames received @ %.2f FPS "
                 "(%d stale frames displaced)",
                 fc, inst_fps, dropped);
  }
}

void PipelineController::enqueueForIntegration(
    std::shared_ptr<sensor::FrameData> frame) {
  std::shared_ptr<sensor::FrameData> displaced;
  bool displaced_a_frame = false;
  {
    std::lock_guard<std::mutex> lk(integration_queue_mutex_);
    if (integration_queue_.size() >= kIntegrationQueueCapacity) {
      displaced = std::move(integration_queue_.front());
      integration_queue_.pop();
      displaced_a_frame = true;
    }
    // Move, never copy: the queue must be the ONLY owner. A second shared_ptr
    // here would run the pooled deleter when the local copy died and hand the
    // same FrameData back to acquireFreeData() while the queue still pointed at
    // it — two writers, one frame.
    integration_queue_.push(std::move(frame));
  }
  integration_queue_cv_.notify_one();
  // Displaced frame returns to the pool here, outside both the queue lock and
  // the pool mutex its deleter takes.
  displaced.reset();
  if (displaced_a_frame) {
    std::lock_guard<std::mutex> lk(metrics_mutex_);
    ++metrics_.dropped_integration_frames;
  }
}

void PipelineController::trackingLoop() {
  try {
    trackingLoopBody();
  } catch (const std::exception& e) {
    onWorkerFault("Tracking", e.what());
  } catch (...) {
    onWorkerFault("Tracking", "unknown exception");
  }
}

void PipelineController::trackingLoopBody() {
  while (running_.load()) {
#ifdef AZU_PIPELINE_TEST_SEAM
    waitForWorkerGate(tracking_gate_);
#endif
    std::shared_ptr<sensor::RawFrame> raw;
    {
      std::unique_lock<std::mutex> lk(tracking_queue_mutex_);
      tracking_queue_cv_.wait(
          lk, [&] { return tracking_shutdown_ || !raw_queue_.empty(); });
      if (tracking_shutdown_)
        break;
      if (raw_queue_.empty())
        continue;
      raw = std::move(raw_queue_.front());
      raw_queue_.pop();
    }

    // Frame boundary: the only place a running thread-count request is allowed
    // to reach the tracker (never mid-track).
    applyPendingThreadCount();

    // One owner snapshot per processed frame, taken after the frame is popped
    // and before anything consumes it. hyper_mutex_ is released immediately: no
    // preprocess/track/integrate call ever runs under it.
    const FusionHyperparams hp = hyperparamsSnapshot();
#ifdef AZU_PIPELINE_TEST_SEAM
    recordTrackingBandForTests(hp, raw->frame_id);
#endif

    // Build processed frame here (using pool)
    auto frame = acquireFreeData();
    if (!frame) {
        KFLOG_WARN("Pipeline", "FrameData pool exhausted — dropping raw frame (increase pool or slow capture)");
        sensor_->releaseFrame(std::move(raw)); // Don't forget to recycle raw
        continue;
    }
    
    frame->frame_id = raw->frame_id;
    frame->rgb_valid = raw->rgb_valid;

    if (preprocessor_) {
        std::lock_guard<std::mutex> pp_lk(preprocessor_mutex_);
        preprocessor_->process(*raw, hp.min_depth, hp.max_depth);
    }
    // Upscaled RGB for TSDF texturing stays DISABLED (future scope).
    // srUpscaledAvailable() is now the gate: the CPU path publishes the buffer
    // only after producing a fresh, correctly sized frame, and any future
    // consumer MUST check that contract before touching getSrRgbUpscaled() —
    // the getter alone hands out preallocated zero bytes, which is what made the
    // uncommented line below produce black textures. GPU upscaled output is
    // deferred, not valid: no CUDA/HIP conditioner implements an EASU upscaled
    // pass, so srUpscaledAvailable() reports false there by contract (see
    // docs/CUDA_HIP_DEFERRED_CHANGES.md).
    // const auto& upscaled_rgb = preprocessor_->getSrRgbUpscaled();
    // sensor::buildFrameData(raw->depth.data(), upscaled_rgb.data(), *frame, hp.min_depth, hp.max_depth);
    sensor::buildFrameData(raw->depth.data(), raw->rgb.data(), *frame, hp.min_depth, hp.max_depth);
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
        const uint64_t origin_id = frame->frame_id;
        // Sole ownership moves into the queue; the previous copy handed this
        // FrameData back to the pool when the local died at the end of the
        // iteration while the queue still referenced it.
        enqueueForIntegration(std::move(frame));

        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            metrics_.tracking_ok = true;
            metrics_.icp_error   = 0.0f;
        }
        KFLOGF_INFO("Pipeline", "First frame accepted (ID: %lu). Initializing world origin.", origin_id);
        continue;
    }

    // Race-condition guard: wait for the integration thread to finish the first raycast.
    // Without this, frame 2 would ICP against an empty model and immediately declare
    // tracking lost, locking the pipeline out of ever recovering.
    if (!model_ready_.load()) {
        // The frame is discarded, not deferred: holding it would only predict
        // against a model that does not exist yet. That startup loss is real, so
        // it is counted rather than silent.
        frame.reset();
        {
            std::lock_guard<std::mutex> lk(metrics_mutex_);
            ++metrics_.dropped_pre_model_frames;
        }
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
      const Eigen::Matrix4f motion_source = last_pose_;
      Eigen::Matrix4f delta = motion_source.inverse() * current_pose_;
      predicted_pose = current_pose_ * delta;
      last_pose_ = current_pose_;
#ifdef AZU_PIPELINE_TEST_SEAM
      recordMotionModelForTests(prev_pose, predicted_pose, motion_source);
#endif
    }

    tracking::ICPResult icp_result;
    bool is_lost = (state_.load() == PipelineState::TrackingLost);

    // Both solvers run with tracker_mutex_ ALREADY HELD by the caller: the
    // solver reads ICPTracker::params_ iteration by iteration, so the lock must
    // span the whole track() (and, in relocalization, the recovery-params swap
    // around it). They do not lock, so a caller can hold the lock across
    // several calls without a self-deadlock.
    auto solve_cpu = [&](const sensor::FramePyramid& pyramid,
                         const Eigen::Matrix4f& estimate) -> tracking::ICPResult {
        return tracker_->track(pyramid, *model_ref, estimate, prev_pose);
    };
    auto solve_gpu = [&](const Eigen::Matrix4f& estimate) -> tracking::ICPResult {
        tracking::ICPResult res;
#ifdef CUDA_ENABLED
        res = tracker_->trackGPU(
            preprocessor_->getGPUDepthMeters(),
            preprocessor_->getGPURgb(),
            frame->width, frame->height,
            *model_ref, estimate, prev_pose
        );
#elif defined(HIP_ENABLED)
        res = tracker_->trackGPU(
            preprocessor_->getGPUDepthMeters(),
            preprocessor_->getGPURgb(),
            frame->width, frame->height,
            *model_ref, estimate, prev_pose
        );
#else
        (void)estimate;
#endif
        return res;
    };

    // gpu_mutex_ first, tracker_mutex_ second — the order every other GPU path
    // uses; the CPU branches take the tracker lock alone.
    std::unique_lock<std::mutex> gpu_lk;
    if (use_gpu_.load())
        gpu_lk = std::unique_lock<std::mutex>(gpu_mutex_);
    std::unique_lock<std::mutex> tracker_lk(tracker_mutex_);

    if (is_lost) {
      // RELOCALIZATION MODE: Multi-hypothesis search. Params come from this
      // frame's snapshot, never from an unsynchronized hyperparams_ read.
      tracking::ICPParams recovery_params = hp.icp;
      recovery_params.dist_threshold *= 2.5f; 
      recovery_params.angle_threshold = 45.0f;
      
      // Increase iterations for recovery
      for (int l = 0; l < sensor::FramePyramid::LEVELS; ++l) {
          recovery_params.max_iterations[l] *= 2;
      }

      const tracking::ICPParams original_params = tracker_->params();
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

      sensor::FramePyramid pyramid;
      if (!use_gpu_.load()) {
          sensor::buildFramePyramid(*frame, pyramid);
      }

      for (const auto& h_pose : hypotheses) {
          tracking::ICPResult res = use_gpu_.load() ? solve_gpu(h_pose)
                                                   : solve_cpu(pyramid, h_pose);

          // Always track the best-inlier hypothesis so diagnostics are
          // meaningful even when every hypothesis fails (tracking_ok==false).
          if (res.inliers > best_result.inliers) {
              best_result = res;
          }
          if (best_result.tracking_ok && best_result.inliers > 5000) break; // Good enough
      }
      
      icp_result = best_result;
      // Restored under the same lock the swap happened under, so the tracker can
      // never be left holding recovery params, and a concurrent setHyperparams()
      // cannot land between the swap and the restore.
      tracker_->setParams(original_params);
    } else {
      if (use_gpu_.load()) {
        icp_result = solve_gpu(predicted_pose);
        if (!icp_result.tracking_ok) {
          icp_result = solve_gpu(prev_pose);
        }
      } else {
        sensor::FramePyramid pyramid;
        sensor::buildFramePyramid(*frame, pyramid);
        icp_result = solve_cpu(pyramid, predicted_pose);
        if (!icp_result.tracking_ok) {
          icp_result = solve_cpu(pyramid, prev_pose);
        }
      }
    }
    tracker_lk.unlock();
    // gpu_lk is only ever owned on the GPU path; unlocking an unowned
    // unique_lock throws system_error(EPERM).
    if (gpu_lk.owns_lock())
        gpu_lk.unlock();

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

            // Enqueue for integration (retain-latest, sole owner moves in).
            enqueueForIntegration(std::move(frame));
        } else {
            // RELOCALIZATION: If tracking lost, don't update state or pose, but don't stop.
            // We just don't integrate the frame. This keeps the model "clean".
             if (!is_lost) {
                 KFLOG_WARN("Pipeline", "Tracking lost! Suspension of TSDF integration. Entering relocalization mode...");
                 if (preprocessor_) {
                     std::lock_guard<std::mutex> pp_lk(preprocessor_mutex_);
                     preprocessor_->resetTemporalState();
                 }

                 // Clear the integration queue to prevent "garbage" poses from being integrated
                 std::lock_guard<std::mutex> lk(integration_queue_mutex_);
                 std::queue<std::shared_ptr<sensor::FrameData>> empty;
                 std::swap(integration_queue_, empty);
            }
            state_.store(PipelineState::TrackingLost);
            frame.reset();
        }
    }
}

void PipelineController::integrationLoop() {
  try {
    integrationLoopBody();
  } catch (const std::exception& e) {
    onWorkerFault("Integration", e.what());
  } catch (...) {
    onWorkerFault("Integration", "unknown exception");
  }
}

void PipelineController::integrationLoopBody() {
  while (running_.load()) {
#ifdef AZU_PIPELINE_TEST_SEAM
    waitForWorkerGate(integration_gate_);
#endif
    std::shared_ptr<sensor::FrameData> frame;
    {
      std::unique_lock<std::mutex> lk(integration_queue_mutex_);
      integration_queue_cv_.wait(
          lk, [&] { return integration_shutdown_ || !integration_queue_.empty(); });
      if (integration_shutdown_)
        break;
      if (integration_queue_.empty())
        continue;
      frame = std::move(integration_queue_.front());
      integration_queue_.pop();
    }

    // Owner snapshot per integrated frame, after the pop and before the TSDF
    // sees the frame: a depth-band change can no longer be applied to whatever
    // happens to be queued next, mid-integration.
    const FusionHyperparams hp = hyperparamsSnapshot();
#ifdef AZU_PIPELINE_TEST_SEAM
    recordIntegrationBandForTests(hp.min_depth, hp.max_depth);
#endif
    const float d_min = hp.min_depth, d_max = hp.max_depth;
    // A depth-only frame still integrates geometry, but its colour is not fused.
    const uint8_t* fuse_rgb = frame->rgb_valid ? frame->rgb.data() : nullptr;

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
              fuse_rgb,
              frame->pose,
              static_cast<float>(sensor::FX), static_cast<float>(sensor::FY),
              static_cast<float>(sensor::CX), static_cast<float>(sensor::CY),
              frame->width, frame->height, d_min, d_max);
      } else {
          gpu_lk.unlock(); // CPU path doesn't need GPU serialization
          tsdf_->integrate(
              frame->depth_meters.data(), fuse_rgb, frame->pose,
              static_cast<float>(sensor::FX), static_cast<float>(sensor::FY),
              static_cast<float>(sensor::CX), static_cast<float>(sensor::CY),
              frame->width, frame->height, d_min, d_max);
      }
#elif defined(HIP_ENABLED)
      if (use_gpu_.load()) {
          tsdf_->integrate(
              frame->depth_meters.data(),
              fuse_rgb,
              frame->pose,
              static_cast<float>(sensor::FX), static_cast<float>(sensor::FY),
              static_cast<float>(sensor::CX), static_cast<float>(sensor::CY),
              frame->width, frame->height, d_min, d_max);
      } else {
          gpu_lk.unlock(); // CPU path doesn't need GPU serialization
          tsdf_->integrate(
              frame->depth_meters.data(), fuse_rgb, frame->pose,
              static_cast<float>(sensor::FX), static_cast<float>(sensor::FY),
              static_cast<float>(sensor::CX), static_cast<float>(sensor::CY),
              frame->width, frame->height, d_min, d_max);
      }
#else
      gpu_lk.unlock(); // CPU-only build: no GPU serialization needed
      tsdf_->integrate(
          frame->depth_meters.data(), fuse_rgb, frame->pose,
          static_cast<float>(sensor::FX), static_cast<float>(sensor::FY),
          static_cast<float>(sensor::CX), static_cast<float>(sensor::CY),
          frame->width, frame->height, d_min, d_max);
#endif
    }

    // 2. Generate model frame for tracking (Ping-Pong back buffer)
    int integrated_count = tsdf_->integratedFrames();
    { // Raycast on every integration; the block scopes the back-buffer bindings.
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

          if (hasFrameReadyCallback()) {
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

            dispatchUiFrame(std::move(ui_frame));
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
          if (hasFrameReadyCallback()) {
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

            dispatchUiFrame(std::move(ui_frame));
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
          if (hasFrameReadyCallback() && (++hip_ui_skip_ % HIP_UI_PREVIEW_INTERVAL == 0)) {
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

            dispatchUiFrame(std::move(ui_frame));
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

    // Drop the worker's reference; the frame's custom deleter recycles it into
    // the pool once every other owner has dropped theirs.
    frame.reset();

    // Mesh cadence: time-based, and a fire-and-forget version bump. The old rule
    // (one request every 5 integrated frames) coupled mesh rate to capture rate,
    // so a faster pipeline meshed more often against the same volume while a slow
    // one starved the view. Timing belongs to the clock, not to the frame count,
    // and the requester never waits for the result.
    requestMeshIfCadenceDue();
  }
}

void PipelineController::meshingLoop() {
  try {
    meshingLoopBody();
  } catch (const std::exception& e) {
    onWorkerFault("Meshing", e.what());
  } catch (...) {
    onWorkerFault("Meshing", "unknown exception");
  }
}

void PipelineController::meshingLoopBody() {
  for (;;) {
    // Claim a version under the request lock. A request can no longer be lost
    // the way a boolean flag was: clearing the flag between reading it and
    // storing false dropped every request that landed in that window, and
    // nothing said which volume the published mesh described.
    uint64_t serving = 0;
    uint64_t generation_at_claim = 0;
    {
      std::unique_lock<std::mutex> lk(mesh_requests_.mtx);
      mesh_requests_.cv.wait(lk, [&] {
        return mesh_requests_.shutdown ||
               mesh_requests_.requested > mesh_requests_.claimed;
      });
      if (mesh_requests_.shutdown)
        return;
      // Serve the NEWEST requested version. Requests that arrived while the
      // previous extraction ran simply left requested > claimed, so they are
      // served by this pass; an older "give me a mesh now" is satisfied by a
      // strictly fresher extraction, which is subsumption, not a merge of two
      // different results. Requests arriving from here on stay pending and are
      // served by the next pass.
      serving = mesh_requests_.requested;
      generation_at_claim = mesh_requests_.generation;
      mesh_requests_.claimed = serving;
    }
#ifdef AZU_PIPELINE_TEST_SEAM
    engageMeshExtractionHookForTests(serving);
#endif

    mesh_extract_progress_.store(0.0f);
    KFLOGF_INFO("Pipeline",
                "Marching Cubes: Extracting mesh from TSDF volume "
                "(request v%llu, generation %llu)...",
                static_cast<unsigned long long>(serving),
                static_cast<unsigned long long>(generation_at_claim));
    std::shared_ptr<meshing::MeshData> mesh;
    {
#ifdef AZU_PIPELINE_TEST_SEAM
      recordMeshExtractionForTests();
#endif
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

    // A reset() bumps the generation. If that happened while this extraction ran,
    // the mesh describes a volume that has been cleared, so it is dropped rather
    // than published: publishing it would hand the UI a mesh of a scan the user
    // already discarded, and report it as the result of a newer request.
    bool stale_generation = false;
    {
      std::lock_guard<std::mutex> lk(mesh_requests_.mtx);
      stale_generation = (mesh_requests_.generation != generation_at_claim);
      if (!stale_generation) {
        mesh_requests_.served = serving;
        mesh_requests_.served_generation = generation_at_claim;
        mesh_requests_.cv.notify_all();
      }
    }
    if (stale_generation) {
#ifdef AZU_PIPELINE_TEST_SEAM
      recordMeshStaleDropForTests();
#endif
      KFLOGF_WARN("Pipeline",
                  "Mesh Extraction DROPPED: request v%llu was extracted against "
                  "generation %llu, which has since been reset.",
                  static_cast<unsigned long long>(serving),
                  static_cast<unsigned long long>(generation_at_claim));
      continue;
    }

    {
      std::lock_guard<std::mutex> lk(metrics_mutex_);
      metrics_.mesh_triangles = mesh ? mesh->triangleCount() : 0;
      metrics_.mesh_extract_pct = 100.0f;
    }

    // Copy under callback_mutex_, invoke outside it: setMeshReadyCallback() can
    // run on the UI thread while this thread is mid-extraction, and the
    // subscriber is arbitrary code that may re-enter the controller.
    MeshReadyCallback on_mesh = meshReadyCallbackCopy();
    if (mesh && !mesh->empty()) {
      KFLOGF_INFO("Pipeline",
                  "Mesh Extraction SUCCESS: %zu triangles, %zu vertices ready "
                  "for rendering.",
                  mesh->triangleCount(), mesh->positions.size());
      shared_mesh_.update(std::move(mesh));
      if (on_mesh)
        on_mesh();
    } else {
      KFLOG_WARN("Pipeline", "Mesh Extraction EMPTY: Volume might be too "
                             "sparse or clipping values too aggressive.");
      if (on_mesh)
        on_mesh();
    }
#ifdef AZU_PIPELINE_TEST_SEAM
    recordMeshPublishForTests();
#endif
  }
}

uint64_t PipelineController::requestMesh() {
  uint64_t version = 0;
  {
    std::lock_guard<std::mutex> lk(mesh_requests_.mtx);
    version = ++mesh_requests_.requested;
    mesh_requests_.cv.notify_one();
  }
  return version;
}

void PipelineController::invalidateMeshState() {
  std::lock_guard<std::mutex> lk(mesh_requests_.mtx);
  ++mesh_requests_.generation;
  // Drain every pending request: after reset() the volume it asked for does not
  // exist any more. served / served_generation deliberately stay untouched —
  // they describe the mesh that IS published, and inventing a served version
  // would let a waiter wake on a mesh that was never produced. A waiter on a
  // drained version therefore times out (bounded), exactly as the pre-Todo-25
  // export loop did when its flag was cleared.
  mesh_requests_.requested = mesh_requests_.claimed =
      std::max(mesh_requests_.requested, mesh_requests_.claimed);
  mesh_requests_.cv.notify_all();
}

void PipelineController::requestMeshIfCadenceDue() {
  const auto now = std::chrono::duration_cast<std::chrono::microseconds>(
      steady_clock::now().time_since_epoch());
  const auto interval = std::chrono::microseconds(mesh_cadence_us_.load());
  if (now - last_mesh_request_ < interval)
    return;
  last_mesh_request_ = now;
  requestMesh();
}

bool PipelineController::awaitMeshVersion(uint64_t version,
                                          std::chrono::milliseconds timeout) {
  std::unique_lock<std::mutex> lk(mesh_requests_.mtx);
  return mesh_requests_.cv.wait_for(lk, timeout, [&] {
    return mesh_requests_.served >= version || mesh_requests_.shutdown;
  });
}

bool PipelineController::exportMesh(const std::string &path,
                                    const MeshWriterFn &writer_fn) {
  uint64_t ver;
  auto mesh = shared_mesh_.snapshot(ver);
  if (!mesh || mesh->empty()) {
    KFLOG_INFO("Pipeline", "No mesh yet, requesting an extraction...");
    // Request a version and wait for THAT version to be published, instead of
    // polling a shared flag that another thread (or the cadence) could clear:
    // the wait is now answered only by the mesh this request caused.
    const uint64_t requested_version = requestMesh();
    awaitMeshVersion(requested_version, std::chrono::milliseconds(5000));
    mesh = shared_mesh_.snapshot(ver);
  }
  if (!mesh || mesh->empty()) {
    KFLOG_WARN("Pipeline", "No mesh to export — scan more frames first.");
    return false;
  }
  {
    std::lock_guard<std::mutex> lk(metrics_mutex_);
    metrics_.export_pct = 50.0f;
  }
  bool ok = false;
  try {
    ok = writer_fn(*mesh, path);
  } catch (const std::exception &e) {
    KFLOG_ERROR("Pipeline", std::string("Export writer threw: ") + e.what());
    ok = false;
  } catch (...) {
    KFLOG_ERROR("Pipeline", "Export writer threw an unknown error.");
    ok = false;
  }
  {
    std::lock_guard<std::mutex> lk(metrics_mutex_);
    metrics_.export_pct = ok ? 100.0f : 0.0f;
  }
  return ok;
}

bool PipelineController::exportPLY(const std::string &path) {
  return exportMesh(path,
                    [](const meshing::MeshData &mesh, const std::string &p) {
                      return export_io::PLYExporter::writeBinary(mesh, p);
                    });
}

bool PipelineController::exportGLB(const std::string &path) {
  return exportMesh(path,
                    [](const meshing::MeshData &mesh, const std::string &p) {
                      return export_io::GLBExporter::write(mesh, p);
                    });
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

#ifdef AZU_PIPELINE_TEST_SEAM
void PipelineController::recordTrackingBandForTests(const FusionHyperparams& h,
                                                    uint64_t frame_id) {
  // Leaf lock only: callers hold no controller lock here (the hyperparams_
  // snapshot was already copied out of hyper_mutex_).
  std::lock_guard<std::mutex> lk(seam_obs_.mtx);
  auto& band        = seam_obs_.tracking_band;
  band.generation  += 1;
  band.min_depth    = h.min_depth;
  band.max_depth    = h.max_depth;
  band.valid        = true;
  seam_obs_.last_popped_frame_id = frame_id;
}

void PipelineController::recordIntegrationBandForTests(float min_depth, float max_depth) {
  std::lock_guard<std::mutex> lk(seam_obs_.mtx);
  auto& band        = seam_obs_.integration_band;
  band.generation  += 1;
  band.min_depth    = min_depth;
  band.max_depth    = max_depth;
  band.valid        = true;
}

PipelineController::DepthBandObservation
PipelineController::lastTrackingDepthBandForTests() const {
  std::lock_guard<std::mutex> lk(seam_obs_.mtx);
  return seam_obs_.tracking_band;
}

PipelineController::DepthBandObservation
PipelineController::lastIntegrationDepthBandForTests() const {
  std::lock_guard<std::mutex> lk(seam_obs_.mtx);
  return seam_obs_.integration_band;
}

int PipelineController::appliedThreadCountForTests() const {
  std::lock_guard<std::mutex> lk(seam_obs_.mtx);
  return seam_obs_.applied_threads;
}

int PipelineController::appliedSrScaleForTests() const {
  std::lock_guard<std::mutex> lk(seam_obs_.mtx);
  return seam_obs_.applied_sr_scale;
}

bool PipelineController::callbackMutexIsFreeForTests() const {
  std::unique_lock<std::mutex> lk(callback_mutex_, std::try_to_lock);
  return lk.owns_lock();
}

bool PipelineController::trackerMutexIsFreeForTests() const {
  std::unique_lock<std::mutex> lk(tracker_mutex_, std::try_to_lock);
  return lk.owns_lock();
}

bool PipelineController::tsdfMutexIsFreeForTests() const {
  // Shared probe on purpose: only the exclusive (writer) sections matter here.
  // A concurrent raycast reader is legitimate during a live scan, so an
  // exclusive probe would make the assertion timing-dependent.
  if (tsdf_mutex_.try_lock_shared()) {
    tsdf_mutex_.unlock_shared();
    return true;
  }
  return false;
}

bool PipelineController::metricsMutexIsFreeForTests() {
  std::unique_lock<std::mutex> lk(metrics_mutex_, std::try_to_lock);
  return lk.owns_lock();
}

void PipelineController::recordMotionModelForTests(
    const Eigen::Matrix4f& prev_pose, const Eigen::Matrix4f& predicted,
    const Eigen::Matrix4f& last_pose_before) {
  // Called while pose_mutex_ is held, so seam_obs_.mtx is the innermost lock and
  // nothing here reaches for pose_mutex_ again.
  std::lock_guard<std::mutex> lk(seam_obs_.mtx);
  auto& m           = seam_obs_.motion;
  m.generation     += 1;
  m.prev_pose       = prev_pose;
  m.predicted_pose  = predicted;
  m.last_pose_before = last_pose_before;
  m.valid           = true;
}

PipelineController::MotionModelObservation
PipelineController::lastMotionModelForTests() {
  std::lock_guard<std::mutex> lk(seam_obs_.mtx);
  return seam_obs_.motion;
}

uint64_t PipelineController::lastPoppedFrameIdForTests() {
  std::lock_guard<std::mutex> lk(seam_obs_.mtx);
  return seam_obs_.last_popped_frame_id;
}

Eigen::Matrix4f PipelineController::lastPoseForTests() {
  std::lock_guard<std::mutex> lk(pose_mutex_);
  return last_pose_;
}

size_t PipelineController::rawQueueDepthForTests() {
  std::lock_guard<std::mutex> lk(tracking_queue_mutex_);
  return raw_queue_.size();
}

size_t PipelineController::integrationQueueDepthForTests() {
  std::lock_guard<std::mutex> lk(integration_queue_mutex_);
  return integration_queue_.size();
}

void PipelineController::waitForWorkerGate(WorkerGate& gate) {
  std::unique_lock<std::mutex> lk(gate.mtx);
  if (!gate.paused)
    return;
  gate.parked = true;
  gate.cv.notify_all();
  // Shutdown releases the gate from stop(), so a parked worker can never block a
  // join: the predicate is "resumed or released", never "some future test call".
  gate.cv.wait(lk, [&gate] { return !gate.paused; });
  gate.parked = false;
}

void PipelineController::releaseWorkerGate(WorkerGate& gate) {
  std::lock_guard<std::mutex> lk(gate.mtx);
  gate.paused = false;
  gate.parked = false;
  gate.cv.notify_all();
}

void PipelineController::pauseTrackingWorkerForTests() {
  std::lock_guard<std::mutex> lk(tracking_gate_.mtx);
  tracking_gate_.paused = true;
}

void PipelineController::pauseIntegrationWorkerForTests() {
  std::lock_guard<std::mutex> lk(integration_gate_.mtx);
  integration_gate_.paused = true;
}

void PipelineController::resumePipelineWorkersForTests() {
  releaseWorkerGate(tracking_gate_);
  releaseWorkerGate(integration_gate_);
}

bool PipelineController::trackingWorkerParkedForTests() {
  std::lock_guard<std::mutex> lk(tracking_gate_.mtx);
  return tracking_gate_.parked;
}

bool PipelineController::integrationWorkerParkedForTests() {
  std::lock_guard<std::mutex> lk(integration_gate_.mtx);
  return integration_gate_.parked;
}

void PipelineController::engageMeshExtractionHookForTests(uint64_t claimed_version) {
  std::unique_lock<std::mutex> lk(mesh_hook_.mtx);
  if (mesh_hook_.holds <= 0)
    return;
  --mesh_hook_.holds;
  {
    std::lock_guard<std::mutex> obs_lk(seam_obs_.mtx);
    seam_obs_.mesh_hook_versions.push_back(claimed_version);
  }
  mesh_hook_.cv.notify_all();
  // Parked until release (or stop(), which releases): the worker is inside
  // meshingLoop() having already claimed its version, and holds no volume lock.
  mesh_hook_.cv.wait(lk, [this] { return mesh_hook_.released; });
  mesh_hook_.released = false;
}

void PipelineController::armMeshExtractionHoldForTests(int count) {
  std::lock_guard<std::mutex> lk(mesh_hook_.mtx);
  mesh_hook_.holds = count;
  mesh_hook_.released = false;
}

void PipelineController::releaseMeshExtractionHoldForTests() {
  std::lock_guard<std::mutex> lk(mesh_hook_.mtx);
  mesh_hook_.released = true;
  mesh_hook_.cv.notify_all();
}

std::vector<uint64_t> PipelineController::meshHookVersionsForTests() {
  std::lock_guard<std::mutex> lk(seam_obs_.mtx);
  return seam_obs_.mesh_hook_versions;
}

void PipelineController::recordMeshExtractionForTests() {
  std::lock_guard<std::mutex> lk(seam_obs_.mtx);
  ++seam_obs_.mesh_extractions;
}

void PipelineController::recordMeshPublishForTests() {
  std::lock_guard<std::mutex> lk(seam_obs_.mtx);
  ++seam_obs_.mesh_publishes;
}

void PipelineController::recordMeshStaleDropForTests() {
  std::lock_guard<std::mutex> lk(seam_obs_.mtx);
  ++seam_obs_.mesh_stale_drops;
}

PipelineController::MeshStateObservation
PipelineController::meshStateForTests() {
  MeshStateObservation s;
  {
    std::lock_guard<std::mutex> lk(mesh_requests_.mtx);
    s.requested         = mesh_requests_.requested;
    s.claimed           = mesh_requests_.claimed;
    s.served            = mesh_requests_.served;
    s.generation        = mesh_requests_.generation;
    s.served_generation = mesh_requests_.served_generation;
  }
  std::lock_guard<std::mutex> obs_lk(seam_obs_.mtx);
  s.extractions = seam_obs_.mesh_extractions;
  s.publishes   = seam_obs_.mesh_publishes;
  s.stale_drops = seam_obs_.mesh_stale_drops;
  return s;
}

uint64_t PipelineController::requestMeshForTests() { return requestMesh(); }

void PipelineController::invalidateMeshStateForTests() { invalidateMeshState(); }

void PipelineController::setMeshCadenceIntervalForTests(
    std::chrono::microseconds interval) {
  mesh_cadence_us_.store(interval.count());
}
#endif

} // namespace app
} // namespace kfusion
