#pragma once

#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <condition_variable>
#include <functional>
#include <queue>
#include <string>
#include <chrono>

#include "sensor/KinectSensor.h"
#include "sensor/FrameData.h"
#include "sensor/Preprocessor.h"
#include "tracking/ICPTracker.h"
#include "tsdf/TSDFVolume.h"
#include "meshing/MarchingCubes.h"
#include "meshing/MeshData.h"
#include "app/FusionHyperparams.h"
#include <Eigen/Core>

namespace kfusion {
namespace app {

enum class PipelineState {
    Idle,
    Running,
    TrackingLost,
    Error,
    Stopped
};

struct PipelineMetrics {
    float    capture_fps        = 0.0f;
    float    tracking_fps       = 0.0f;
    int      frame_count        = 0;
    int      integrated_frames  = 0;
    float    icp_error          = 0.0f;
    bool     tracking_ok        = true;
    // Fraction of live depth points that found a model correspondence this
    // frame (100*valid_model/valid_live). Range [0,100]; -1 means "ICP has not
    // run yet" — 0 is a real reading once ICP counted >=1 live point. Two CUDA/
    // CPU divergences make CUDA read lower at identical geometry: valid_live
    // counts pre-ref-z-check on GPU, post both checks on CPU; and GPU gates
    // model normals at ||n||^2>0.9 (ICPTracker_cuda.cu) vs CPU reject-only.
    // The UI bands are calibrated for CUDA.
    float    icp_overlap_pct    = -1.0f;
    // Raw correspondence count driving overlap_pct; UI shows "warming up"
    // (neutral grey) below ~5000 because a young volume legitimately overlaps
    // little. -1 = not measured yet.
    int      icp_valid_model    = -1;
    float    volume_usage_pct   = 0.0f;
    size_t   mesh_triangles     = 0;
    float    mesh_extract_pct   = 0.0f;
    float    export_pct         = 0.0f;
    PipelineState state         = PipelineState::Idle;
};

using MetricsCallback    = std::function<void(const PipelineMetrics&)>;
using FrameReadyCallback = std::function<void(const sensor::FrameData&)>;
using MeshReadyCallback  = std::function<void()>; // mesh updated in shared mesh

class PipelineController {
public:
    explicit PipelineController(sensor::PreprocessBackend preferred_backend = sensor::PreprocessBackend::Auto);
    ~PipelineController();

    bool start();
    void stop();
    void reset();

    /** True while capture + pipeline worker threads are active. */
    bool isRunning() const { return running_.load(); }

    bool exportPLY(const std::string& path);
    bool exportGLB(const std::string& path);

    /** Optional subscriber; not invoked automatically — poll metricsSnapshot() from a timer instead. */
    void setMetricsCallback(MetricsCallback cb)    { metrics_cb_ = std::move(cb); }
    void setFrameReadyCallback(FrameReadyCallback cb) { frame_ready_cb_ = std::move(cb); }
    void setMeshReadyCallback(MeshReadyCallback cb)   { mesh_ready_cb_ = std::move(cb); }

    meshing::SharedMesh& sharedMesh() { return shared_mesh_; }
    /** Thread-safe copy for UI / diagnostics (locks internal metrics mutex). */
    PipelineMetrics metricsSnapshot() const;
    PipelineState state() const { return state_.load(); }
    /** Copy of live camera pose (locks pose_mutex_); for cage-color UI at timer rate. */
    Eigen::Matrix4f currentPose() const;

    FusionHyperparams hyperparamsSnapshot() const;
    void              setHyperparams(const FusionHyperparams& h);
    sensor::PreprocessBackend activeBackend() const { return active_backend_.load(); }
    sensor::PreprocessBackend preferredBackend() const { return preferred_backend_; }

private:
    FusionHyperparams          hyperparams_{FusionHyperparams::defaults()};
    mutable std::mutex         hyper_mutex_;
    sensor::PreprocessBackend  preferred_backend_ = sensor::PreprocessBackend::Auto;
    std::atomic<sensor::PreprocessBackend> active_backend_{sensor::PreprocessBackend::CPU};
    std::unique_ptr<sensor::Preprocessor> preprocessor_;

    // Components
    std::unique_ptr<sensor::KinectSensor>    sensor_;
    std::unique_ptr<tracking::ICPTracker>    tracker_;
    std::unique_ptr<tsdf::TSDFVolume>        tsdf_;
    std::shared_mutex                        tsdf_mutex_;
    // Serializes ALL GPU kernel dispatches across tracking/integration/meshing threads.
    // On the AMD 5650u iGPU the ROCm ring buffer cannot handle concurrent kernel submissions
    // from multiple host threads on the default stream — this causes TDR (GPU Hang).
    mutable std::mutex                       gpu_mutex_;
    std::unique_ptr<meshing::MarchingCubes>  cubes_;
    meshing::SharedMesh                      shared_mesh_;

    // State
    std::atomic<PipelineState> state_{PipelineState::Idle};
    Eigen::Matrix4f            current_pose_;
    Eigen::Matrix4f            last_pose_{Eigen::Matrix4f::Identity()};
    mutable std::mutex         pose_mutex_;

    // Metrics
    PipelineMetrics            metrics_;
    mutable std::mutex         metrics_mutex_;
    MetricsCallback            metrics_cb_;
    FrameReadyCallback         frame_ready_cb_;
    MeshReadyCallback          mesh_ready_cb_;

    // Threads
    std::thread                tracking_thread_;
    std::thread                integration_thread_;
    std::thread                meshing_thread_;
    std::atomic<bool>          running_{false};
    std::atomic<int>           num_threads_{0}; // 0 = Auto

    // Frame recycling pool (FrameData) - Self-recycling via custom deleter
    static constexpr size_t    DATA_POOL_SIZE = 6;
    struct DataPool {
        std::vector<std::shared_ptr<sensor::FrameData>> data_pool;
        std::queue<sensor::FrameData*>                  free_data_queue;
        std::mutex                                      mutex;
    };
    std::shared_ptr<DataPool> data_pool_state_;

    // Raw frame queue (sensor callback → tracking thread)
    std::queue<std::shared_ptr<sensor::RawFrame>>  raw_queue_;
    std::mutex                                     tracking_queue_mutex_;
    std::condition_variable                        tracking_queue_cv_;

    std::queue<std::shared_ptr<sensor::FrameData>> integration_queue_;
    std::mutex                                     integration_queue_mutex_;
    std::condition_variable                        integration_queue_cv_;

    // Timing
    std::chrono::steady_clock::time_point last_capture_time_;
    std::chrono::steady_clock::time_point last_tracking_time_;
    int                                   frame_count_    = 0;
    bool                                  first_frame_    = true;

    // Per-instance log throttle counters — replaces static locals in worker threads
    // to avoid UB data races when threads are stopped and restarted.
    int                                   ui_skip_counter_      = 0;
    int                                   lost_log_counter_     = 0;
    int                                   success_log_counter_  = 0;
    // HIP UI preview throttle — every N integrated frames to avoid starving
    // the GNOME compositor on the shared AMD iGPU (5650u).
    int                                   hip_ui_skip_          = 0;

    // Tracking model frame (Triple-buffered to prevent data race)
    struct TripleBufferModel {
        std::shared_ptr<tracking::ModelFrame> buffers[3];
        std::atomic<int>     front_idx{0}; // Read by tracking
        std::atomic<int>     back_idx{1};  // Written by integration
        std::atomic<int>     ready_idx{2}; // Most recently completed raycast
        std::mutex           mtx;

        void swap() {
            std::lock_guard<std::mutex> lk(mtx);
            // The buffer we just finished writing (back_idx) becomes the new ready_idx.
            // The old ready_idx becomes the new back_idx, provided it's not being read.
            // If it IS being read, we must use the third buffer.
            int old_ready = ready_idx.load();
            ready_idx.store(back_idx.load());
            
            // Find a buffer that is neither the new ready nor the current front
            for (int i = 0; i < 3; ++i) {
                if (i != ready_idx.load() && i != front_idx.load()) {
                    back_idx.store(i);
                    break;
                }
            }
        }

        int acquireFront() {
            std::lock_guard<std::mutex> lk(mtx);
            front_idx.store(ready_idx.load());
            return front_idx.load();
        }
    } model_buffers_;

    // Mesh extraction trigger
    std::atomic<bool>                     mesh_extraction_requested_{false};
    std::atomic<bool>                     is_meshing_{false};
    std::atomic<float>                    mesh_extract_progress_{0.0f};
    std::atomic<float>                    export_progress_{0.0f};
    std::atomic<bool>                     use_gpu_{false};
    // Set to true by integrationLoop after the first model raycast completes.
    // Prevents trackingLoop from running ICP against an empty model buffer.
    std::atomic<bool>                     model_ready_{false};
    sensor::cudaStream_t                  cuda_stream_ = nullptr;
    mutable std::mutex                    control_mutex_;

    bool startInternal(bool engage_sensor);
    void onRawFrame(std::shared_ptr<sensor::RawFrame> raw);
    void configurePreprocessor();
    /**
     * Centralized UI-frame callback delivery.
     * Production (qApp exists): queued delivery on the GUI thread via
     * QMetaObject::invokeMethod(qApp, ...) — identical to the previous inline code.
     * Under AZU_PIPELINE_TEST_SEAM with qApp == nullptr: synchronous delivery
     * through the registered test hook (or frame_ready_cb_), recording the
     * delivery count. With neither registered: deterministic no-op — a null
     * qApp is never dereferenced.
     */
    void dispatchUiFrame(std::shared_ptr<sensor::FrameData> ui_frame);
    void trackingLoop();
    void integrationLoop();
    void meshingLoop();

    // Pool helpers
    std::shared_ptr<sensor::FrameData> acquireFreeData();
    void releaseData(std::shared_ptr<sensor::FrameData> data);
    
public:
    void setNumThreads(int n) {
        num_threads_.store(n);
        if (tracker_) tracker_->setNumThreads(n);
    }

#ifdef AZU_PIPELINE_TEST_SEAM
    // ---- Headless test seam (compile-time; defined only for test targets) ----
    // Starts the real controller and all worker threads, bypassing ONLY
    // sensor_->init()/sensor_->start(); no Kinect device is ever opened.
    bool startWithoutSensorForTests();
    // Feeds a frame through the production onRawFrame()/raw-queue path.
    void injectRawFrameForTests(std::shared_ptr<sensor::RawFrame> raw) {
        onRawFrame(std::move(raw));
    }
    // Hook invoked synchronously by dispatchUiFrame() when qApp is null.
    static void registerUiFrameTestHookForTests(std::function<void(const sensor::FrameData&)> hook);
    // Number of UI-frame callbacks delivered through the null-qApp seam path.
    static int  uiFrameDeliveryCountForTests();
    // Clears hook + delivery counter (call before each seam test).
    static void resetUiFrameTestStateForTests();
#endif
};

} // namespace app
} // namespace kfusion
