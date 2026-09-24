#pragma once

// Per-frame pipeline trace (CSV), for studying a run offline. One "track" row per
// graded frame from the tracking worker and one "integ" row per integrated frame
// from the integration worker. Writes are buffered and mutex-guarded; they never
// happen on the sensor (libusb) thread. Enabled by AZU_TRACE=<file> or
// PipelineController::setTracePath().

#include <Eigen/Core>

#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

namespace kfusion {
namespace app {

struct FrameTraceRow {
    const char* kind = "track";      // "track" | "integ"
    uint64_t frame_id = 0;
    double   t_ms = 0.0;             // sensor time of the depth frame (unwrapped)
    // tracking
    int      quality = -1;           // tracking::TrackQuality, -1 = first frame
    bool     relocalizing = false;
    bool     queued_for_integration = false;
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f predicted = Eigen::Matrix4f::Identity();
    int      inliers = 0, valid_live = 0, projected = 0, valid_model = 0;
    int      dist_filtered = 0, angle_filtered = 0;
    float    rms_m = 0.0f, final_step = 0.0f;
    bool     converged = false;
    uint64_t model_frame_id = 0;     // frame the model image was raycast from
    float    outside_volume = 0.0f;  // fraction of live points outside the TSDF box
    int      degenerate_dofs = 0;    // unobservable directions held at the previous pose
    float    ms_preprocess = 0.0f, ms_icp = 0.0f, ms_track = 0.0f;
    // integration
    float    ms_integrate = 0.0f, ms_raycast = 0.0f;
    // gravity (tracking/Gravity.h): pose vs accelerometer, -1 = not measurable
    float    tilt_err_deg = -1.0f;
    // relocalization (tracking/Relocalizer.h); only on relocalizing rows
    int         reloc_solves = 0, reloc_refines = 0;
    const char* reloc_source = "";
    const char* reloc_reject = "";
    float       reloc_consistent = -1.0f, reloc_violation = -1.0f, reloc_coverage = -1.0f;
    float       reloc_eig = -1.0f;
};

class FrameTrace {
public:
    bool open(const std::string& path);
    bool isOpen() const { return open_; }
    void write(const FrameTraceRow& row);
    void close();
    ~FrameTrace() { close(); }

private:
    std::mutex    mtx_;
    std::ofstream out_;
    bool          open_ = false;
    int           unflushed_ = 0;
};

} // namespace app
} // namespace kfusion
