// pipeline_spin_contract: stand in the middle of a room and turn a full 360
// degrees. This is the capture that broke after ~15 degrees on real hardware.
//
// Real PipelineController through the seam, lockstep (each frame is tracked, and
// if integrated also raycast, before the next is injected). The camera sits at the
// centre of the closed tests/support/SyntheticScene room and yaws clockwise (to
// the right, +y rotation in camera coordinates) 1.5 degrees per frame -- 45
// deg/s at 30 Hz -- for 240 frames, then holds still. Depth is quantized to raw
// Kinect 11-bit codes and passes the production preprocessing.
//
// The pipeline's world frame is the first camera, so the TSDF volume is placed
// centred on it (the room fits: walls <= 1.25 m away, 2.56 m cube). Mostly the
// camera sees one or two bare walls, so several motion directions are
// unobservable; that is what keepObservableMotion() is for.
//
// Env: AZU_SPIN_VERBOSE=1 prints every 10th frame; AZU_SPIN_PRESET=room|chair
// applies that GUI preset (Chair is a box in front of the camera for objects and
// is expected to fail a full turn); chair-legacy is the field-report config.
//
//   A  tracking is never lost for the whole turn
//   B  >= 90% of frames are graded Good (integrated)
//   C  after 360 degrees the pose is within 5 cm / 2 degrees of the truth
#include "app/PipelineController.h"
#include "gui/FusionUiModel.h"
#include "sensor/DepthValidity.h"
#include "sensor/KinectSensor.h"
#include "support/SyntheticScene.h"

#include <QCoreApplication>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <algorithm>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "pipeline_spin_contract needs AZU_PIPELINE_TEST_SEAM"
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

using kfusion::app::PipelineController;
using kfusion::app::PipelineState;
using kfusion::sensor::RawFrame;
using namespace std::chrono_literals;

std::shared_ptr<RawFrame> rawFrom(const std::vector<float>& depth, uint64_t id) {
    auto f = std::make_shared<RawFrame>();
    f->frame_id    = id;
    f->depth_valid = true;
    f->rgb_valid   = true;
    for (size_t i = 0; i < depth.size(); ++i) {
        f->depth[i] = depth[i] > 0.0f ? kfusion::sensor::cpuDepthMetersToRaw(depth[i], 0.3f, 5.0f) : 0;
    }
    for (size_t i = 0; i < f->rgb.size(); ++i) f->rgb[i] = static_cast<uint8_t>(90 + (i % 7) * 20);
    return f;
}

// Wait until `pred` holds or `timeout` passes.
template <class Pred>
bool waitFor(Pred pred, std::chrono::milliseconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!pred()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

}  // namespace

int main() {
    CHECK(qApp == nullptr, "seam precondition: no QApplication");
    const bool verbose = std::getenv("AZU_SPIN_VERBOSE") != nullptr;
    const azu_test::SyntheticScene scene;
    // Camera at the room centre (room box centre, eye height).
    const Eigen::Matrix4f room_from_start =
        azu_test::makePose({0.0f, 0.0f, 0.0f}, {0.0f, -0.05f, 1.15f});

    PipelineController pc(kfusion::sensor::PreprocessBackend::CPU);
    auto hp = pc.hyperparamsSnapshot();
    hp.tsdf.origin = Eigen::Vector3f(-1.28f, -1.28f, -1.28f);   // centred on the start camera
    // AZU_SPIN_PRESET=room|chair applies the GUI preset (gui/FusionUiModel.h).
    // "chair-legacy" is the field report: the Chair box left at the old default
    // origin, off-centre in front of the camera.
    if (const char* preset = std::getenv("AZU_SPIN_PRESET")) {
        const std::string name = preset;
        if (name == "room") {
            hp = kfusion::gui::applyFusionPreset(hp, kfusion::gui::FusionPreset::kRoom);
        } else if (name == "chair" || name == "chair-legacy") {
            hp = kfusion::gui::applyFusionPreset(hp, kfusion::gui::FusionPreset::kChair);
            if (name == "chair-legacy") hp.tsdf.origin = Eigen::Vector3f(-1.28f, -1.28f, 0.0f);
        }
    }
    // Experiment overrides on top: AZU_SPIN_VOXEL / _RES / _TRUNC / _DIST / _ANGLE.
    auto envf = [](const char* k, float& v) { if (const char* e = std::getenv(k)) v = std::strtof(e, nullptr); };
    envf("AZU_SPIN_VOXEL", hp.tsdf.voxel_size);
    envf("AZU_SPIN_TRUNC", hp.tsdf.truncation);
    envf("AZU_SPIN_DIST", hp.icp.dist_threshold);
    envf("AZU_SPIN_ANGLE", hp.icp.angle_threshold);
    if (const char* e = std::getenv("AZU_SPIN_RES")) hp.tsdf.resolution = std::atoi(e);
    if (std::getenv("AZU_SPIN_VOXEL") || std::getenv("AZU_SPIN_RES")) {
        const float half = 0.5f * hp.tsdf.voxel_size * static_cast<float>(hp.tsdf.resolution);
        hp.tsdf.origin = Eigen::Vector3f::Constant(-half);
    }
    pc.setHyperparams(hp);
    CHECK(pc.startWithoutSensorForTests(), "seam start");

    // AZU_SPIN_GLITCH=<start>,<count>: that many frames with no depth at all
    // mid-turn (the camera keeps turning), e.g. a USB hiccup.
    int glitch_start = -1, glitch_count = 0;
    if (const char* g = std::getenv("AZU_SPIN_GLITCH")) std::sscanf(g, "%d,%d", &glitch_start, &glitch_count);
    // AZU_SPIN_STEP_DEG: turn rate per frame (default 1.5).
    float step_deg = 1.5f;
    if (const char* st = std::getenv("AZU_SPIN_STEP_DEG")) step_deg = std::strtof(st, nullptr);
    constexpr int   kHoldFrames = 10;
    const float     kStepRad    = step_deg * 3.14159265f / 180.0f;
    const int       turn_frames = static_cast<int>(std::lround(360.0f / step_deg));
    int lost_frames = 0, good = 0, graded = 0, first_lost = -1, timeouts = 0;
    Eigen::Matrix4f truth = Eigen::Matrix4f::Identity();
    float worst_trans = 0.0f, worst_rot = 0.0f;
    for (int i = 0; i < turn_frames + kHoldFrames; ++i) {
        const float yaw = kStepRad * static_cast<float>(std::min(i, turn_frames));
        truth = azu_test::makePose({0.0f, yaw, 0.0f}, {0.0f, 0.0f, 0.0f});
        const uint64_t tracked_before = pc.trackedFrameCountForTests();
        const int integrated_before = pc.metricsSnapshot().integrated_frames;
        std::vector<float> depth = scene.renderDepth(room_from_start * truth);
        if (i >= glitch_start && i < glitch_start + glitch_count) {
            std::fill(depth.begin(), depth.end(), 0.0f);   // sensor glitch: no depth
        }
        pc.injectRawFrameForTests(rawFrom(depth, static_cast<uint64_t>(i + 1)));
        if (i == 0) {
            waitFor([&] { return pc.metricsSnapshot().integrated_frames > integrated_before; }, 3s);
            continue;
        }
        if (!waitFor([&] { return pc.trackedFrameCountForTests() > tracked_before; }, 3s)) {
            ++timeouts;
            continue;
        }
        const auto m = pc.metricsSnapshot();
        ++graded;
        if (m.tracking_quality == 0) {
            ++good;
            // Lockstep: the next frame tracks against the model this one produced.
            waitFor([&] { return pc.metricsSnapshot().integrated_frames > integrated_before; }, 3s);
        }
        if (m.state == PipelineState::TrackingLost) {
            ++lost_frames;
            if (first_lost < 0) first_lost = i;
        }
        const Eigen::Matrix4f est = pc.currentPose();
        const auto e = azu_test::poseError(est, truth);
        worst_trans = std::max(worst_trans, e.trans_m);
        worst_rot = std::max(worst_rot, e.rot_deg);
        if (verbose && i % 10 == 0) {
            const float est_yaw = std::atan2(est(0, 2), est(2, 2)) * 57.2958f;
            std::printf("  frame %3d yaw %6.1f est %6.1f  t=(%+.3f %+.3f %+.3f)  err %.1f mm / %.2f deg"
                        "  q=%d model=%d rms=%.4f\n",
                        i, yaw * 57.2958f, est_yaw, est(0, 3), est(1, 3), est(2, 3),
                        e.trans_m * 1e3, e.rot_deg, m.tracking_quality, m.icp_valid_model,
                        std::sqrt(std::max(0.0f, m.icp_error)));
        }
    }
    const auto fin = azu_test::poseError(pc.currentPose(), truth);
    pc.stop();

    std::printf("  spin: first lost at frame %d (%.1f deg), lost frames %d, good %d/%d, timeouts %d\n",
                first_lost, first_lost < 0 ? 0.0f : step_deg * first_lost, lost_frames, good, graded,
                timeouts);
    std::printf("  final error %.1f mm / %.2f deg, worst %.1f mm / %.2f deg\n", fin.trans_m * 1e3,
                fin.rot_deg, worst_trans * 1e3, worst_rot);
    CHECK(timeouts == 0, "every frame was tracked");
    CHECK(lost_frames == 0 || glitch_count >= 3, "A: tracking is never lost during a 360 degree turn");
    CHECK(good * 10 >= graded * 9, "B: >= 90% of frames graded Good");
    CHECK(fin.trans_m < 0.05f && fin.rot_deg < 2.0f, "C: back at the start within 5 cm / 2 deg");

    if (g_failures == 0) {
        std::printf("pipeline_spin_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("pipeline_spin_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
