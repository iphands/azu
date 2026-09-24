// pipeline_gravity_contract (big-fix-two T3.13): relocalization checks
// re-acquisitions against the accelerometer, holds integration on probation,
// and does not replay the relocalization jump as motion.
//
// Real PipelineController through the seam, synthetic room, the relocalization
// scenario of pipeline_relocalization_contract (smooth motion, 5 frames facing a
// 5 cm wall -> lost, return 12 deg / 3 cm away), with an accelerometer reading
// on every frame (world up = -y, rest magnitude 9.1 m/s^2).
//
//   run 1, the reading agrees with the true pose throughout:
//     A  tracking is lost, then recovers within 10 frames of the return
//     B  the pose after recovery is within 2 cm / 1 deg of the truth
//     C  probation: the re-acquired frame and the next two are not integrated;
//        integration resumes after
//     D  the frame after the re-acquisition is predicted from the re-acquired
//        pose with zero motion (the jump is not replayed as velocity)
//   run 2, frames 30-35 read a 40 deg roll the camera does not have:
//     E  no re-acquisition while the reading contradicts every pose
//     F  recovery within 10 frames once the reading is right (frame 36)
//     G  the pose after recovery is within 2 cm / 1 deg of the truth
// The window is short on purpose: the camera keeps turning 0.25 deg per frame,
// and the relocalizer's +-10 deg hypothesis grid reaches ~13 deg (12 deg offset
// plus 6 frames). With a 12-frame window it re-acquired at ~15 deg with the
// position 21.6 cm off: a relocalizer reach limit (big-fix-two T4.5), which
// gravity cannot see.
#include "app/PipelineController.h"
#include "sensor/DepthValidity.h"
#include "sensor/KinectSensor.h"
#include "support/SyntheticScene.h"

#include <QCoreApplication>

#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "pipeline_gravity_contract needs AZU_PIPELINE_TEST_SEAM"
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

constexpr float kRest = 9.1f;

// Depth rendered at `truth`; accelerometer as a camera at `accel_pose` reads it.
std::shared_ptr<RawFrame> rawFrom(const std::vector<float>& depth, uint64_t id,
                                  const Eigen::Matrix4f& accel_pose) {
    auto f = std::make_shared<RawFrame>();
    f->frame_id    = id;
    f->depth_valid = true;
    f->rgb_valid   = true;
    for (size_t i = 0; i < depth.size(); ++i) {
        f->depth[i] = depth[i] > 0.0f ? kfusion::sensor::cpuDepthMetersToRaw(depth[i], 0.3f, 5.0f) : 0;
    }
    const Eigen::Vector3f up_cam =
        accel_pose.block<3,3>(0,0).transpose() * Eigen::Vector3f(0.0f, -1.0f, 0.0f) * kRest;
    f->accel[0] = -up_cam.x();
    f->accel[1] = -up_cam.y();
    f->accel[2] = up_cam.z();
    f->accel_valid = true;
    return f;
}

struct Run {
    bool entered_lost = false;
    int  recovered_at = -1;
    int  running_while_bad = 0;
    azu_test::PoseError final_err{0.0f, 0.0f};
    std::vector<int> integrated;      // integrated_frames after each frame
    azu_test::PoseError after_reacquire_prediction{-1.0f, -1.0f};
};

Run run(const azu_test::SyntheticScene& scene, int bad_from, int bad_to, float roll_rad) {
    Run r;
    PipelineController pc(kfusion::sensor::PreprocessBackend::CPU);
    pc.setFrameReadyCallback([](const kfusion::sensor::FrameData&) {});
    CHECK(pc.startWithoutSensorForTests(), "seam start");

    auto smooth = [](int i) {
        return azu_test::makePose({0.0f, 0.00436f * i, 0.0f}, {0.002f * i, 0.0f, 0.0f});
    };
    const Eigen::Matrix4f last_good = smooth(24);
    const Eigen::Matrix4f away      = azu_test::makePose({0.0f, 3.14159f, 0.0f}, {0.0f, 0.0f, 0.0f});
    const Eigen::Matrix4f offset    = azu_test::makePose({0.0f, 0.2094f, 0.0f}, {0.03f, 0.0f, 0.0f});
    const Eigen::Matrix4f roll      = azu_test::makePose({0.0f, 0.0f, roll_rad}, {0.0f, 0.0f, 0.0f});

    Eigen::Matrix4f truth = Eigen::Matrix4f::Identity();
    for (int i = 0; i < 70; ++i) {
        if (i < 25)      truth = smooth(i);
        else if (i < 30) truth = away;
        else             truth = last_good * offset * smooth(i - 30);
        const bool bad = i >= bad_from && i < bad_to;
        const uint64_t before = pc.trackedFrameCountForTests();
        pc.injectRawFrameForTests(rawFrom(scene.renderDepth(truth), static_cast<uint64_t>(i + 1),
                                          bad ? Eigen::Matrix4f(truth * roll) : truth));
        const auto deadline = std::chrono::steady_clock::now() + 3s;
        while (i > 0 && pc.trackedFrameCountForTests() == before &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(2ms);
        }
        std::this_thread::sleep_for(60ms);   // let an enqueued frame integrate
        const auto m = pc.metricsSnapshot();
        r.integrated.push_back(m.integrated_frames);
        if (m.state == PipelineState::TrackingLost) r.entered_lost = true;
        if (bad && m.state == PipelineState::Running && i >= 30) ++r.running_while_bad;
        if (i >= 30 && r.recovered_at < 0 && m.state == PipelineState::Running) r.recovered_at = i;
        if (r.recovered_at >= 0 && i == r.recovered_at + 1) {
            const auto obs = pc.lastMotionModelForTests();
            r.after_reacquire_prediction = azu_test::poseError(obs.prev_pose, obs.predicted_pose);
        }
    }
    r.final_err = azu_test::poseError(pc.currentPose(), truth);
    pc.stop();
    return r;
}

}  // namespace

int main() {
    CHECK(qApp == nullptr, "seam precondition: no QApplication");
    const azu_test::SyntheticScene scene;

    const Run a = run(scene, -1, -1, 0.0f);
    std::printf("  run 1: lost=%d recovered_at=%d final %.2f mm / %.3f deg, "
                "next prediction %.3f mm / %.4f deg from the re-acquired pose\n",
                a.entered_lost, a.recovered_at, a.final_err.trans_m * 1e3, a.final_err.rot_deg,
                a.after_reacquire_prediction.trans_m * 1e3, a.after_reacquire_prediction.rot_deg);
    CHECK(a.entered_lost, "A: looking at a 5 cm wall enters TrackingLost");
    CHECK(a.recovered_at >= 30 && a.recovered_at < 40, "A: recovers within 10 frames");
    CHECK(a.final_err.trans_m < 0.02f && a.final_err.rot_deg < 1.0f, "B: pose within 2 cm / 1 deg");
    if (a.recovered_at >= 30 && a.recovered_at + 6 < static_cast<int>(a.integrated.size())) {
        const int at = a.recovered_at;
        std::printf("  run 1: integrated frames around recovery: %d | %d %d %d | %d %d %d\n",
                    a.integrated[at - 1], a.integrated[at], a.integrated[at + 1], a.integrated[at + 2],
                    a.integrated[at + 3], a.integrated[at + 4], a.integrated[at + 5]);
        CHECK(a.integrated[at + 2] == a.integrated[at - 1],
              "C: the re-acquired frame and the next two are not integrated");
        CHECK(a.integrated[at + 6] > a.integrated[at + 2], "C: integration resumes after probation");
    } else {
        CHECK(false, "C: recovery early enough to observe probation");
    }
    // poseError inverts a float 4x4 and takes acos near 1: identical poses read
    // up to ~0.04 deg (seen under load). A replayed jump here is ~12 deg.
    CHECK(a.after_reacquire_prediction.trans_m >= 0.0f &&
              a.after_reacquire_prediction.trans_m < 1e-4f &&
              a.after_reacquire_prediction.rot_deg < 0.1f,
          "D: the frame after a re-acquisition is predicted with zero motion");

    const Run b = run(scene, 30, 36, 40.0f * 3.14159265f / 180.0f);
    std::printf("  run 2: lost=%d running while the reading is wrong=%d recovered_at=%d final %.2f mm / %.3f deg\n",
                b.entered_lost, b.running_while_bad, b.recovered_at, b.final_err.trans_m * 1e3,
                b.final_err.rot_deg);
    CHECK(b.running_while_bad == 0, "E: no re-acquisition while the reading contradicts the pose");
    CHECK(b.recovered_at >= 36 && b.recovered_at < 46, "F: recovers within 10 frames of a right reading");
    CHECK(b.final_err.trans_m < 0.02f && b.final_err.rot_deg < 1.0f, "G: pose within 2 cm / 1 deg");

    if (g_failures == 0) {
        std::printf("pipeline_gravity_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("pipeline_gravity_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
