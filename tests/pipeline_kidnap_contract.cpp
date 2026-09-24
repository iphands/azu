// pipeline_kidnap_contract (relocalization rework): the real PipelineController
// recovers from kidnaps the old relocalizer could not (it only tried poses
// within ~14 deg of the last good one and refused any result over 0.15 m / 30
// deg from it).
//
// A furnished version of the synthetic room: the SyntheticScene room, box and
// sphere plus furniture on every side (cabinet, desk, chair, bin, lamp, shelf,
// pictures, door frame, ceiling light). The bare room is nearly square and
// 4-fold symmetric, so a view of bare walls 90 or 180 deg away is genuinely
// ambiguous (relocalizer_contract H), and a view where one small object is all
// that pins the pose along a wall sits at the relocalizer's constraint threshold
// (tracking/Relocalizer.h min_eig_ratio, 1e-3): with only a sphere in view this
// room read 0.7e-3..1.3e-3. So the room is furnished like a real one.
//
// Lockstep (waitIdle), accelerometer readings from the true pose (world up -y).
//   1  build the model: stand at the room centre looking 15 deg down and turn a
//      full circle at 1.5 deg/frame; tracking is never lost
//   2  kidnap events, each 4 frames without depth (lost after 3) and then a hold
//      at the new pose, each onto a view with furniture in it:
//        +60 deg; 180 deg about a point 0.12 m behind the camera (a body turn);
//        -125 deg onto the cabinet; -55 deg and 0.14 m (beyond the old 30 deg
//        gate). The model turn orbits the body like a person turning in place,
//        so each target is near a pose tracked before: the relocalizer refuses a
//        pose the camera has never been near (RelocParams::max_visited_*).
//      Weaker views are not targeted: a sphere with bare walls read 1.2e-3 and
//      the door frame 2.9e-3, against the 1e-3 threshold; whether those
//      re-acquire or wait is not what this test is about.
//      The targets stay away from the turn's loop seam at yaw 0: the model there
//      holds the start and the end of the turn (0.9 deg of drift apart), and even
//      continuous tracking to yaw -45 / 20 deg down ended 65 mm / 5.1 deg from the
//      truth (a kidnap there re-acquired 36 mm / 3.4 deg off: the model's best
//      fit, a loop-closure matter, not relocalization's).
//      A  re-acquired (Running) within 10 frames of the resume
//      B  at the end of the hold, the pose is within 2 cm / 2 deg of the truth (a
//         re-acquisition can only be as right as the model: the orbiting model
//         turn itself ends ~2 cm / 1.1 deg off)
//      C  nothing is integrated from the loss until re-acquisition + 2
//         (probation), and integration resumes afterwards
//      D  the keyframe database (fed during the turn) offered candidates
//      E  re-acquired within 2 frames: a keyframe near the target is tried on
//         the first lost frame with depth. The CPU sweep alone (6 candidates
//         per frame) took 5 frames for the body turn and 3 for -125 deg
//         (AZU_RELOC_FERNS=0).
//   3  negative: a bare wall 0.6 m away for 10 frames stays lost and integrates
//      nothing
#include "app/PipelineController.h"
#include "sensor/DepthValidity.h"
#include "sensor/KinectSensor.h"
#include "support/SyntheticScene.h"
#include "tracking/RelocHypotheses.h"

#include <QCoreApplication>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "pipeline_kidnap_contract needs AZU_PIPELINE_TEST_SEAM"
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
constexpr float kDeg = 3.14159265f / 180.0f;
const Eigen::Vector3f kUp(0.0f, -1.0f, 0.0f);

float boxSdf(const Eigen::Vector3f& p, const Eigen::Vector3f& c, const Eigen::Vector3f& h) {
    const Eigen::Vector3f q = (p - c).cwiseAbs() - h;
    return q.cwiseMax(0.0f).norm() + std::min(q.maxCoeff(), 0.0f);
}

// The synthetic room plus furniture, in scene coordinates (camera y down).
class FurnishedRoom {
public:
    float sdf(const Eigen::Vector3f& p) const {
        float d = base_.sdf(p);
        d = std::min(d, boxSdf(p, {1.0f, 0.3f, 0.5f}, {0.15f, 0.6f, 0.3f}));     // cabinet on the +x wall
        d = std::min(d, boxSdf(p, {-0.8f, 0.75f, 2.0f}, {0.2f, 0.15f, 0.2f}));   // box on the floor
        d = std::min(d, (p - Eigen::Vector3f(-0.9f, -0.5f, 0.2f)).norm() - 0.2f); // lamp
        d = std::min(d, boxSdf(p, {0.5f, -0.3f, -0.05f}, {0.4f, 0.1f, 0.1f}));   // shelf on the front wall
        d = std::min(d, boxSdf(p, {0.0f, 0.2f, 2.2f}, {0.5f, 0.03f, 0.2f}));      // desk on the back wall
        d = std::min(d, boxSdf(p, {-0.45f, 0.55f, 2.2f}, {0.03f, 0.35f, 0.15f})); // desk legs
        d = std::min(d, boxSdf(p, {0.45f, 0.55f, 2.2f}, {0.03f, 0.35f, 0.15f}));
        d = std::min(d, boxSdf(p, {0.1f, 0.05f, 2.3f}, {0.2f, 0.12f, 0.05f}));    // monitor on the desk
        d = std::min(d, boxSdf(p, {-1.17f, -0.3f, 1.0f}, {0.03f, 0.2f, 0.3f}));   // picture, -x wall
        d = std::min(d, boxSdf(p, {1.17f, -0.4f, 1.8f}, {0.03f, 0.25f, 0.2f}));   // picture, +x wall
        d = std::min(d, boxSdf(p, {0.8f, 0.5f, 1.9f}, {0.2f, 0.4f, 0.2f}));       // chair
        d = std::min(d, (p - Eigen::Vector3f(0.9f, 0.72f, 2.2f)).norm() - 0.18f);  // bin
        d = std::min(d, boxSdf(p, {-0.5f, 0.1f, -0.07f}, {0.45f, 0.8f, 0.04f}));  // door frame, front wall
        d = std::min(d, (p - Eigen::Vector3f(0.0f, -0.95f, 1.2f)).norm() - 0.15f); // ceiling light
        d = std::min(d, boxSdf(p, {-1.0f, 0.6f, 0.9f}, {0.15f, 0.3f, 0.25f}));    // side table, -x wall
        return d;
    }
    std::vector<float> renderDepth(const Eigen::Matrix4f& pose) const {
        const azu_test::Intrinsics K;
        std::vector<float> depth(static_cast<size_t>(K.width) * K.height, 0.0f);
        const Eigen::Matrix3f R = pose.block<3,3>(0,0);
        const Eigen::Vector3f o = pose.block<3,1>(0,3);
        #pragma omp parallel for schedule(dynamic, 8)
        for (int v = 0; v < K.height; ++v) {
            for (int u = 0; u < K.width; ++u) {
                const Eigen::Vector3f rc((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0f);
                const float norm = rc.norm();
                const Eigen::Vector3f d = R * rc / norm;
                float t = 0.05f;
                for (int i = 0; i < 256 && t < 8.0f; ++i) {
                    const float s = sdf(o + d * t);
                    if (s < 1e-5f) {
                        depth[static_cast<size_t>(v) * K.width + u] = t / norm;
                        break;
                    }
                    t += s;
                }
            }
        }
        return depth;
    }

private:
    azu_test::SyntheticScene base_;
};

// World = the first camera, at the room centre.
const Eigen::Matrix4f kRoomFromWorld = azu_test::makePose({0, 0, 0}, {0.0f, -0.05f, 1.15f});

Eigen::Matrix4f yawPitch(float yaw_deg, float pitch_deg, const Eigen::Vector3f& t = Eigen::Vector3f::Zero()) {
    return azu_test::makePose({0.0f, yaw_deg * kDeg, 0.0f}, t) *
           azu_test::makePose({pitch_deg * kDeg, 0.0f, 0.0f}, {0, 0, 0});
}

std::shared_ptr<RawFrame> rawFrom(const std::vector<float>& depth, uint64_t id, const Eigen::Matrix4f& truth) {
    auto f = std::make_shared<RawFrame>();
    f->frame_id    = id;
    f->depth_valid = true;
    f->rgb_valid   = true;
    for (size_t i = 0; i < depth.size(); ++i) {
        f->depth[i] = depth[i] > 0.0f ? kfusion::sensor::cpuDepthMetersToRaw(depth[i], 0.3f, 5.0f) : 0;
    }
    const Eigen::Vector3f up_cam = truth.block<3,3>(0,0).transpose() * kUp * 9.1f;
    f->accel[0] = -up_cam.x();
    f->accel[1] = -up_cam.y();
    f->accel[2] = up_cam.z();
    f->accel_valid = true;
    return f;
}

}  // namespace

int main() {
    CHECK(qApp == nullptr, "seam precondition: no QApplication");
    const FurnishedRoom room;

    PipelineController pc(kfusion::sensor::PreprocessBackend::CPU);
    auto hp = pc.hyperparamsSnapshot();
    hp.tsdf.origin = Eigen::Vector3f(-1.28f, -1.28f, -1.28f);   // centred on the first camera
    pc.setHyperparams(hp);
    CHECK(pc.startWithoutSensorForTests(), "seam start");

    uint64_t id = 0;
    auto feed = [&](const Eigen::Matrix4f& truth, bool blank = false) {
        std::vector<float> depth = room.renderDepth(kRoomFromWorld * truth);
        if (blank) std::fill(depth.begin(), depth.end(), 0.0f);
        pc.injectRawFrameForTests(rawFrom(depth, ++id, truth));
        CHECK(pc.waitIdle(30s), "frame handled in time");
        return pc.metricsSnapshot();
    };

    // 1: the model. The pipeline's world is its first camera, which already looks
    // 15 deg down: tracked poses compare against truth relative to it.
    int lost = 0;
    Eigen::Matrix4f pose = yawPitch(0.0f, -15.0f);
    const Eigen::Matrix4f first_inv = pose.inverse();
    auto trackedError = [&](const Eigen::Matrix4f& truth) {
        return azu_test::poseError(pc.currentPose(), first_inv * truth);
    };
    // A person turning in place: the camera orbits the body, 0.12 m in front of a
    // pivot 0.12 m behind the start position (as measured on spin360-slow).
    auto orbit = [](float yaw_deg, float pitch_deg) {
        const float y = yaw_deg * kDeg;
        const Eigen::Vector3f pivot(0.0f, 0.0f, -0.12f);
        return yawPitch(yaw_deg, pitch_deg, pivot + 0.12f * Eigen::Vector3f(std::sin(y), 0.0f, std::cos(y)));
    };
    for (int i = 0; i <= 240; ++i) {
        pose = orbit(1.5f * static_cast<float>(i), -15.0f);
        if (feed(pose).state == PipelineState::TrackingLost) ++lost;
    }
    for (int i = 0; i < 5; ++i) feed(pose);
    const auto e0 = trackedError(pose);
    std::printf("  model: full turn, lost frames %d, error %.1f mm / %.2f deg, integrated %d\n", lost,
                e0.trans_m * 1e3, e0.rot_deg, pc.metricsSnapshot().integrated_frames);
    CHECK(lost == 0, "1: the model turn never loses tracking");

    // 2: kidnaps.
    struct Event {
        const char*     name;
        Eigen::Matrix4f target;
    };
    // Every target view has furniture in it and is fully constrained (a view of
    // bare floor and wall is rightly refused: one direction slides).
    const Eigen::Matrix4f p1 = orbit(60.0f, -20.0f);
    const Eigen::Matrix4f p2 = kfusion::tracking::yawAboutUp(
        p1, 180.0f * kDeg, kUp, kfusion::tracking::bodyPivot(p1, kUp, 0.12f));
    const Eigen::Matrix4f p3 = orbit(115.0f, -20.0f);                                                 // cabinet
    const Eigen::Matrix4f p4 = yawPitch(60.0f, -20.0f, Eigen::Vector3f(p3.block<3,1>(0,3)) +
                                                           Eigen::Vector3f(0.1f, 0.0f, 0.1f));        // box, desk
    const Event events[] = {{"+60 deg", p1}, {"180 deg body turn", p2}, {"-125 deg", p3},
                            {"-55 deg + 0.14 m", p4}};
    for (const Event& ev : events) {
        const kfusion::tracking::PoseGap jump = kfusion::tracking::poseGap(pose, ev.target);
        std::printf("  %s: kidnap of %.1f deg / %.2f m\n", ev.name, jump.rot_deg, jump.trans_m);
        const int integrated_at_loss = pc.metricsSnapshot().integrated_frames;
        for (int i = 0; i < 4; ++i) feed(pose, /*blank=*/true);
        pose = ev.target;
        int recovered = -1;
        std::vector<int> integrated;
        for (int i = 0; i < 14; ++i) {
            const auto m = feed(pose);
            integrated.push_back(m.integrated_frames);
            if (recovered < 0 && m.state == PipelineState::Running) recovered = i;
            if (std::getenv("AZU_KIDNAP_VERBOSE")) {
                const auto e = trackedError(pose);
                const auto o = pc.lastRelocForTests();
                std::printf("    hold %2d state %d q %d err %6.1f mm / %5.2f deg  reloc gen %llu acc %d src %s rej %s cons %.3f eig %.1e\n", i,
                            static_cast<int>(m.state), m.tracking_quality, e.trans_m * 1e3, e.rot_deg,
                            static_cast<unsigned long long>(o.generation), o.accepted,
                            kfusion::tracking::hypothesisSourceName(o.source),
                            kfusion::tracking::relocRejectName(o.reject), o.consistent, o.eig_ratio);
            }
        }
        const auto err = trackedError(pose);
        const auto obs = pc.lastRelocForTests();
        std::printf("  %-18s re-acquired at hold frame %d (source %s, %d+%d solves, eig %.1e last run), final %.2f mm / %.3f deg, "
                    "integrated %d -> %d\n",
                    ev.name, recovered, kfusion::tracking::hypothesisSourceName(obs.source), obs.coarse_solves,
                    obs.refines, obs.eig_ratio, err.trans_m * 1e3, err.rot_deg, integrated_at_loss, integrated.back());
        const std::string n = ev.name;
        CHECK(recovered >= 0 && recovered < 10, "A: " + n + " re-acquired within 10 frames");
        CHECK(err.trans_m < 0.02f && err.rot_deg < 2.0f, "B: " + n + " within 2 cm / 2 deg");
        if (recovered >= 0 && recovered + 2 < static_cast<int>(integrated.size())) {
            CHECK(integrated[recovered + 2] == integrated_at_loss,
                  "C: " + n + " integrates nothing until re-acquisition + 2");
            CHECK(integrated.back() > integrated_at_loss, "C: " + n + " integration resumes");
        }
        CHECK(obs.keyframes > 0, "D: " + n + " keyframes offered");
        CHECK(recovered >= 0 && recovered <= 2, "E: " + n + " re-acquired within 2 frames");
    }

    // 3: a bare wall.
    {
        const int integrated_at_loss = pc.metricsSnapshot().integrated_frames;
        for (int i = 0; i < 4; ++i) feed(pose, /*blank=*/true);
        const Eigen::Matrix4f wall = yawPitch(-90.0f, 0.0f, {-0.6f, 0.0f, 0.0f});
        int running = 0;
        for (int i = 0; i < 10; ++i) running += feed(wall).state == PipelineState::Running;
        const auto obs = pc.lastRelocForTests();
        std::printf("  bare wall: running %d of 10, last reject %s, integrated %d -> %d\n", running,
                    kfusion::tracking::relocRejectName(obs.reject), integrated_at_loss,
                    pc.metricsSnapshot().integrated_frames);
        CHECK(running == 0, "3: a bare wall is never re-acquired");
        CHECK(pc.metricsSnapshot().integrated_frames == integrated_at_loss, "3: nothing integrated");
    }

    pc.stop();
    if (g_failures == 0) {
        std::printf("pipeline_kidnap_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("pipeline_kidnap_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
