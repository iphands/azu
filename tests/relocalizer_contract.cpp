// relocalizer_contract (relocalization rework, step 7): tracking::Relocalizer on
// the CPU backend (TSDFVolume::raycast + ICPTracker::track) against an exact
// model of the synthetic room (tests/support/RoomModel.h). The camera was last
// tracked at the room centre, level; the live frame is elsewhere, looking 20 deg
// down as a room scan does.
// CPU budget as the pipeline uses it: 6 sweep candidates and 2 refines per call.
//
// Views are chosen to constrain every direction. Measured from the centre (the
// weakest/strongest eigenvalue ratio of a converged ICP): level views of one
// bare wall read 8e-6..4e-4 and are rightly refused (verification needs >=
// 1e-3); 20 deg down the floor joins in and they read 3e-3..2e-2, except
// straight back from +z (7e-5). The room is also nearly square, so a view of one
// bare corner and the floor matches the corner 90 deg over: the kidnap targets
// look at the sphere (yaw -40 deg, 20 deg down), as a real room has furniture,
// and H checks that the bare-corner case is refused as ambiguous.
//
//   A  in-place kidnaps of 45, 90 and 180 deg, and 180 deg about the body pivot
//      (0.12 m behind the camera), found by the sweep within ceil(34/6) calls,
//      within 1 cm / 0.5 deg
//   B  0.2 m + 30 deg away, with a keyframe near the truth: found on the first
//      call, from the keyframe
//   C  no gravity reading at all, a level 90 deg kidnap: still found (old grid +
//      sweep about the default up)
//   D  a steady reading with a 40 deg roll the camera does not have: never
//      accepted
//   E  facing a bare wall 0.6 m away: never accepted (one wall is a guess)
//   F  a frame without depth costs no solve
//   G  a basin whose probation failed is refused
//   H  a 180 deg kidnap onto a bare corner of the square room: ambiguous, refused
#include "support/RoomModel.h"
#include "tracking/Relocalizer.h"
#include "sensor/FrameData.h"

#include <cmath>
#include <cstdio>
#include <string>

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

using namespace kfusion;
using namespace kfusion::tracking;
constexpr float kDeg = 3.14159265f / 180.0f;
const Eigen::Vector3f kUp(0.0f, -1.0f, 0.0f);

class CpuBackend {
public:
    CpuBackend(const azu_test::RoomModel& room, const std::vector<float>& depth) : room_(room) {
        sensor::FrameData f;
        azu_test::fillFrame(depth, f, room.intrinsics());
        sensor::buildFramePyramid(f, pyr_);
        tracker_.setIntrinsics(room.camera());
    }
    RelocBackend hooks() {
        RelocBackend b;
        b.render = [this](const Eigen::Matrix4f& pose, RenderSize size, bool) -> const ModelFrame& {
            ModelFrame& m = size == RenderSize::Coarse ? coarse_ : size == RenderSize::Refine ? refine_ : full_;
            const sensor::CameraIntrinsics k = sensor::scaleIntrinsics(room_.camera(), 640, 480, m.width, m.height);
            room_.volume().raycast(pose, k.fx, k.fy, k.cx, k.cy, m.width, m.height, m.vertices.data(),
                                   m.normals.data(), m.colors.data());
            m.pose = pose;
            return m;
        };
        b.solve = [this](const ModelFrame& m, const Eigen::Matrix4f& est, const ICPParams& p) {
            tracker_.setParams(p);
            return tracker_.track(pyr_, m, est, m.pose);
        };
        return b;
    }

private:
    const azu_test::RoomModel& room_;
    sensor::FramePyramid pyr_;
    ICPTracker tracker_;
    ModelFrame coarse_{160, 120}, refine_{320, 240}, full_{640, 480};
};

GravityContext gravityFor(const Eigen::Matrix4f& reading_pose) {
    GravityContext g;
    g.have_ref = true;
    g.up_world = kUp;
    g.ref_norm = 9.1f;
    g.have_reading = true;
    g.up_cam = reading_pose.block<3,3>(0,0).transpose() * kUp * 9.1f;
    return g;
}

RelocParams cpuParams() {
    RelocParams p;
    p.sweep_budget = 6;
    p.refine_top = 2;
    return p;
}

struct Attempt {
    bool accepted = false;
    int calls = 0;
    RelocOutcome last;
    azu_test::PoseError err{-1.0f, -1.0f};
    bool saw_blacklisted = false;
};

const Eigen::Matrix4f kLastGood = azu_test::RoomModel::yawPitch(15.0f * kDeg, 0.0f);
constexpr float kSphereYaw = -40.0f;   // the view with the sphere in it

Eigen::Matrix4f level(float yaw_deg) { return azu_test::RoomModel::yawPitch(yaw_deg * kDeg, 0.0f); }

// A camera at `from`'s position, turned `yaw_deg` from it (the RoomModel::yawPitch
// sense: about +y, so -yaw about world up, which is -y) and looking `pitch_deg`
// (negative = down).
Eigen::Matrix4f turned(const Eigen::Matrix4f& from, float yaw_deg, float pitch_deg, bool body_pivot = false) {
    const Eigen::Vector3f pivot =
        body_pivot ? bodyPivot(from, kUp, 0.12f) : Eigen::Vector3f(from.block<3,1>(0,3));
    return yawAboutUp(from, -yaw_deg * kDeg, kUp, pivot) *
           azu_test::makePose({pitch_deg * kDeg, 0.0f, 0.0f}, {0, 0, 0});
}

Attempt relocalize(const azu_test::RoomModel& room, const Eigen::Matrix4f& truth, int max_calls,
                   const GravityContext& g, std::vector<Eigen::Matrix4f> keyframes = {},
                   Relocalizer* reloc_in = nullptr, const std::vector<float>* depth_in = nullptr,
                   const Eigen::Matrix4f& last_good = kLastGood) {
    const std::vector<float> depth = depth_in ? *depth_in : room.depthAt(truth);
    CpuBackend be(room, depth);
    Relocalizer local(cpuParams());
    Relocalizer& reloc = reloc_in ? *reloc_in : local;
    reloc.beginLoss();
    RelocRequest req;
    req.last_good = last_good;
    req.model_pose = last_good;
    req.keyframes = keyframes;
    req.gravity = g;
    req.live_depth = depth.data();
    req.live_w = room.intrinsics().width;
    req.live_h = room.intrinsics().height;
    Attempt a;
    for (a.calls = 1; a.calls <= max_calls; ++a.calls) {
        a.last = reloc.run(req, be.hooks());
        if (a.last.reject == RelocReject::Blacklisted) a.saw_blacklisted = true;
        if (a.last.accepted) {
            a.accepted = true;
            a.err = azu_test::poseError(a.last.result.pose, truth);
            break;
        }
    }
    return a;
}

void report(const char* what, const Attempt& a) {
    std::printf("  %-34s %s after %d call(s), source %s, reject %s, solves %d+%d, consistent %.3f, err %.2f mm / %.3f deg\n",
                what, a.accepted ? "ACCEPTED" : "not accepted", std::min(a.calls, 99),
                hypothesisSourceName(a.last.source), relocRejectName(a.last.reject), a.last.coarse_solves,
                a.last.refines, a.last.consistency.consistentFraction(), a.err.trans_m * 1e3, a.err.rot_deg);
}

}  // namespace

int main() {
    const azu_test::RoomModel room;
    const int sweep_calls = (34 + 5) / 6;

    // A
    for (float deg : {45.0f, 90.0f, 180.0f}) {
        const Eigen::Matrix4f from = level(kSphereYaw - deg);
        const Eigen::Matrix4f truth = turned(from, deg, -20.0f);
        const Attempt a = relocalize(room, truth, sweep_calls, gravityFor(truth), {}, nullptr, nullptr, from);
        const std::string what = "A: in place " + std::to_string(static_cast<int>(deg)) + " deg";
        report(what.c_str(), a);
        CHECK(a.accepted && a.err.trans_m < 0.01f && a.err.rot_deg < 0.5f, what + ": found within 1 cm / 0.5 deg");
    }
    {
        const Eigen::Matrix4f from = level(kSphereYaw - 180.0f);
        const Eigen::Matrix4f truth = turned(from, 180.0f, -20.0f, /*body_pivot=*/true);
        const Attempt a = relocalize(room, truth, sweep_calls, gravityFor(truth), {}, nullptr, nullptr, from);
        report("A: 180 deg about the body pivot", a);
        CHECK(a.accepted && a.err.trans_m < 0.01f && a.err.rot_deg < 0.5f,
              "A: 180 deg about the body pivot found within 1 cm / 0.5 deg");
    }

    // B
    {
        const Eigen::Matrix4f truth = kLastGood * azu_test::makePose({-0.35f, 30.0f * kDeg, 0.0f}, {0.2f, 0.0f, 0.05f});
        const Eigen::Matrix4f kf = truth * azu_test::makePose({0.0f, 5.0f * kDeg, 0.0f}, {0.02f, 0.0f, 0.0f});
        const Attempt a = relocalize(room, truth, 1, gravityFor(truth), {kf});
        report("B: 0.2 m + 30 deg, keyframe", a);
        CHECK(a.accepted && a.calls == 1 && a.last.source == HypothesisSource::Keyframe,
              "B: found on the first call from the keyframe");
        CHECK(a.err.trans_m < 0.01f && a.err.rot_deg < 0.5f, "B: within 1 cm / 0.5 deg");
    }

    // C: level views are only constrained facing a corner, so start 30 deg to
    // the right and turn 90 deg to the corner at 60 deg (ratio 2e-2).
    {
        const Eigen::Matrix4f from = azu_test::RoomModel::yawPitch(-30.0f * kDeg, 0.0f);
        const Eigen::Matrix4f truth = azu_test::RoomModel::yawPitch(60.0f * kDeg, 0.0f);
        const Attempt a = relocalize(room, truth, (34 + 8 + 5) / 6, GravityContext{}, {}, nullptr, nullptr, from);
        report("C: 90 deg, no gravity", a);
        CHECK(a.accepted && a.err.trans_m < 0.01f && a.err.rot_deg < 0.5f, "C: found without gravity");
    }

    // D
    {
        const Eigen::Matrix4f truth = turned(kLastGood, kSphereYaw - 15.0f, -20.0f);
        const Eigen::Matrix4f rolled = truth * azu_test::makePose({0.0f, 0.0f, 40.0f * kDeg}, {0, 0, 0});
        const Attempt a = relocalize(room, truth, sweep_calls, gravityFor(rolled));
        report("D: wrong steady reading (40 deg roll)", a);
        CHECK(!a.accepted, "D: never accepted against a contradicting reading");
    }

    // E
    {
        const Eigen::Matrix4f truth = azu_test::makePose({0.0f, 90.0f * kDeg, 0.0f}, {0.6f, 0.0f, 0.0f});
        const Attempt a = relocalize(room, truth, sweep_calls, gravityFor(truth));
        report("E: bare wall 0.6 m away", a);
        CHECK(!a.accepted, "E: a bare wall is never accepted");
    }

    // F
    {
        const std::vector<float> none(static_cast<size_t>(room.intrinsics().width) * room.intrinsics().height, 0.0f);
        const Attempt a = relocalize(room, Eigen::Matrix4f::Identity(), 1, gravityFor(Eigen::Matrix4f::Identity()), {},
                                     nullptr, &none);
        report("F: no depth", a);
        CHECK(!a.accepted && a.last.reject == RelocReject::NoDepth && a.last.coarse_solves == 0,
              "F: no depth, no solve");
    }

    // G
    {
        const Eigen::Matrix4f truth = turned(kLastGood, kSphereYaw - 15.0f, -20.0f);
        Relocalizer reloc(cpuParams());
        reloc.rejectBasin(truth);
        const Attempt a = relocalize(room, truth, sweep_calls, gravityFor(truth), {}, &reloc);
        report("G: blacklisted basin", a);
        CHECK(!a.accepted && a.saw_blacklisted, "G: a failed basin is refused");
    }

    // H
    {
        const Eigen::Matrix4f truth = turned(kLastGood, 180.0f, -20.0f);
        const Attempt a = relocalize(room, truth, sweep_calls, gravityFor(truth));
        report("H: 180 deg onto a bare corner", a);
        CHECK(!a.accepted && a.last.reject == RelocReject::Ambiguous,
              "H: the square room's bare corner is ambiguous and refused");
    }

    if (g_failures == 0) {
        std::printf("relocalizer_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("relocalizer_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
