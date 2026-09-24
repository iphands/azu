// gravity_contract (big-fix-two T3.13): the accelerometer-to-camera mapping and
// the tilt gate in include/tracking/Gravity.h.
//
//   A  a level Kinect (accelerometer +y, as recorded) means up = -y in camera
//      axes (y points down)
//   B  the tilt error is the pitch/roll between the pose and the reading
//   C  yaw is invisible to gravity
//   D  verdicts: Unknown without a reference, a reading or a gate; Unsteady
//      when |a| is far from the rest magnitude; Agree / Disagree at the gate
#include "tracking/Gravity.h"

#include <Eigen/Geometry>

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

using namespace kfusion::tracking;
constexpr float kDeg = 3.14159265f / 180.0f;
constexpr float kRest = 9.1f;   // the dev unit reads ~9.1 m/s^2 at rest

Eigen::Matrix4f rot(const Eigen::Vector3f& axis, float deg) {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,3>(0,0) = Eigen::AngleAxisf(deg * kDeg, axis.normalized()).toRotationMatrix();
    return T;
}

// The accelerometer reading a camera at `pose` sees when world up is -y.
void readingAt(const Eigen::Matrix4f& pose, float out[3], float magnitude = kRest) {
    const Eigen::Vector3f up_cam =
        pose.block<3,3>(0,0).transpose() * Eigen::Vector3f(0.0f, -1.0f, 0.0f) * magnitude;
    out[0] = -up_cam.x();
    out[1] = -up_cam.y();
    out[2] = up_cam.z();
}

bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

}  // namespace

int main() {
    const Eigen::Vector3f up_world(0.0f, -1.0f, 0.0f);

    // A
    const float level[3] = {0.0f, kRest, 0.0f};
    const Eigen::Vector3f up = accelUpInCamera(level);
    CHECK(near(up.x(), 0.0f, 1e-6f) && near(up.y(), -kRest, 1e-6f) && near(up.z(), 0.0f, 1e-6f),
          "A: level reading -> up = -y in camera axes");
    // Real first-frame reading of cap_001 (camera pitched up ~5 deg): up leans forward.
    const float cap001[3] = {0.395f, 9.060f, 0.746f};
    CHECK(accelUpInCamera(cap001).z() > 0.0f && accelUpInCamera(cap001).y() < 0.0f,
          "A: cap_001 start reading maps to up with a small forward lean");

    // B
    float a[3];
    for (float deg : {10.0f, 30.0f, 60.0f}) {
        readingAt(rot({1, 0, 0}, deg), a);
        const float t_true = tiltErrorDeg(rot({1, 0, 0}, deg), up_world, accelUpInCamera(a));
        const float t_level = tiltErrorDeg(Eigen::Matrix4f::Identity(), up_world, accelUpInCamera(a));
        CHECK(near(t_true, 0.0f, 0.05f), "B: the true pitched pose agrees with its reading");
        CHECK(near(t_level, deg, 0.05f), "B: a level pose is off by the pitch");
    }
    readingAt(rot({0, 0, 1}, 40.0f), a);
    CHECK(near(tiltErrorDeg(Eigen::Matrix4f::Identity(), up_world, accelUpInCamera(a)), 40.0f, 0.05f),
          "B: roll counts the same way");

    // C
    readingAt(Eigen::Matrix4f::Identity(), a);
    CHECK(near(tiltErrorDeg(rot({0, 1, 0}, 90.0f), up_world, accelUpInCamera(a)), 0.0f, 0.05f),
          "C: a 90 deg yaw is invisible to gravity");

    // D
    const GravityGate gate;
    float tilt = 0.0f;
    readingAt(rot({1, 0, 0}, 20.0f), a);
    const Eigen::Vector3f u20 = accelUpInCamera(a);
    const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
    CHECK(judgeGravity(I, false, up_world, kRest, true, u20, gate, &tilt) == GravityVerdict::Unknown &&
              tilt < 0.0f,
          "D: no reference -> Unknown, tilt -1");
    CHECK(judgeGravity(I, true, up_world, kRest, false, u20, gate) == GravityVerdict::Unknown,
          "D: no reading -> Unknown");
    GravityGate off;
    off.max_tilt_deg = 0.0f;
    CHECK(judgeGravity(I, true, up_world, kRest, true, u20, off) == GravityVerdict::Unknown,
          "D: gate 0 turns it off");
    CHECK(judgeGravity(I, true, up_world, kRest, true, u20, gate, &tilt) == GravityVerdict::Disagree &&
              near(tilt, 20.0f, 0.05f),
          "D: 20 deg off -> Disagree, tilt reported");
    CHECK(judgeGravity(rot({1, 0, 0}, 10.0f), true, up_world, kRest, true, u20, gate) ==
              GravityVerdict::Agree,
          "D: 10 deg off -> Agree");
    readingAt(rot({1, 0, 0}, 20.0f), a, kRest * 1.3f);
    CHECK(judgeGravity(I, true, up_world, kRest, true, accelUpInCamera(a), gate, &tilt) ==
              GravityVerdict::Unsteady && near(tilt, 20.0f, 0.05f),
          "D: |a| 30% over rest -> Unsteady (tilt still reported)");
    readingAt(rot({1, 0, 0}, 20.0f), a, kRest * 1.1f);
    CHECK(judgeGravity(I, true, up_world, kRest, true, accelUpInCamera(a), gate) ==
              GravityVerdict::Disagree,
          "D: |a| 10% over rest is still judged");

    if (g_failures == 0) {
        std::printf("gravity_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("gravity_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
