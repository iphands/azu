#pragma once

// Gravity from the Kinect v1 accelerometer (big-fix-two T3.13).
//
// libfreenect reports the accelerometer's specific force in m/s^2 in its own
// axes (RawFrame::accel). At rest that is the reaction to gravity: it points
// UP. The accelerometer axes are the depth camera's (x right, y down, z
// forward) turned 180 deg about z: up in camera axes = (-ax, -ay, +az). That is
// the sign flip azu_replay's search picked on both handheld takes (spin360-slow
// and cap_001), with a 1-3 deg mean tilt error while tracking.
//
// The world is the first camera, so up_world is the first frame's up. A pose
// (world-from-camera) predicts up in camera axes as R^T * up_world; the angle
// to the measured up is the tilt error. It cannot see yaw, only pitch and roll.
// A hand that accelerates bends the measurement, so a sample counts only when
// its magnitude stays near the reference (rest) magnitude.
//
// Used as a gate: a relocalization whose pose disagrees with gravity is a wrong
// basin (cap_001: floor fitted to a wall at 87-93 deg, and a 17 deg Poor fit).

#include <Eigen/Core>

#include <algorithm>
#include <cmath>

namespace kfusion {
namespace tracking {

inline Eigen::Vector3f accelUpInCamera(const float a[3]) {
    return Eigen::Vector3f(-a[0], -a[1], a[2]);
}

struct GravityGate {
    float max_tilt_deg = 15.0f;   // pose vs accelerometer; tracked frames reach ~12 deg
    float max_norm_dev = 0.2f;    // |a| within 20% of the reference magnitude
};

// Angle (deg) between the up direction `pose` predicts and the measured one.
inline float tiltErrorDeg(const Eigen::Matrix4f& pose, const Eigen::Vector3f& up_world,
                          const Eigen::Vector3f& up_cam) {
    const Eigen::Vector3f pred = pose.block<3,3>(0,0).transpose() * up_world;
    const float c = pred.normalized().dot(up_cam.normalized());
    return std::acos(std::max(-1.0f, std::min(1.0f, c))) * 57.2957795f;
}

// A sample is steady enough to judge tilt when its magnitude is within
// `max_dev` (relative) of the reference magnitude.
inline bool accelSteady(const Eigen::Vector3f& up_cam, float ref_norm, float max_dev) {
    return ref_norm > 0.0f && std::fabs(up_cam.norm() / ref_norm - 1.0f) <= max_dev;
}

enum class GravityVerdict {
    Unknown,    // no reference or no reading: nothing to judge (sources without an accelerometer)
    Unsteady,   // a reading, but the hand is accelerating: cannot judge
    Agree,
    Disagree,
};

inline GravityVerdict judgeGravity(const Eigen::Matrix4f& pose, bool have_ref,
                                   const Eigen::Vector3f& up_world, float ref_norm,
                                   bool have_reading, const Eigen::Vector3f& up_cam,
                                   const GravityGate& gate, float* tilt_deg = nullptr) {
    if (tilt_deg) *tilt_deg = -1.0f;
    if (!have_ref || !have_reading || gate.max_tilt_deg <= 0.0f) return GravityVerdict::Unknown;
    const float tilt = tiltErrorDeg(pose, up_world, up_cam);
    if (tilt_deg) *tilt_deg = tilt;
    if (!accelSteady(up_cam, ref_norm, gate.max_norm_dev)) return GravityVerdict::Unsteady;
    return tilt <= gate.max_tilt_deg ? GravityVerdict::Agree : GravityVerdict::Disagree;
}

// Everything the gravity checks need for one frame, in one place: the world-up
// reference (from the first frame with a reading) and this frame's reading.
struct GravityContext {
    bool            have_ref = false;
    Eigen::Vector3f up_world{0.0f, -1.0f, 0.0f};
    float           ref_norm = 0.0f;
    bool            have_reading = false;
    Eigen::Vector3f up_cam = Eigen::Vector3f::Zero();
    GravityGate     gate;

    GravityVerdict judge(const Eigen::Matrix4f& pose, float* tilt_deg = nullptr) const {
        return judgeGravity(pose, have_ref, up_world, ref_norm, have_reading, up_cam, gate, tilt_deg);
    }
    bool steady() const {
        return have_ref && have_reading && accelSteady(up_cam, ref_norm, gate.max_norm_dev);
    }
    // Candidate poses may be rotated onto the measured gravity.
    bool canSnap() const { return gate.max_tilt_deg > 0.0f && steady(); }
    // A reading exists but the hand is accelerating: no pose can be vouched for.
    bool blocksEveryPose() const {
        return gate.max_tilt_deg > 0.0f && have_ref && have_reading && !steady();
    }
    // While lost, a pose may not contradict gravity, and a reading bent by an
    // accelerating hand cannot vouch for one. Without any reading (no
    // accelerometer, synthetic input) there is nothing to check.
    static bool allowsReacquire(GravityVerdict v) {
        return v == GravityVerdict::Agree || v == GravityVerdict::Unknown;
    }
};

} // namespace tracking
} // namespace kfusion
