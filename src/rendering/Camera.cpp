#include "rendering/Camera.h"
#include <cmath>
#include <algorithm>

namespace kfusion {
namespace rendering {

namespace {

// Every input surface (mouse drag, gizmo drag, WASD tick, wheel) reads the same
// named constants, so no call site carries its own copy of a factor.
constexpr float kOrbitAngularSensitivity = 0.01f;   // rad / px, left-drag orbit
constexpr float kFreeAngularSensitivity  = 0.005f;  // rad / px, free-look
constexpr float kPitchLimitRad           = 1.5f;    // zenith guard (no asin/gimbal)
constexpr float kPanPixelsToDistance     = 0.005f;  // world units / px at unit distance
constexpr float kZoomPerPixel            = 0.3f;
constexpr float kMinDistance             = 0.1f;
constexpr float kMaxDistance             = 100.0f;

} // namespace

float clampFrameDeltaSeconds(double elapsed_seconds, bool first_frame, float tick_seconds) {
    if (first_frame) {
        // Construction-time elapsed is not a frame interval; see the header note.
        return tick_seconds;
    }
    if (!std::isfinite(elapsed_seconds) || elapsed_seconds <= 0.0) {
        return 0.0f;  // monotonic clock can only stall; never move backwards
    }
    return static_cast<float>(elapsed_seconds);
}

Camera::Camera() {
    reset();
}

CameraBasis Camera::cameraBasis() const {
    const float cos_y = std::cos(yaw_);
    const float sin_y = std::sin(yaw_);
    const float cos_p = std::cos(pitch_);
    const float sin_p = std::sin(pitch_);

    CameraBasis b;
    // forward is unit by construction: sin^2(y)cos^2(p) + sin^2(p) + cos^2(y)cos^2(p) = 1
    b.forward = Eigen::Vector3f(-sin_y * cos_p, sin_p, -cos_y * cos_p);
    b.right   = Eigen::Vector3f(cos_y, 0.0f, -sin_y);
    b.up      = b.right.cross(b.forward).normalized();

    if (roll_ != 0.0f) {
        // Roll spins up/right about the (unchanged) forward axis, then right is
        // re-derived from the crossed pair so the basis stays orthonormal.
        b.up    = Eigen::AngleAxisf(roll_, b.forward) * b.up;
        b.right = b.forward.cross(b.up).normalized();
    }
    return b;
}

Eigen::Vector3f Camera::orbitEye() const {
    // The single orbit-eye definition. forward.y() == +sin(pitch), so the eye
    // offset is -distance*sin(pitch): one negative-Y convention everywhere.
    return target_ - distance_ * cameraBasis().forward;
}

void Camera::rotate(float dx, float dy) {
    const float sensitivity = (mode_ == Mode::Orbit) ? kOrbitAngularSensitivity
                                                     : kFreeAngularSensitivity;
    yaw_   -= dx * sensitivity;
    pitch_ -= dy * sensitivity;
    pitch_  = std::clamp(pitch_, -kPitchLimitRad, kPitchLimitRad);
}

void Camera::zoom(float delta) {
    if (mode_ == Mode::Orbit) {
        distance_ -= delta * kZoomPerPixel;
        distance_  = std::clamp(distance_, kMinDistance, kMaxDistance);
    } else {
        move(Eigen::Vector3f(0.0f, 0.0f, -1.0f), delta * kZoomPerPixel);
    }
}

void Camera::pan(float dx, float dy) {
    // One shared, roll-aware basis: panning stays aligned with what the user
    // actually sees even after the gizmo/sliders rolled the camera.
    const CameraBasis b = cameraBasis();

    if (mode_ == Mode::Orbit) {
        const float scale = kPanPixelsToDistance * distance_;
        target_ += b.right * (-dx * scale) + b.up * (-dy * scale);
    } else {
        position_ += b.right * (-dx * kPanPixelsToDistance) +
                     b.up    * ( dy * kPanPixelsToDistance);
    }
}

void Camera::move(const Eigen::Vector3f& dir, float amount) {
    const CameraBasis b = cameraBasis();
    const Eigen::Vector3f step = b.right   * (dir.x() * amount) +
                                 b.up      * (dir.y() * amount) +
                                 b.forward * (dir.z() * amount);

    if (mode_ == Mode::Free) {
        position_ += step;
    } else {
        // In Orbit mode, move() translates the target
        target_ += step;
    }
}

void Camera::setMode(Mode mode) {
    if (mode_ == mode) return;

    if (mode == Mode::Free) {
        updateFreeFromOrbit();
    } else {
        updateOrbitFromFree();
    }
    mode_ = mode;
}

void Camera::updateFreeFromOrbit() {
    // Same helper the view matrix uses: Orbit -> Free keeps the eye identical.
    position_ = orbitEye();
}

void Camera::updateOrbitFromFree() {
    // Inverse of orbitEye() along the same forward axis.
    target_ = position_ + distance_ * cameraBasis().forward;
}

void Camera::reset() {
    mode_ = Mode::Free; // Default to free for "flying around"
    yaw_       = 0.0f;
    pitch_     = 0.0f;
    roll_      = 0.0f;
    distance_  = 2.0f;
    target_    = Eigen::Vector3f(0.0f, 0.0f, 1.28f);
    position_  = Eigen::Vector3f(0.0f, 0.0f, -1.0f); // Place camera slightly back
}

Eigen::Matrix4f Camera::viewMatrix() const {
    const CameraBasis b = cameraBasis();
    // Both modes go through the same basis; the orbit eye comes from the single
    // orbitEye() helper so a mode switch cannot change the rendered view.
    const Eigen::Vector3f eye = (mode_ == Mode::Orbit) ? orbitEye() : position_;

    Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
    V(0,0) =  b.right.x();   V(0,1) =  b.right.y();   V(0,2) =  b.right.z();   V(0,3) = -b.right.dot(eye);
    V(1,0) =  b.up.x();      V(1,1) =  b.up.y();      V(1,2) =  b.up.z();      V(1,3) = -b.up.dot(eye);
    V(2,0) = -b.forward.x(); V(2,1) = -b.forward.y(); V(2,2) = -b.forward.z(); V(2,3) =  b.forward.dot(eye);
    return V;
}

Eigen::Matrix4f Camera::projectionMatrix(float aspect, float near_z, float far_z) const {
    float fov_rad  = fov_degrees * M_PI / 180.0f;
    float tan_half = std::tan(fov_rad * 0.5f);

    Eigen::Matrix4f P = Eigen::Matrix4f::Zero();
    P(0,0) = 1.0f / (aspect * tan_half);
    P(1,1) = 1.0f / tan_half;
    P(2,2) = -(far_z + near_z) / (far_z - near_z);
    P(2,3) = -(2.0f * far_z * near_z) / (far_z - near_z);
    P(3,2) = -1.0f;
    return P;
}

} // namespace rendering
} // namespace kfusion
