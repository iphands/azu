#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace kfusion {
namespace rendering {

/**
 * Right-handed view basis shared by every camera surface (big-fix todo 27).
 *
 * `forward` points from the eye towards the orbit target (view-space -Z),
 * `right` is view-space +X and `up` is view-space +Y. `roll_` rotates up/right
 * about forward, so the basis is the SAME one viewMatrix() rows are built from;
 * panning and free-flight translation that go through it therefore stay
 * screen-aligned no matter how the camera is rolled.
 */
struct CameraBasis {
    Eigen::Vector3f forward{0.0f, 0.0f, -1.0f};
    Eigen::Vector3f right{1.0f, 0.0f, 0.0f};
    Eigen::Vector3f up{0.0f, 1.0f, 0.0f};
};

/**
 * First-frame clamp for the free-flight integration tick (big-fix todo 27).
 *
 * The physics QTimer starts in the widget constructor, so its very first
 * timeout is measured against construction time (widget setup, first resize,
 * first paint) rather than against a previous frame. Feeding that interval to
 * the movement integrator teleports the camera on the first tick. The first
 * frame is therefore replaced by one nominal physics tick; every later frame
 * uses the real elapsed time. A pure free function so the policy is testable
 * without a display, an OpenGL context or a wall clock.
 *
 * @param elapsed_seconds real seconds since the previous tick (<= 0 or
 *                        non-finite input yields 0.0f, i.e. no movement)
 * @param first_frame     true while processing the very first tick
 * @param tick_seconds    nominal tick used for the first frame
 */
[[nodiscard]] float clampFrameDeltaSeconds(double elapsed_seconds,
                                          bool first_frame,
                                          float tick_seconds);

class Camera {
public:
    enum class Mode {
        Orbit,
        Free
    };

    Camera();

    void rotate(float dx, float dy);
    void zoom(float delta);
    void pan(float dx, float dy);
    void move(const Eigen::Vector3f& dir, float amount);
    void reset();

    void setMode(Mode mode);
    Mode mode() const { return mode_; }

    void setAzimuth(float az) { yaw_ = az; }
    void setElevation(float el) { pitch_ = el; }
    void setRoll(float r) { roll_ = r; }

    float azimuth() const { return yaw_; }
    float elevation() const { return pitch_; }
    float roll() const { return roll_; }

    void setTarget(const Eigen::Vector3f& t) { target_ = t; }
    Eigen::Vector3f target() const { return target_; }

    /** The one view-basis helper: roll-aware forward/right/up (todo 27). */
    CameraBasis cameraBasis() const;

    /**
     * The one orbit eye helper: `target - distance * forward`.
     *
     * Because forward carries `+sin(pitch)`, the eye offset carries
     * `-distance * sin(pitch)`: the negative-Y orbit offset is a consequence of
     * the definition instead of a sign written at each call site. viewMatrix(),
     * the Orbit->Free transition and the Free->Orbit transition all read this,
     * so switching modes cannot jump the view.
     */
    Eigen::Vector3f orbitEye() const;

    Eigen::Matrix4f viewMatrix() const;
    Eigen::Matrix4f projectionMatrix(float aspect, float near_z = 0.01f, float far_z = 100.0f) const;

    float fov_degrees = 60.0f;

private:
    Mode mode_ = Mode::Orbit;

    // Rotation (radians)
    float yaw_   = 0.0f;
    float pitch_ = 0.3f;
    float roll_  = 0.0f;

    // Orbit-specific
    float distance_  = 3.0f;
    Eigen::Vector3f target_{0.0f, 0.0f, 1.0f};

    // Free-specific
    Eigen::Vector3f position_{0.0f, 0.0f, 0.0f};

    void updateFreeFromOrbit();
    void updateOrbitFromFree();
};

} // namespace rendering
} // namespace kfusion
