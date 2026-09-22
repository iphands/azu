// camera_basis_contract (big-fix todo 27): CPU-only, display-free contract for
// the orbit/pan/free-flight camera math, the roll-aware shared basis, the
// single orbit-eye convention and the first-frame tick clamp.
//
// Public kfusion::rendering::Camera API plus the clampFrameDeltaSeconds() free
// function only. Every expected number is hand-derived from the definition of
// the orbit eye (eye = target - distance * forward, with
// forward = (-sin(y)cos(p), sin(p), -cos(y)cos(p))) or from an orthonormality /
// on-axis-target invariant, never snapshotted from product output. The
// invariants are chosen so the two pre-fix defects cannot pass:
//   * a "+distance*sin(pitch)" orbit eye breaks normalize(target-eye)==forward,
//     the negative-Y sign check and the on-axis target projection;
//   * a roll-free pan/move basis lands a rolled camera's pan on the wrong axis.
// No device, display, GPU, OpenGL context, sensor, thread timing, wall clock or
// filesystem access.

#include "rendering/Camera.h"

#include <Eigen/Core>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "camera_basis_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

using kfusion::rendering::Camera;
using kfusion::rendering::CameraBasis;
// Called through the real namespace: an unqualified local replica would
// not be the product's tick policy.
namespace cam = kfusion::rendering;

constexpr float kHalfPi = 1.57079632679489661923f;
constexpr float kPi     = 3.14159265358979323846f;

// reset() leaves distance_ at exactly 2.0f, so the pan oracles below evaluate
// to 0.005f * 2.0f * 100 px == 1.0 world unit per axis.
constexpr float kDistance = 2.0f;
constexpr float kPanPx    = 100.0f;
constexpr float kTol      = 1.0e-5f;
constexpr float kTightTol = 1.0e-6f;

int g_failures = 0;
int g_checks   = 0;

bool nearF(float a, float b, float tol) {
    return a == b || std::fabs(a - b) <= tol;
}

bool nearV(const Eigen::Vector3f& v, float x, float y, float z, float tol) {
    return nearF(v.x(), x, tol) && nearF(v.y(), y, tol) && nearF(v.z(), z, tol);
}

bool nearM(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b, float tol) {
    return (a - b).cwiseAbs().maxCoeff() <= tol;
}

#define CHECK(cond, what)                                                        \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            std::printf("FAIL: %s  [%s:%d]\n", std::string(what).c_str(),        \
                        __FILE__, __LINE__);                                     \
            ++g_failures;                                                        \
        }                                                                        \
    } while (false)

// NOTE: the body must not spell .x()/.y()/.z(), those tokens are the parameter
// names; indexed access keeps the macro hygienic.
#define CHECK_VEC(vec, x, y, z, what)                                            \
    do {                                                                         \
        ++g_checks;                                                              \
        const Eigen::Vector3f v_ = (vec);                                        \
        if (!nearV(v_, (x), (y), (z), kTol)) {                                   \
            std::printf("FAIL: %s: got (% .6f, % .6f, % .6f) want (% .6f, "      \
                        "% .6f, % .6f)  [%s:%d]\n", std::string(what).c_str(),    \
                        v_(0), v_(1), v_(2), float(x), float(y), float(z),        \
                        __FILE__, __LINE__);                                     \
            ++g_failures;                                                        \
        }                                                                        \
    } while (false)

#define CHECK_NEAR(a, b, tol, what)                                              \
    do {                                                                         \
        ++g_checks;                                                              \
        const float got_ = (a), want_ = (b), tol_ = (tol);                        \
        if (!(got_ == want_) && !(std::fabs(got_ - want_) <= tol_)) {             \
            std::printf("FAIL: %s: got %.9f want %.9f (tol %.1e)  [%s:%d]\n",     \
                        std::string(what).c_str(), got_, want_, tol_,             \
                        __FILE__, __LINE__);                                     \
            ++g_failures;                                                        \
        }                                                                        \
    } while (false)

// eye recovered from a view matrix: t = -R * eye and R is orthonormal, so
// eye = -R^T * t. Independent of the product's own eye computation.
Eigen::Vector3f eyeFromView(const Eigen::Matrix4f& V) {
    const Eigen::Vector3f t = V.block<3, 1>(0, 3);
    const Eigen::Matrix3f R = V.block<3, 3>(0, 0);
    return -R.transpose() * t;
}

// Orbit-mode camera at the given pose. reset() is the only way to reach a known
// distance (2.0f) and the Free->Orbit transition rewrites target_, so the pose
// is applied after the mode switch.
Camera orbitCam(float yaw, float pitch, float roll, const Eigen::Vector3f& target) {
    Camera c;
    c.setMode(Camera::Mode::Orbit);
    c.setAzimuth(yaw);
    c.setElevation(pitch);
    c.setRoll(roll);
    c.setTarget(target);
    return c;
}

// Free-mode camera at the given pose; eye == position_ == (0, 0, -1).
Camera freeCam(float yaw, float pitch, float roll) {
    Camera c;  // reset(): Mode::Free, yaw/pitch/roll 0, position (0, 0, -1)
    c.setAzimuth(yaw);
    c.setElevation(pitch);
    c.setRoll(roll);
    return c;
}

// The invariant every correct orbit eye must satisfy.
void checkEyeInvariants(const Camera& c, const char* what) {
    const CameraBasis b = c.cameraBasis();
    const Eigen::Vector3f eye = c.orbitEye();
    CHECK_VEC((c.target() - eye).normalized(), b.forward.x(), b.forward.y(),
              b.forward.z(), what);
    CHECK_NEAR((eye - c.target()).norm(), kDistance, kTightTol,
               "orbit eye stays on the orbit sphere");
}

// Independent hand-derived basis + eye + view matrix for one pose.
void checkPose(const char* what, float yaw, float pitch, float roll) {
    const Eigen::Vector3f target(0.0f, 0.0f, 0.0f);
    const Camera c = orbitCam(yaw, pitch, roll, target);
    const CameraBasis b = c.cameraBasis();

    const float cy = std::cos(yaw),   sy = std::sin(yaw);
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const Eigen::Vector3f fwd(-sy * cp, sp, -cy * cp);
    CHECK_VEC(b.forward, fwd.x(), fwd.y(), fwd.z(), "forward from the trig definition");

    const Eigen::Vector3f eye = target - kDistance * fwd;
    CHECK_VEC(c.orbitEye(), eye.x(), eye.y(), eye.z(), "orbitEye == target - d*forward");
    CHECK_NEAR(c.orbitEye().y(), -kDistance * sp, kTightTol,
               "orbit eye Y carries -distance*sin(pitch)");

    const Eigen::Matrix4f V = c.viewMatrix();
    CHECK_VEC(V.row(0).head<3>().transpose(), b.right.x(), b.right.y(), b.right.z(),
              "view row0 == basis right");
    CHECK_VEC(V.row(1).head<3>().transpose(), b.up.x(), b.up.y(), b.up.z(),
              "view row1 == basis up");
    CHECK_VEC(V.row(2).head<3>().transpose(), -b.forward.x(), -b.forward.y(),
              -b.forward.z(), "view row2 == -basis forward");
    CHECK_VEC(eyeFromView(V), eye.x(), eye.y(), eye.z(),
              "view translation encodes orbitEye()");
    CHECK_NEAR(V(3, 0), 0.0f, 0.0f, "view row3 is affine");
    CHECK_NEAR(V(3, 1), 0.0f, 0.0f, "view row3 is affine");
    CHECK_NEAR(V(3, 2), 0.0f, 0.0f, "view row3 is affine");
    CHECK_NEAR(V(3, 3), 1.0f, 0.0f, "view row3 is affine");

    const Eigen::Matrix3f R = V.block<3, 3>(0, 0);
    CHECK_NEAR((R * R.transpose() - Eigen::Matrix3f::Identity()).cwiseAbs().maxCoeff(),
               0.0f, kTol, "view rotation is orthonormal");
    CHECK_NEAR(R.determinant(), 1.0f, kTol, "view rotation is det +1 (no mirror)");
    CHECK_VEC(R.row(0).cross(R.row(1)).transpose(), R.row(2).x(), R.row(2).y(),
              R.row(2).z(), "row0 x row1 == row2");

    // The orbit target must land dead-centre: view-space (0, 0, -distance).
    const Eigen::Vector3f tv = R * target + V.block<3, 1>(0, 3);
    CHECK_VEC(tv, 0.0f, 0.0f, -kDistance, "orbit target projects on-axis");
    checkEyeInvariants(c, what);
}

// --- 1. basis / eye / view matrix at hand-chosen poses -----------------------

void sectionBasisAndEye() {
    // yaw=pitch=roll=0 is exactly representable, so these are literal values.
    const Camera c0 = orbitCam(0.0f, 0.0f, 0.0f, Eigen::Vector3f(0.0f, 0.0f, 1.0f));
    const CameraBasis b0 = c0.cameraBasis();
    CHECK_VEC(b0.forward, 0.0f, 0.0f, -1.0f, "identity forward");
    CHECK_VEC(b0.right,   1.0f, 0.0f,  0.0f, "identity right");
    CHECK_VEC(b0.up,      0.0f, 1.0f,  0.0f, "identity up");
    CHECK_VEC(c0.orbitEye(), 0.0f, 0.0f, 3.0f, "identity eye at target - distance*forward");

    checkPose("generic pose",   0.4f, 0.6f,  0.25f);
    checkPose("negative yaw",  -0.4f, 0.6f,  0.25f);
    checkPose("negative pitch", 0.4f, -0.6f, 0.25f);
    checkPose("negative roll",  0.4f, 0.6f, -0.25f);
    checkPose("quarter-yaw pose", kHalfPi, 0.3f, 0.0f);
    checkPose("half-turn pose", kPi, 0.2f, 0.1f);
}

// --- 2. the negative-Y orbit convention, explicitly --------------------------

void sectionNegativeY() {
    const float pitch = 0.6f;
    const Camera c = orbitCam(0.0f, pitch, 0.0f, Eigen::Vector3f(0.0f, 0.0f, 0.0f));
    const Eigen::Vector3f eye = c.orbitEye();

    CHECK(eye.y() < 0.0f, "positive pitch moves the orbit eye to NEGATIVE Y");
    CHECK_NEAR(eye.y(), -kDistance * std::sin(pitch), kTightTol,
               "orbit eye Y == -distance*sin(pitch)");
    CHECK_NEAR(eye.z(), kDistance * std::cos(pitch), kTightTol,
               "orbit eye Z == +distance*cos(pitch)");

    // The mirrored convention is rejected: it does not look at the target.
    const Eigen::Vector3f wrong_eye(0.0f, kDistance * std::sin(pitch),
                                    kDistance * std::cos(pitch));
    const Eigen::Vector3f wrong_dir = (Eigen::Vector3f(0.0f, 0.0f, 0.0f) - wrong_eye).normalized();
    CHECK(!nearV(wrong_dir, c.cameraBasis().forward.x(), c.cameraBasis().forward.y(),
                 c.cameraBasis().forward.z(), kTol),
          "a +sin(pitch) orbit eye does NOT look at the target");

    // Same convention at every yaw octant, so no path can re-flip it.
    const float ys[4] = {0.0f, kHalfPi, kPi, -kHalfPi};
    for (float yaw : ys) {
        const Camera cc = orbitCam(yaw, 0.9f, 0.0f, Eigen::Vector3f(1.0f, -2.0f, 0.5f));
        CHECK(cc.orbitEye().y() < cc.target().y(),
              "the eye stays below the target for positive pitch at every yaw");
        checkEyeInvariants(cc, "yaw-octant eye invariant");
    }
}

// --- 3. roll enters the shared basis used by pan -----------------------------

void sectionRollAwarePan() {
    const Eigen::Vector3f origin(0.0f, 0.0f, 0.0f);

    {
        Camera c = orbitCam(0.0f, 0.0f, 0.0f, origin);
        c.pan(kPanPx, kPanPx);
        CHECK_VEC(c.target(), -1.0f, -1.0f, 0.0f, "pan at roll 0 moves -X -Y");
    }
    // Rolled +90 deg about forward(-Z): up -> +X, right -> -Y. The SAME drag now
    // moves the target by (-1, +1); a roll-free basis would repeat (-1, -1).
    {
        Camera c = orbitCam(0.0f, 0.0f, kHalfPi, origin);
        const CameraBasis b = c.cameraBasis();
        CHECK_VEC(b.right,   0.0f, -1.0f,  0.0f, "roll +90 right == -Y");
        CHECK_VEC(b.up,      1.0f,  0.0f,  0.0f, "roll +90 up == +X");
        CHECK_VEC(b.forward, 0.0f,  0.0f, -1.0f, "roll leaves forward alone");
        c.pan(kPanPx, kPanPx);
        CHECK_VEC(c.target(), -1.0f, 1.0f, 0.0f, "pan at roll +90 moves -X +Y");
    }
    // Rolled -90 deg flips right/up the other way: right = +Y, up = -X, so the
    // same drag is exactly the negation of the +90 deg result.
    {
        Camera c = orbitCam(0.0f, 0.0f, -kHalfPi, origin);
        const CameraBasis b = c.cameraBasis();
        CHECK_VEC(b.right,  0.0f, 1.0f, 0.0f, "roll -90 right == +Y");
        CHECK_VEC(b.up,    -1.0f, 0.0f, 0.0f, "roll -90 up == -X");
        c.pan(kPanPx, kPanPx);
        // right * (-1) + up * (-1) == (0,-1,0) + (1,0,0)
        CHECK_VEC(c.target(), 1.0f, -1.0f, 0.0f, "pan at roll -90 moves +X -Y");
    }
    // Pitched, unrolled: up == (0, cos p, sin p) and a vertical pan follows it.
    {
        const float p = 0.6f;
        Camera c = orbitCam(0.0f, p, 0.0f, origin);
        CHECK_VEC(c.cameraBasis().up, 0.0f, std::cos(p), std::sin(p), "pitched up vector");
        c.pan(0.0f, kPanPx);
        CHECK_VEC(c.target(), 0.0f, -std::cos(p), -std::sin(p),
                  "vertical pan follows the pitched up vector");
    }
    // Yawed: right == (cos y, 0, -sin y), so a half turn flips pan to +X.
    {
        Camera c = orbitCam(kPi, 0.0f, 0.0f, origin);
        CHECK_VEC(c.cameraBasis().right, -1.0f, 0.0f, 0.0f, "half-turn right == -X");
        c.pan(kPanPx, 0.0f);
        CHECK_VEC(c.target(), 1.0f, 0.0f, 0.0f, "pan direction flips with yaw");
    }
    // Pan scales with orbit distance: 10 wheel steps out (0.3 per step) takes
    // the distance from 2 to 5, so the same drag moves 2.5 units.
    {
        Camera c = orbitCam(0.0f, 0.0f, 0.0f, origin);
        CHECK_NEAR((c.orbitEye() - origin).norm(), kDistance, kTightTol,
                   "reset distance is 2");
        c.zoom(-10.0f);
        CHECK_NEAR((c.orbitEye() - origin).norm(), 5.0f, kTol,
                   "zoom(-10) takes the distance to 5");
        c.pan(kPanPx, 0.0f);
        CHECK_VEC(c.target(), -2.5f, 0.0f, 0.0f, "pan scales with orbit distance");
    }
}

// --- 4. one orbit eye: switching modes cannot jump the view ------------------

void sectionModeSwitchContinuity() {
    const Camera base = orbitCam(0.4f, 0.6f, 0.25f, Eigen::Vector3f(0.5f, 0.25f, 1.0f));
    const Eigen::Matrix4f v_orbit = base.viewMatrix();
    const Eigen::Vector3f eye_orbit = base.orbitEye();

    Camera c = base;
    c.setMode(Camera::Mode::Free);
    const Eigen::Matrix4f v_free = c.viewMatrix();
    CHECK(nearM(v_orbit, v_free, kTightTol),
          "Orbit -> Free keeps the identical view (both use orbitEye())");
    CHECK_VEC(eyeFromView(v_free), eye_orbit.x(), eye_orbit.y(), eye_orbit.z(),
              "the Free eye equals the orbit eye");

    c.setMode(Camera::Mode::Orbit);
    CHECK(nearM(v_orbit, c.viewMatrix(), kTol),
          "Free -> Orbit round-trips back to the same view");
    CHECK_VEC(c.orbitEye(), eye_orbit.x(), eye_orbit.y(), eye_orbit.z(),
              "the orbit eye round-trips through Free");
}

// --- 5. free-flight WASD/QE basis: roll-aware, signs preserved ---------------

void sectionFreeFlight() {
    {
        Camera c = freeCam(0.0f, 0.0f, 0.0f);
        c.move(Eigen::Vector3f(0.0f, 0.0f, 1.0f), 0.5f);  // W
        CHECK_VEC(eyeFromView(c.viewMatrix()), 0.0f, 0.0f, -1.5f,
                  "W advances along forward");
    }
    {
        Camera c = freeCam(0.0f, 0.0f, 0.0f);
        c.move(Eigen::Vector3f(1.0f, 0.0f, 0.0f), 0.5f);  // D
        CHECK_VEC(eyeFromView(c.viewMatrix()), 0.5f, 0.0f, -1.0f,
                  "D strafes along right");
    }
    {
        Camera c = freeCam(0.0f, 0.0f, 0.0f);
        c.move(Eigen::Vector3f(0.0f, 1.0f, 0.0f), 0.5f);  // E
        CHECK_VEC(eyeFromView(c.viewMatrix()), 0.0f, 0.5f, -1.0f, "E rises along up");
    }
    {
        Camera c = freeCam(0.0f, 0.0f, 0.0f);
        c.move(Eigen::Vector3f(0.0f, -1.0f, 0.0f), 0.5f); // Q
        CHECK_VEC(eyeFromView(c.viewMatrix()), 0.0f, -0.5f, -1.0f, "Q drops along -up");
    }
    {
        // A quarter turn of yaw makes forward -X, so W now travels -X.
        Camera c = freeCam(kHalfPi, 0.0f, 0.0f);
        c.move(Eigen::Vector3f(0.0f, 0.0f, 1.0f), 0.5f);
        CHECK_VEC(eyeFromView(c.viewMatrix()), -0.5f, 0.0f, -1.0f,
                  "W follows the yawed forward");
    }
    {
        // Rolled 90 deg: up is +X, so E rises along +X instead of +Y.
        Camera c = freeCam(0.0f, 0.0f, kHalfPi);
        c.move(Eigen::Vector3f(0.0f, 1.0f, 0.0f), 0.5f);
        CHECK_VEC(eyeFromView(c.viewMatrix()), 0.5f, 0.0f, -1.0f,
                  "E follows the roll-aware up");
    }
    {
        // Free-mode pan keeps its signs (0.005 per px, unscaled by distance).
        Camera c = freeCam(0.0f, 0.0f, 0.0f);
        c.pan(kPanPx, kPanPx);
        CHECK_VEC(eyeFromView(c.viewMatrix()), -0.5f, 0.5f, -1.0f, "free pan signs");
    }
    {
        // Orbit-mode move() translates the TARGET, and the eye follows it.
        Camera c = orbitCam(0.0f, 0.0f, 0.0f, Eigen::Vector3f(0.0f, 0.0f, 1.0f));
        c.move(Eigen::Vector3f(0.0f, 0.0f, 1.0f), 0.5f);
        CHECK_VEC(c.target(),  0.0f, 0.0f, 0.5f, "orbit move() translates the target");
        CHECK_VEC(c.orbitEye(), 0.0f, 0.0f, 2.5f, "the orbit eye follows the target");
    }
}

// --- 6. rotate() sensitivities and the pitch clamp ---------------------------

void sectionRotate() {
    Camera o = orbitCam(0.0f, 0.0f, 0.0f, Eigen::Vector3f(0.0f, 0.0f, 0.0f));
    o.rotate(kPanPx, kPanPx);
    CHECK_NEAR(o.azimuth(),   -1.0f, kTightTol, "orbit drag is 0.01 rad per px");
    CHECK_NEAR(o.elevation(), -1.0f, kTightTol, "orbit pitch uses the same scale");

    Camera f = freeCam(0.0f, 0.0f, 0.0f);
    f.rotate(kPanPx, kPanPx);
    CHECK_NEAR(f.azimuth(),   -0.5f, kTightTol, "free-look is half the orbit rate");
    CHECK_NEAR(f.elevation(), -0.5f, kTightTol, "free-look pitch is halved too");

    Camera clamp_hi = orbitCam(0.0f, 0.0f, 0.0f, Eigen::Vector3f::Zero());
    clamp_hi.rotate(0.0f, -1.0e6f);
    CHECK(clamp_hi.elevation() == 1.5f, "pitch clamps at +1.5 rad");
    Camera clamp_lo = orbitCam(0.0f, 0.0f, 0.0f, Eigen::Vector3f::Zero());
    clamp_lo.rotate(0.0f, 1.0e6f);
    CHECK(clamp_lo.elevation() == -1.5f, "pitch clamps at -1.5 rad");

    // A clamped pose still satisfies the negative-Y eye convention.
    const Eigen::Matrix4f V = clamp_hi.viewMatrix();
    CHECK_VEC(eyeFromView(V), 0.0f, -kDistance * std::sin(1.5f),
              kDistance * std::cos(1.5f), "clamped pitch keeps the negative-Y eye");
    CHECK(std::isfinite(V(1, 3)) && std::isfinite(V(2, 3)), "clamped view stays finite");
}

// --- 7. first-frame dt clamp (pure function, no wall clock) ------------------

void sectionFirstFrameClamp() {
    constexpr float kTick = 0.016f;
    const float kNaN = std::numeric_limits<float>::quiet_NaN();
    const double kInf = std::numeric_limits<double>::infinity();

    // First frame: one nominal tick, whatever the elapsed time was.
    CHECK(cam::clampFrameDeltaSeconds(4.21875, true, kTick) == kTick,
          "first frame yields exactly one nominal tick");
    CHECK(cam::clampFrameDeltaSeconds(0.0, true, kTick) == kTick,
          "the first-frame clamp ignores a zero elapsed");
    CHECK(cam::clampFrameDeltaSeconds(1.0e9, true, kTick) == kTick,
          "the first-frame clamp ignores an absurd elapsed (no teleport)");
    CHECK(cam::clampFrameDeltaSeconds(kNaN, true, kTick) == kTick,
          "the first-frame clamp ignores NaN");
    CHECK(cam::clampFrameDeltaSeconds(4.21875, true, 0.01f) == 0.01f,
          "the clamp follows the tick it is given");

    // Later frames: the real elapsed time, uncapped.
    CHECK(cam::clampFrameDeltaSeconds(4.21875, false, kTick) == 4.21875f,
          "later frames keep the real elapsed seconds");
    CHECK(cam::clampFrameDeltaSeconds(10.0, false, kTick) == 10.0f,
          "later frames are NOT capped by the tick");
    CHECK(cam::clampFrameDeltaSeconds(1.0 / 3.0, false, kTick) == static_cast<float>(1.0 / 3.0),
          "later frames pass through the exact float conversion");
    CHECK(cam::clampFrameDeltaSeconds(1.0 / 60.0, false, kTick) != kTick,
          "a real 1/60 s tick stays distinguishable from the nominal one");

    // Degenerate/non-monotonic clocks move nobody.
    CHECK(cam::clampFrameDeltaSeconds(0.0, false, kTick) == 0.0f, "zero elapsed -> zero step");
    CHECK(cam::clampFrameDeltaSeconds(-3.0, false, kTick) == 0.0f, "negative elapsed -> zero step");
    CHECK(cam::clampFrameDeltaSeconds(kNaN, false, kTick) == 0.0f, "NaN elapsed -> zero step");
    CHECK(cam::clampFrameDeltaSeconds(kInf, false, kTick) == 0.0f, "infinite elapsed -> zero step");

    // Observable equivalent: at the widget's 2 m/s speed one nominal tick moves
    // the free-flight eye by exactly 0.032 m, not by a construction-time jump.
    Camera c = freeCam(0.0f, 0.0f, 0.0f);
    const float dt = cam::clampFrameDeltaSeconds(419.4211, true, kTick);
    c.move(Eigen::Vector3f(0.0f, 0.0f, 1.0f), 2.0f * dt);
    CHECK_NEAR(std::fabs(eyeFromView(c.viewMatrix()).z() + 1.0f), 2.0f * kTick, kTightTol,
               "the clamped first frame integrates one tick of travel");
}

} // namespace

int main() {
    sectionBasisAndEye();
    sectionNegativeY();
    sectionRollAwarePan();
    sectionModeSwitchContinuity();
    sectionFreeFlight();
    sectionRotate();
    sectionFirstFrameClamp();

    std::printf("camera_basis_contract: %d checks, %d failures\n", g_checks, g_failures);
    if (g_failures != 0) {
        std::printf("camera_basis_contract: FAIL\n");
        return 1;
    }
    std::printf("camera_basis_contract: PASS\n");
    return 0;
}
