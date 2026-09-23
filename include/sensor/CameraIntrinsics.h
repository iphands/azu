#pragma once

// Pinhole intrinsics of the depth (IR) camera, in 640x480 pixels.
//
// The Kinect v1 reports its own IR camera geometry in the factory registration
// block (libfreenect freenect_copy_registration, recorded as device.json by
// fakenect-record): zero_plane_info.reference_distance (mm) and
// reference_pixel_size (mm at 1280 px). libfreenect's camera-to-world mapping
// uses f = reference_distance / (2 * reference_pixel_size); on the dev unit
// that is 120 / (2 * 0.1042) = 575.8 px. The historical 525 px is an RGB-camera
// value; applied to IR depth it scales the scene laterally by ~10%, which
// cancels under translation but bends the reconstruction under rotation.

namespace kfusion {
namespace sensor {

struct CameraIntrinsics {
    float fx = 525.0f;
    float fy = 525.0f;
    float cx = 319.5f;
    float cy = 239.5f;
};

// What everything used before calibrated intrinsics existed; still the default
// when no device calibration is available (synthetic data, tests).
inline constexpr CameraIntrinsics kLegacyIntrinsics{};

inline bool intrinsicsFromZeroPlane(double reference_distance_mm, double reference_pixel_size_mm,
                                    CameraIntrinsics* out) {
    if (!(reference_distance_mm > 0.0) || !(reference_pixel_size_mm > 0.0)) return false;
    const float f = static_cast<float>(reference_distance_mm / (2.0 * reference_pixel_size_mm));
    if (!(f > 100.0f && f < 2000.0f)) return false;
    *out = CameraIntrinsics{f, f, 319.5f, 239.5f};
    return true;
}

} // namespace sensor
} // namespace kfusion
