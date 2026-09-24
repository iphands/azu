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

// The same camera rendered at w x h instead of ref_w x ref_h: focal lengths scale
// with the size and pixel centres stay pixel centres ((c + 0.5) * s - 0.5).
// Returns k itself at the same size, so full-resolution callers are unchanged.
// The low-resolution model images of relocalization are raycast AND projected
// into with this one function; any mismatch shifts every correspondence.
inline CameraIntrinsics scaleIntrinsics(const CameraIntrinsics& k, int ref_w, int ref_h, int w,
                                        int h) {
    if (w == ref_w && h == ref_h) return k;
    const float sx = static_cast<float>(w) / static_cast<float>(ref_w);
    const float sy = static_cast<float>(h) / static_cast<float>(ref_h);
    return CameraIntrinsics{k.fx * sx, k.fy * sy, (k.cx + 0.5f) * sx - 0.5f,
                            (k.cy + 0.5f) * sy - 0.5f};
}

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
