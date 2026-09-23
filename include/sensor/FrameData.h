#pragma once

#include <vector>
#include <cstdint>
#include <Eigen/Core>

#include "sensor/CameraIntrinsics.h"

namespace kfusion {
namespace sensor {

static constexpr int FRAME_W = 640;
static constexpr int FRAME_H = 480;

// Legacy / uncalibrated intrinsics (sensor/CameraIntrinsics.h). The pipeline
// uses the device's calibrated intrinsics when it has them; these constants are
// the fallback and what the synthetic tests render with.
static constexpr float FX = kLegacyIntrinsics.fx;
static constexpr float FY = kLegacyIntrinsics.fy;
static constexpr float CX = kLegacyIntrinsics.cx;
static constexpr float CY = kLegacyIntrinsics.cy;

// Processed depth frame: per-pixel 3D vertex + normal + valid flag
struct FrameData {
    // vertices[y*W+x] = 3D point in camera space (meters). (0,0,0) = invalid
    std::vector<Eigen::Vector3f> vertices;
    // normals[y*W+x] = surface normal in camera space. (0,0,0) = invalid
    std::vector<Eigen::Vector3f> normals;
    // rgb[y*W+x*3 + c] = RGB bytes
    std::vector<uint8_t> rgb;
    // depth in meters per pixel
    std::vector<float> depth_meters;

    int width  = FRAME_W;
    int height = FRAME_H;
    uint64_t frame_id = 0;
    // false when the sensor published this frame depth-only (no RGB sample
    // within the pairing window); `rgb` then holds stale bytes and must not be
    // fused into the volume.
    bool rgb_valid = true;
    // Sensor time of the depth frame, ms on the Kinect clock (RawFrame::timestamp_depth).
    double timestamp_ms = 0.0;
    // Intrinsics the vertices were back-projected with (set by buildFrameData).
    CameraIntrinsics intrinsics{};
    
    // The world-from-camera transformation found by the tracker for this specific frame
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();

    FrameData() {
        vertices.resize(FRAME_W * FRAME_H, Eigen::Vector3f::Zero());
        normals.resize(FRAME_W * FRAME_H, Eigen::Vector3f::Zero());
        rgb.resize(FRAME_W * FRAME_H * 3, 0);
        depth_meters.resize(FRAME_W * FRAME_H, 0.0f);
    }

    inline bool isValid(int x, int y) const {
        return depth_meters[y * width + x] > 0.0f;
    }
};

// Pyramid of FrameData (downsampled) used for multi-resolution ICP
struct FramePyramid {
    static constexpr int LEVELS = 3;
    FrameData levels[LEVELS]; // levels[0] = full res, levels[2] = coarsest

    FramePyramid() {
        levels[0].width = FRAME_W;    levels[0].height = FRAME_H;
        levels[1].width = FRAME_W/2;  levels[1].height = FRAME_H/2;
        levels[2].width = FRAME_W/4;  levels[2].height = FRAME_H/4;
        for (int l = 0; l < LEVELS; ++l) {
            int n = levels[l].width * levels[l].height;
            levels[l].vertices.assign(n, Eigen::Vector3f::Zero());
            levels[l].normals.assign(n, Eigen::Vector3f::Zero());
            levels[l].rgb.assign(n * 3, 0);
            levels[l].depth_meters.assign(n, 0.0f);
        }
    }
};

// Build FrameData from raw depth + rgb, applying intrinsics
void buildFrameData(const uint16_t* raw_depth,
                    const uint8_t*  raw_rgb,
                    FrameData& out,
                    float min_depth = 0.3f,
                    float max_depth = 5.0f,
                    const CameraIntrinsics& intrinsics = CameraIntrinsics{});

// Build pyramid by successive 2x downsampling
void buildFramePyramid(const FrameData& full_res, FramePyramid& pyramid);

// Compute normals from vertex map via cross-product of neighbors
void computeNormals(FrameData& frame);

} // namespace sensor
} // namespace kfusion
