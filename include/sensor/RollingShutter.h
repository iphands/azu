#pragma once

// Rolling-shutter unwarp of a raw Kinect v1 depth frame (opt-in experiment,
// AZU_RS_READOUT_MS=<readout ms>).
//
// The Kinect v1 IR sensor reads its rows top to bottom over most of a frame
// period, so under fast motion each row is seen from a slightly different pose:
// a pitch sweep stretches or squashes the frame vertically, a pan shears it.
// Rigid ICP cannot absorb that, and on a handheld room take (cap_001, pitch
// sweeps at 45-105 deg/s) the ICP RMS rose from ~4 mm to 10-20 mm with the
// pitch rate. Unwarping with a 20-33 ms readout brought the worst sweep
// segment from 15.1 to 8.7 mm mean RMS; a negative readout (bottom row first)
// made it worse and 60 ms overshot, which is what a real top-to-bottom rolling
// shutter predicts.
//
// Model: row v is captured at t(v) = (v / (H - 1) - 0.5) * readout relative to
// the frame's mid-row time, and the camera moves with a constant per-frame
// motion `step` (the previous-to-current pose increment, camera frame) over
// `frame_period`. A point seen at row v in that row's camera, x_v, is at
// exp(s * log(step)) * x_v in the mid-row camera, s = t(v) / frame_period.
//
// The output is resampled by inverse mapping (for each output pixel, find the
// source pixel whose warped point lands there; three fixed-point iterations on
// the source depth), so it has no splat holes. Invalid or out-of-band depth
// stays invalid.

#include "sensor/CameraIntrinsics.h"
#include "sensor/DepthValidity.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <cstdint>
#include <vector>

namespace kfusion {
namespace sensor {

inline void unwarpRollingShutter(const uint16_t* src, uint16_t* dst, int width, int height,
                                 const CameraIntrinsics& k, const Eigen::Matrix4f& step,
                                 float readout_s, float frame_period_s,
                                 float min_depth_m, float max_depth_m) {
    const Eigen::AngleAxisf aa(Eigen::Matrix3f(step.block<3,3>(0,0)));
    const Eigen::Vector3f t = step.block<3,1>(0,3);
    const float scale = readout_s / frame_period_s;
    // Per-row inverse warp: output (mid-row camera) -> source row camera.
    std::vector<Eigen::Matrix3f> R_inv(height);
    std::vector<Eigen::Vector3f> t_inv(height);
    for (int v = 0; v < height; ++v) {
        const float s = (static_cast<float>(v) / static_cast<float>(height - 1) - 0.5f) * scale;
        const Eigen::Matrix3f R = Eigen::AngleAxisf(aa.angle() * s, aa.axis()).toRotationMatrix();
        R_inv[v] = R.transpose();
        t_inv[v] = -(R.transpose() * (t * s));
    }
    auto depthAt = [&](int u, int v) -> float {
        if (u < 0 || v < 0 || u >= width || v >= height) return 0.0f;
        return cpuDepthMeters(src[static_cast<size_t>(v) * width + u], min_depth_m, max_depth_m);
    };
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            // Output ray; its depth is unknown, so iterate on the source depth.
            const float rx = (static_cast<float>(u) - k.cx) / k.fx;
            const float ry = (static_cast<float>(v) - k.cy) / k.fy;
            int su = u, sv = v;
            float z_out = depthAt(u, v);
            uint16_t out = 0;
            for (int it = 0; it < 3 && z_out > 0.0f; ++it) {
                const Eigen::Vector3f x_out(rx * z_out, ry * z_out, z_out);
                // The source row decides the warp; start from the output row.
                const Eigen::Vector3f x_src = R_inv[sv] * x_out + t_inv[sv];
                if (!(x_src.z() > 0.0f)) { z_out = 0.0f; break; }
                su = static_cast<int>(std::lround(k.fx * x_src.x() / x_src.z() + k.cx));
                sv = static_cast<int>(std::lround(k.fy * x_src.y() / x_src.z() + k.cy));
                const float z_src = depthAt(su, sv);
                if (z_src <= 0.0f) { z_out = 0.0f; break; }
                // Depth of the source sample expressed in the output camera.
                const float srx = (static_cast<float>(su) - k.cx) / k.fx;
                const float sry = (static_cast<float>(sv) - k.cy) / k.fy;
                const Eigen::Vector3f p_src(srx * z_src, sry * z_src, z_src);
                const Eigen::Vector3f p_out = R_inv[sv].transpose() * (p_src - t_inv[sv]);
                z_out = p_out.z();
            }
            if (z_out > 0.0f) out = cpuDepthMetersToRaw(z_out, min_depth_m, max_depth_m);
            dst[static_cast<size_t>(v) * width + u] = out;
        }
    }
}

} // namespace sensor
} // namespace kfusion
