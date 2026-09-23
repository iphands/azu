// Shared ground-truth scene for CPU tests (big-fix-two T0.1).
//
// An analytic signed-distance room that fits the default 256^3 / 1 cm volume
// (x, y in [-1.28, 1.28], z in [0, 2.56]): side walls at x = +-1.2, floor at
// y = +0.9 (camera y points down), ceiling at y = -1.0, back wall at z = 2.4,
// plus a box and a sphere so every 6-DoF direction is constrained for ICP.
// renderDepth() sphere-traces the SDF through pinhole intrinsics and returns
// camera-plane Z-depth in meters (0 = no return), optionally with the Kinect v1
// axial noise model sigma(z) = 0.0012 + 0.0019 (z - 0.4)^2 from a seeded RNG.
#pragma once

#include "sensor/FrameData.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace azu_test {

struct Intrinsics {
    int   width  = kfusion::sensor::FRAME_W;
    int   height = kfusion::sensor::FRAME_H;
    float fx = kfusion::sensor::FX, fy = kfusion::sensor::FY;
    float cx = kfusion::sensor::CX, cy = kfusion::sensor::CY;
};

class SyntheticScene {
public:
    // Signed distance (meters) to the nearest surface; positive in free space.
    float sdf(const Eigen::Vector3f& p) const {
        // Inside of the room box: distance to the nearest wall.
        const Eigen::Vector3f c(0.0f, -0.05f, 1.15f), h(1.2f, 0.95f, 1.25f);
        const Eigen::Vector3f q = (p - c).cwiseAbs() - h;
        const float room = -(q.cwiseMax(0.0f).norm() + std::min(q.maxCoeff(), 0.0f));
        return std::min({room, box(p), sphere(p)});
    }

    // Z-depth image seen from world-from-camera `pose`.
    std::vector<float> renderDepth(const Eigen::Matrix4f& pose, const Intrinsics& K = {},
                                   float noise_scale = 0.0f, uint32_t seed = 42) const {
        std::vector<float> depth(static_cast<size_t>(K.width) * K.height, 0.0f);
        const Eigen::Matrix3f R = pose.block<3,3>(0,0);
        const Eigen::Vector3f o = pose.block<3,1>(0,3);
        #pragma omp parallel for schedule(dynamic, 8)
        for (int v = 0; v < K.height; ++v) {
            for (int u = 0; u < K.width; ++u) {
                const Eigen::Vector3f rc((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0f);
                const float norm = rc.norm();
                const Eigen::Vector3f d = R * rc / norm;
                float t = 0.05f;
                for (int i = 0; i < 256 && t < 8.0f; ++i) {
                    const float s = sdf(o + d * t);
                    if (s < 1e-5f) {
                        depth[static_cast<size_t>(v) * K.width + u] = t / norm;
                        break;
                    }
                    t += s;
                }
            }
        }
        if (noise_scale > 0.0f) {
            std::mt19937 rng(seed);
            std::normal_distribution<float> n01(0.0f, 1.0f);
            for (float& z : depth) {
                if (z <= 0.0f) continue;
                const float sigma = 0.0012f + 0.0019f * (z - 0.4f) * (z - 0.4f);
                z += noise_scale * sigma * n01(rng);
            }
        }
        return depth;
    }

private:
    static float box(const Eigen::Vector3f& p) {
        const Eigen::Vector3f c(0.35f, 0.55f, 1.5f), h(0.3f, 0.35f, 0.25f);
        const Eigen::Vector3f q = (p - c).cwiseAbs() - h;
        return q.cwiseMax(0.0f).norm() + std::min(q.maxCoeff(), 0.0f);
    }
    static float sphere(const Eigen::Vector3f& p) {
        return (p - Eigen::Vector3f(-0.5f, 0.2f, 1.7f)).norm() - 0.3f;
    }
};

// Fill a FrameData (depth, vertices, normals) from a Z-depth image, the way
// buildFrameData does from raw sensor codes.
inline void fillFrame(const std::vector<float>& depth, kfusion::sensor::FrameData& f,
                      const Intrinsics& K = {}) {
    f.width  = K.width;
    f.height = K.height;
    f.depth_meters = depth;
    f.vertices.assign(depth.size(), Eigen::Vector3f::Zero());
    f.normals.assign(depth.size(), Eigen::Vector3f::Zero());
    for (int v = 0; v < K.height; ++v) {
        for (int u = 0; u < K.width; ++u) {
            const size_t i = static_cast<size_t>(v) * K.width + u;
            const float z = depth[i];
            if (z <= 0.0f) continue;
            f.vertices[i] = Eigen::Vector3f((u - K.cx) / K.fx * z, (v - K.cy) / K.fy * z, z);
        }
    }
    kfusion::sensor::computeNormals(f);
}

// Rigid transform from axis-angle (radians) and translation.
inline Eigen::Matrix4f makePose(const Eigen::Vector3f& axis_angle, const Eigen::Vector3f& t) {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    const float a = axis_angle.norm();
    if (a > 0.0f) T.block<3,3>(0,0) = Eigen::AngleAxisf(a, axis_angle / a).toRotationMatrix();
    T.block<3,1>(0,3) = t;
    return T;
}

// Translation (m) and rotation (deg) between two poses.
struct PoseError {
    float trans_m;
    float rot_deg;
};
inline PoseError poseError(const Eigen::Matrix4f& a, const Eigen::Matrix4f& b) {
    const Eigen::Matrix4f d = a.inverse() * b;
    const float c = std::max(-1.0f, std::min(1.0f, (d.block<3,3>(0,0).trace() - 1.0f) * 0.5f));
    return {d.block<3,1>(0,3).norm(), std::acos(c) * 57.2957795f};
}

}  // namespace azu_test
