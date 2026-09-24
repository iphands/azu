// An exact TSDF model of the synthetic room for relocalization tests.
//
// The world is the camera standing at the room centre (as in
// pipeline_spin_contract, whose world is its first camera): the scene's pose of
// a world pose W is room_from_start * W. The volume (256^3 / 1 cm, centred on
// that camera) is fused noise-free from known poses — a full turn of 24 views
// 15 deg apart, plus a ring pitched 25 deg down and one 25 deg up for the floor
// and ceiling — so a test starts from a complete, exact model instead of a
// tracked one.
#pragma once

#include "sensor/CameraIntrinsics.h"
#include "support/SyntheticScene.h"
#include "tracking/ICPTracker.h"
#include "tsdf/TSDFVolume.h"

#include <Eigen/Core>

#include <memory>
#include <vector>

namespace azu_test {

class RoomModel {
public:
    explicit RoomModel(int views = 24) {
        kfusion::tsdf::TSDFParams p;
        p.origin = Eigen::Vector3f(-1.28f, -1.28f, -1.28f);
        p.min_depth = kMinDepth;
        p.max_depth = kMaxDepth;
        vol_ = std::make_unique<kfusion::tsdf::TSDFVolume>(p);
        const float step = 2.0f * 3.14159265f / static_cast<float>(views);
        for (float pitch : {0.0f, -0.436f, 0.436f}) {
            for (int i = 0; i < views; ++i) {
                const Eigen::Matrix4f w = yawPitch(step * static_cast<float>(i), pitch);
                const std::vector<float> d = depthAt(w);
                vol_->integrate(d.data(), nullptr, w, K_.fx, K_.fy, K_.cx, K_.cy, K_.width, K_.height,
                                kMinDepth, kMaxDepth);
            }
        }
    }

    // A camera at the world origin turned by yaw (about camera/world y) then pitch.
    static Eigen::Matrix4f yawPitch(float yaw, float pitch, const Eigen::Vector3f& t = Eigen::Vector3f::Zero()) {
        return makePose({0.0f, yaw, 0.0f}, t) * makePose({pitch, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f});
    }

    std::vector<float> depthAt(const Eigen::Matrix4f& world_pose, float noise = 0.0f,
                               uint32_t seed = 42) const {
        return scene_.renderDepth(room_from_start_ * world_pose, K_, noise, seed);
    }

    kfusion::tracking::ModelFrame render(const Eigen::Matrix4f& pose, int w, int h) const {
        const kfusion::sensor::CameraIntrinsics k =
            kfusion::sensor::scaleIntrinsics(camera(), K_.width, K_.height, w, h);
        kfusion::tracking::ModelFrame m(w, h);
        vol_->raycast(pose, k.fx, k.fy, k.cx, k.cy, w, h, m.vertices.data(), m.normals.data(),
                      m.colors.data());
        m.pose = pose;
        return m;
    }

    kfusion::sensor::CameraIntrinsics camera() const { return {K_.fx, K_.fy, K_.cx, K_.cy}; }
    const Intrinsics& intrinsics() const { return K_; }
    const kfusion::tsdf::TSDFVolume& volume() const { return *vol_; }

    static constexpr float kMinDepth = 0.3f;
    static constexpr float kMaxDepth = 3.0f;

private:
    SyntheticScene scene_;
    Intrinsics K_;
    Eigen::Matrix4f room_from_start_ = makePose({0.0f, 0.0f, 0.0f}, {0.0f, -0.05f, 1.15f});
    std::unique_ptr<kfusion::tsdf::TSDFVolume> vol_;
};

}  // namespace azu_test
