// depth_consistency_contract (relocalization rework, step 5): render-and-compare
// verification in include/tracking/DepthConsistency.h, against an exact model
// of the synthetic room (tests/support/RoomModel.h).
//
//   A  tau(z): 3 cm floor, 3 sigma of the Kinect v1 model far away
//   B  the true pose passes: consistent >= 0.9, violation <= 0.02 (with noise)
//   C  25 deg of yaw off fails
//   D  30 cm forward of the truth fails with many violations (the live frame
//      sees surfaces the candidate puts closer)
//   E  looking at the floor while the live frame sees a wall fails
//   F  an object the model does not have, in front of the wall, counts as "in
//      front" and does not fail the true pose
#include "support/RoomModel.h"
#include "tracking/DepthConsistency.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, what)                                                        \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            std::printf("FAIL: %s  [%s:%d]\n", std::string(what).c_str(),        \
                        __FILE__, __LINE__);                                     \
            ++g_failures;                                                        \
        }                                                                        \
    } while (false)

using namespace kfusion::tracking;
constexpr float kDeg = 3.14159265f / 180.0f;

DepthConsistency check(const azu_test::RoomModel& room, const Eigen::Matrix4f& candidate,
                       const std::vector<float>& live) {
    const ModelFrame m = room.render(candidate, 320, 240);
    const auto& K = room.intrinsics();
    return depthConsistency(m.vertices.data(), m.normals.data(), m.width, m.height, candidate,
                            live.data(), K.width, K.height);
}

void print(const char* what, const DepthConsistency& c) {
    std::printf("  %-28s coverage %.3f consistent %.3f violation %.3f in_front %d -> %s\n", what,
                c.coverage(), c.consistentFraction(), c.violationFraction(), c.in_front,
                c.passes() ? "pass" : "fail");
}

}  // namespace

int main() {
    // A
    CHECK(std::fabs(consistencyTau(1.0f) - 0.03f) < 1e-6f, "A: 3 cm floor at 1 m");
    CHECK(std::fabs(consistencyTau(4.0f) - 3.0f * (0.0012f + 0.0019f * 3.6f * 3.6f)) < 1e-6f,
          "A: 3 sigma at 4 m");

    const azu_test::RoomModel room;
    const Eigen::Matrix4f truth = azu_test::RoomModel::yawPitch(40.0f * kDeg, -0.1f);
    const std::vector<float> live = room.depthAt(truth, 1.0f);

    // B
    const DepthConsistency b = check(room, truth, live);
    print("B true pose", b);
    CHECK(b.passes() && b.consistentFraction() >= 0.9f && b.violationFraction() <= 0.02f,
          "B: the true pose passes");

    // C
    const DepthConsistency c = check(room, truth * azu_test::makePose({0.0f, 25.0f * kDeg, 0.0f}, {0, 0, 0}), live);
    print("C 25 deg yaw off", c);
    CHECK(!c.passes(), "C: 25 deg of yaw off fails");

    // D
    const DepthConsistency d = check(room, truth * azu_test::makePose({0, 0, 0}, {0.0f, 0.0f, 0.30f}), live);
    print("D 30 cm forward", d);
    CHECK(!d.passes() && d.violationFraction() >= 0.2f, "D: 30 cm forward fails with violations");

    // E
    const Eigen::Matrix4f floor_pose = azu_test::RoomModel::yawPitch(40.0f * kDeg, -1.4f);
    const DepthConsistency e = check(room, floor_pose, live);
    print("E floor for a wall", e);
    CHECK(!e.passes(), "E: the floor does not pass for a wall");

    // F
    std::vector<float> occluded = live;
    const auto& K = room.intrinsics();
    for (int v = K.height / 3; v < K.height / 2; ++v)
        for (int u = K.width / 3; u < K.width / 2; ++u) occluded[static_cast<size_t>(v) * K.width + u] = 0.6f;
    const DepthConsistency f = check(room, truth, occluded);
    print("F new object in front", f);
    CHECK(f.in_front > b.in_front && f.passes(), "F: a new object in front does not fail the true pose");

    if (g_failures == 0) {
        std::printf("depth_consistency_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("depth_consistency_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
