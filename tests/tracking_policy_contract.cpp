// tracking_policy_contract (big-fix-two T0.12): the frame grading used by the
// pipeline (include/tracking/TrackingPolicy.h). Good integrates, Poor only moves
// the pose, Failed keeps the old pose; convergence is not an input.
#include "tracking/TrackingPolicy.h"

#include <cmath>
#include <cstdio>
#include <limits>
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

Eigen::Matrix4f move(float tx, float yaw_rad) {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,3>(0,0) = Eigen::AngleAxisf(yaw_rad, Eigen::Vector3f::UnitY()).toRotationMatrix();
    T(0,3) = tx;
    return T;
}

ICPResult fit(const Eigen::Matrix4f& pose, int inliers, int valid_live, float error) {
    ICPResult r;
    r.pose = pose;
    r.inliers = inliers;
    r.valid_live_points = valid_live;
    r.error = error;
    r.converged = false;       // deliberately: convergence must not matter
    r.tracking_ok = false;
    return r;
}

}  // namespace

int main() {
    const Eigen::Matrix4f prev = move(0.3f, 0.2f);
    const Eigen::Matrix4f near = prev * move(0.02f, 0.03f);

    CHECK(classifyTracking(fit(near, 150000, 200000, 4e-6f), prev) == TrackQuality::Good,
          "non-converged solve with a tight fit and small motion is Good");
    CHECK(classifyTracking(fit(near, 30000, 200000, 4e-6f), prev) == TrackQuality::Poor,
          "inlier ratio 0.15 < 0.30 is Poor");
    CHECK(classifyTracking(fit(near, 150000, 200000, 9e-4f), prev) == TrackQuality::Poor,
          "RMS 3 cm > 1.5 cm is Poor");
    CHECK(classifyTracking(fit(prev * move(0.2f, 0.0f), 150000, 200000, 4e-6f), prev) ==
              TrackQuality::Failed,
          "20 cm in one frame is Failed");
    CHECK(classifyTracking(fit(prev * move(0.0f, 0.7f), 150000, 200000, 4e-6f), prev) ==
              TrackQuality::Failed,
          "40 deg in one frame is Failed");
    CHECK(classifyTracking(fit(near, 99, 200000, 4e-6f), prev) == TrackQuality::Failed,
          "fewer than 100 inliers is Failed");
    Eigen::Matrix4f bad = near;
    bad(0, 3) = std::numeric_limits<float>::quiet_NaN();
    CHECK(classifyTracking(fit(bad, 150000, 200000, 4e-6f), prev) == TrackQuality::Failed,
          "a non-finite pose is Failed");
    CHECK(classifyTracking(fit(near, 150000, 200000, std::numeric_limits<float>::infinity()), prev) ==
              TrackQuality::Failed,
          "a non-finite error is Failed");
    CHECK(TrackingPolicy{}.failures_before_lost == 3, "three consecutive failures enter Lost");

    if (g_failures == 0) {
        std::printf("tracking_policy_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("tracking_policy_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
