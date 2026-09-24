// tracking_policy_contract (big-fix-two T0.12, policy v2): the frame grading used
// by the pipeline (include/tracking/TrackingPolicy.h). Good integrates, Poor only
// moves the pose, Failed keeps the old pose; convergence is not an input, and the
// fit ratio is taken over model correspondences, not over all live points.
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

ICPResult fit(const Eigen::Matrix4f& pose, int inliers, int valid_model, float error,
              int valid_live = 300000) {
    ICPResult r;
    r.pose = pose;
    r.inliers = inliers;
    r.valid_model_points = valid_model;
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
          "fit ratio 0.15 of the model correspondences < 0.40 is Poor");
    // v2: turning toward new geometry leaves most LIVE points off the model.
    // 40k inliers of 45k model correspondences out of 300k live points is a
    // tight fit that must integrate (v1 graded it Poor: 40k/300k < 0.30).
    CHECK(classifyTracking(fit(near, 40000, 45000, 4e-6f, 300000), prev) == TrackQuality::Good,
          "a tight fit on a small overlap (new geometry in view) is Good");
    CHECK(classifyTracking(fit(near, 1500, 1600, 4e-6f), prev) == TrackQuality::Poor,
          "fewer than 2000 inliers cannot be Good");
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

    // classifyFit: the same fit grades without the motion gate (relocalization).
    const Eigen::Matrix4f far = prev * move(0.45f, 0.7f);
    CHECK(classifyFit(fit(far, 150000, 200000, 4e-6f)) == TrackQuality::Good,
          "classifyFit: a tight fit 45 cm / 40 deg away is still Good");
    CHECK(classifyTracking(fit(far, 150000, 200000, 4e-6f), prev) == TrackQuality::Failed,
          "classifyTracking: the same result fails the motion gate");
    CHECK(classifyFit(fit(far, 30000, 200000, 4e-6f)) == TrackQuality::Poor,
          "classifyFit: a weak fit is Poor");
    CHECK(classifyFit(fit(far, 99, 200000, 4e-6f)) == TrackQuality::Failed,
          "classifyFit: fewer than 100 inliers is Failed");
    CHECK(withinFrameMotion(near, prev) && !withinFrameMotion(far, prev),
          "withinFrameMotion: 2 cm / 1.7 deg inside, 45 cm / 40 deg outside");

    // weakestDirectionRatio: a fully constrained system vs one with a free direction.
    Eigen::Matrix<float, 6, 6> info = Eigen::Matrix<float, 6, 6>::Identity() * 100.0f;
    info(5, 5) = 1.0f;
    CHECK(std::fabs(weakestDirectionRatio(info) - 0.01f) < 1e-5f, "weakestDirectionRatio: 1/100");
    info(5, 5) = 0.0f;
    CHECK(weakestDirectionRatio(info) == 0.0f, "weakestDirectionRatio: an unobserved direction is 0");
    CHECK(weakestDirectionRatio(Eigen::Matrix<float, 6, 6>::Zero()) == 0.0f,
          "weakestDirectionRatio: an empty system is 0");

    if (g_failures == 0) {
        std::printf("tracking_policy_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("tracking_policy_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
