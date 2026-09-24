// reloc_hypotheses_contract (relocalization rework, step 4): the candidate-pose
// math in include/tracking/RelocHypotheses.h.
//
//   A  snapToGravity: afterwards the pose predicts exactly the measured up
//      (0.01 deg), keeps its translation, and turns by the least angle (the
//      tilt error); a no-op when already aligned; deterministic when opposite
//   B  yawAboutUp keeps the tilt; the camera centre moves on a circle about
//      the pivot; bodyPivot is 0.12 m behind along the horizontal forward
//   C  buildSweep: 34 candidates, ordered by |yaw|, 180 deg once per pivot
//   D  dedupeHypotheses keeps the first of near-duplicates, drops non-finite
#include "tracking/Gravity.h"
#include "tracking/RelocHypotheses.h"

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

Eigen::Matrix4f pose(const Eigen::Vector3f& aa_deg, const Eigen::Vector3f& t) {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    const float a = aa_deg.norm();
    if (a > 0.0f) T.block<3,3>(0,0) = Eigen::AngleAxisf(a * kDeg, aa_deg / a).toRotationMatrix();
    T.block<3,1>(0,3) = t;
    return T;
}

// Up the accelerometer measures for a camera at `truth` (world up -y).
Eigen::Vector3f measuredUp(const Eigen::Matrix4f& truth) {
    return truth.block<3,3>(0,0).transpose() * Eigen::Vector3f(0.0f, -1.0f, 0.0f) * 9.1f;
}

bool near(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

}  // namespace

int main() {
    const Eigen::Vector3f up_world(0.0f, -1.0f, 0.0f);

    // A
    const Eigen::Matrix4f truth = pose({20.0f, 50.0f, -8.0f}, {0.3f, -0.1f, 0.5f});
    const Eigen::Matrix4f guess = pose({2.0f, 55.0f, 5.0f}, {0.3f, -0.1f, 0.5f});
    const Eigen::Vector3f m = measuredUp(truth);
    const float tilt_before = tiltErrorDeg(guess, up_world, m);
    const Eigen::Matrix4f snapped = snapToGravity(guess, up_world, m);
    CHECK(tilt_before > 10.0f, "A: the guess is visibly tilted");
    CHECK(tiltErrorDeg(snapped, up_world, m) < 0.01f, "A: snapped pose predicts the measured up");
    CHECK((snapped.block<3,1>(0,3) - guess.block<3,1>(0,3)).norm() == 0.0f, "A: translation kept");
    CHECK(near(poseGap(guess, snapped).rot_deg, tilt_before, 0.01f),
          "A: the turn is the least one (its angle is the tilt error)");
    const Eigen::Matrix4f again = snapToGravity(snapped, up_world, m);
    CHECK(poseGap(again, snapped).rot_deg < 0.01f, "A: snapping an aligned pose is a no-op");
    const Eigen::Matrix4f flipped =
        snapToGravity(Eigen::Matrix4f::Identity(), up_world, Eigen::Vector3f(0.0f, 9.1f, 0.0f));
    CHECK(tiltErrorDeg(flipped, up_world, Eigen::Vector3f(0.0f, 9.1f, 0.0f)) < 0.01f &&
              near(poseGap(flipped, Eigen::Matrix4f::Identity()).rot_deg, 180.0f, 0.01f),
          "A: an upside-down reading turns 180 deg deterministically");

    // B
    const Eigen::Vector3f pivot = bodyPivot(snapped, up_world, 0.12f);
    const Eigen::Vector3f c0 = snapped.block<3,1>(0,3);
    const Eigen::Vector3f back = c0 - pivot;
    CHECK(near(back.norm(), 0.12f, 1e-5f) && near(back.dot(up_world), 0.0f, 1e-5f),
          "B: body pivot 0.12 m behind, horizontally");
    Eigen::Vector3f fwd = snapped.block<3,1>(0,2);
    fwd -= up_world * fwd.dot(up_world);
    CHECK(back.normalized().dot(fwd.normalized()) > 0.9999f, "B: ...along the horizontal forward");
    for (float deg : {35.0f, 90.0f, 180.0f}) {
        const Eigen::Matrix4f turned = yawAboutUp(snapped, deg * kDeg, up_world, pivot);
        CHECK(tiltErrorDeg(turned, up_world, measuredUp(yawAboutUp(truth, deg * kDeg, up_world, pivot))) <
                  0.01f,
              "B: yaw about up keeps the tilt against gravity");
        CHECK(near((turned.block<3,1>(0,3) - pivot).norm(), 0.12f, 1e-5f),
              "B: the camera stays on the circle about the pivot");
        CHECK(near(poseGap(turned, snapped).rot_deg, deg, 0.02f), "B: turned by the yaw");
    }

    // C
    const std::vector<Hypothesis> sweep = buildSweep(snapped, up_world);
    CHECK(sweep.size() == 34, "C: 34 sweep candidates");
    float last = 0.0f;
    bool ordered = true;
    int half_turns = 0;
    for (const Hypothesis& h : sweep) {
        const float yaw = poseGap(h.pose, snapped).rot_deg;
        if (yaw + 0.05f < last) ordered = false;
        last = yaw;
        if (near(yaw, 180.0f, 0.05f)) ++half_turns;
        if (h.source != HypothesisSource::Sweep) ordered = false;
    }
    CHECK(ordered, "C: ordered by |yaw|, all tagged Sweep");
    CHECK(half_turns == 2, "C: 180 deg once per pivot");
    CHECK(near(poseGap(sweep.front().pose, snapped).rot_deg, 20.0f, 0.02f) &&
              poseGap(sweep.front().pose, snapped).trans_m < 1e-5f,
          "C: first candidate is +20 deg about the camera centre");

    // D
    std::vector<Hypothesis> hs = {{snapped, HypothesisSource::LastGood},
                                  {snapped * pose({0.0f, 2.0f, 0.0f}, {0.01f, 0.0f, 0.0f}),
                                   HypothesisSource::ModelPose},
                                  {snapped * pose({0.0f, 6.0f, 0.0f}, {0.0f, 0.0f, 0.0f}),
                                   HypothesisSource::Keyframe}};
    Eigen::Matrix4f nan_pose = snapped;
    nan_pose(0, 3) = std::nanf("");
    hs.push_back({nan_pose, HypothesisSource::CarryOver});
    const std::vector<Hypothesis> kept = dedupeHypotheses(hs);
    CHECK(kept.size() == 2 && kept[0].source == HypothesisSource::LastGood &&
              kept[1].source == HypothesisSource::Keyframe,
          "D: 2 deg / 1 cm duplicate dropped, 6 deg kept, non-finite dropped");
    CHECK(pitchGrid(snapped).size() == 8, "D: the old grid has 8 candidates");

    if (g_failures == 0) {
        std::printf("reloc_hypotheses_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("reloc_hypotheses_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
