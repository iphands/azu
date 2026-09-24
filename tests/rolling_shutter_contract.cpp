// rolling_shutter_contract: the rolling-shutter unwarp (include/sensor/RollingShutter.h)
// on the synthetic room rendered row by row, each row from its own pose.
//
//   A  no motion or no readout: the frame comes back unchanged
//   B  a fast pitch + yaw + slide (3 deg / 1 deg / 1 cm per frame, full-frame
//      readout): the unwarped frame matches the global-shutter render at the
//      mid-row pose far better than the rolling-shutter frame does
//   C  unwarping with the wrong sign (bottom row first) makes it worse
#include "sensor/RollingShutter.h"
#include "support/SyntheticScene.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

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

using namespace kfusion;

constexpr float kMin = 0.3f, kMax = 8.0f;
constexpr float kPeriod = 1.0f / 30.0f;

// Row v seen from mid * exp(s(v) * log(step)), the model the unwarp inverts.
std::vector<uint16_t> renderRaw(const azu_test::SyntheticScene& scene, const Eigen::Matrix4f& mid,
                                const Eigen::Matrix4f& step, float readout_s,
                                const azu_test::Intrinsics& K) {
    const Eigen::AngleAxisf aa(Eigen::Matrix3f(step.block<3,3>(0,0)));
    const Eigen::Vector3f t = step.block<3,1>(0,3);
    std::vector<uint16_t> raw(static_cast<size_t>(K.width) * K.height, 0);
    // One parallel region over rows; renderDepth's own region is then nested
    // (inactive) instead of 480 one-row regions.
    #pragma omp parallel for schedule(dynamic, 8)
    for (int v = 0; v < K.height; ++v) {
        azu_test::Intrinsics row_k = K;
        row_k.height = 1;
        const float s = (static_cast<float>(v) / static_cast<float>(K.height - 1) - 0.5f) *
                        (readout_s / kPeriod);
        Eigen::Matrix4f d = Eigen::Matrix4f::Identity();
        d.block<3,3>(0,0) = Eigen::AngleAxisf(aa.angle() * s, aa.axis()).toRotationMatrix();
        d.block<3,1>(0,3) = t * s;
        row_k.cy = K.cy - static_cast<float>(v);   // render only image row v
        const std::vector<float> z = scene.renderDepth(mid * d, row_k);
        for (int u = 0; u < K.width; ++u) {
            raw[static_cast<size_t>(v) * K.width + u] = sensor::cpuDepthMetersToRaw(z[u], kMin, kMax);
        }
    }
    return raw;
}

// Share of pixels valid in both frames whose raw code differs by more than 2.
double mismatch(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    size_t both = 0, off = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] == 0 || b[i] == 0) continue;
        ++both;
        if (std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i])) > 2) ++off;
    }
    return both ? static_cast<double>(off) / static_cast<double>(both) : 1.0;
}

// Valid share away from the border. Resampling into the mid-row camera shifts
// the top and bottom rows by up to half the per-frame motion (1.5 deg ~ 14 px
// here), so the border legitimately loses data; inside it nothing should.
double interiorValidShare(const std::vector<uint16_t>& a, int width, int height) {
    size_t n = 0, all = 0;
    for (int v = 24; v < height - 24; ++v) {
        for (int u = 12; u < width - 12; ++u) {
            ++all;
            n += a[static_cast<size_t>(v) * width + u] != 0;
        }
    }
    return static_cast<double>(n) / static_cast<double>(all);
}

}  // namespace

int main() {
    const azu_test::SyntheticScene scene;
    const azu_test::Intrinsics K;
    const sensor::CameraIntrinsics k{K.fx, K.fy, K.cx, K.cy};
    const Eigen::Matrix4f mid = azu_test::makePose({0.05f, -0.1f, 0.0f}, {0.1f, 0.0f, 0.2f});
    constexpr float kDeg = 3.14159265f / 180.0f;
    const Eigen::Matrix4f step = azu_test::makePose({3.0f * kDeg, 1.0f * kDeg, 0.0f}, {0.01f, 0.0f, 0.0f});

    const std::vector<uint16_t> global = renderRaw(scene, mid, step, 0.0f, K);
    std::vector<uint16_t> out(global.size());

    // A
    sensor::unwarpRollingShutter(global.data(), out.data(), K.width, K.height, k,
                                 Eigen::Matrix4f::Identity(), kPeriod, kPeriod, kMin, kMax);
    CHECK(out == global, "A: no motion leaves the frame unchanged");
    sensor::unwarpRollingShutter(global.data(), out.data(), K.width, K.height, k, step, 0.0f,
                                 kPeriod, kMin, kMax);
    CHECK(out == global, "A: zero readout leaves the frame unchanged");

    // B
    const std::vector<uint16_t> rolling = renderRaw(scene, mid, step, kPeriod, K);
    sensor::unwarpRollingShutter(rolling.data(), out.data(), K.width, K.height, k, step, kPeriod,
                                 kPeriod, kMin, kMax);
    const double before = mismatch(rolling, global), after = mismatch(out, global);
    const double valid_in = interiorValidShare(rolling, K.width, K.height);
    const double valid_out = interiorValidShare(out, K.width, K.height);
    std::printf("  B: pixels off by > 2 codes vs global shutter: rolling %.3f, unwarped %.3f "
                "(interior valid %.4f -> %.4f)\n", before, after, valid_in, valid_out);
    CHECK(before > 0.1, "B: the rolling-shutter frame is visibly distorted");
    CHECK(after < 0.1 * before, "B: the unwarp removes most of the distortion");
    CHECK(valid_out > 0.99 * valid_in, "B: the unwarp does not punch holes inside the frame");

    // C
    sensor::unwarpRollingShutter(rolling.data(), out.data(), K.width, K.height, k, step, -kPeriod,
                                 kPeriod, kMin, kMax);
    const double wrong = mismatch(out, global);
    std::printf("  C: wrong-sign unwarp %.3f\n", wrong);
    CHECK(wrong > before, "C: the wrong readout direction makes it worse");

    if (g_failures == 0) {
        std::printf("rolling_shutter_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("rolling_shutter_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
