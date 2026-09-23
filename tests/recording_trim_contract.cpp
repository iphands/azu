// recording_trim_contract: the still-period detector behind tools/azu_trim
// (tools/Stillness.h), on synthetic per-frame motion series at 30 fps whose
// levels come from a real handheld take (depth change: still ~0.004, turning
// ~0.1; accelerometer std: still ~0.015 g, turning ~0.04 g, reach ~0.08 g).
//
//   A  protocol take: move 2 s / hold 10 s / turn 30 s / hold 10 s / reach 2 s
//      -> cuts within 0.5 s of the hold boundaries (plus the 0.25 s margin)
//   B  a 1.2 s mid-turn lull that looks still on both signals is never used
//   C  the reach barely moves the camera (depth still) but the accelerometer
//      shows it: the end cut lands before the reach
//   D  a 0.2 s jolt inside a hold (0.7 s through the accelerometer window) does
//      not split it
//   E  no hold long enough: "not found" plus the best candidate
//   F  no accelerometer data: depth alone decides
//   G  depthChange / accelStd primitives
#include "Stillness.h"

#include <cmath>
#include <cstdio>
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

using azu_rec::FrameMotion;
using azu_rec::StillOptions;

struct Segment {
    double seconds, depth, accel_g;
};

std::vector<FrameMotion> series(const std::vector<Segment>& segs, bool with_accel = true) {
    std::vector<FrameMotion> f;
    double t = 0.0;
    for (const Segment& s : segs) {
        const int n = static_cast<int>(std::lround(s.seconds * 30.0));
        for (int i = 0; i < n; ++i, t += 1.0 / 30.0) {
            FrameMotion m;
            m.t = t;
            // Deterministic wobble so values are not flat.
            m.depth_change = s.depth * (1.0 + 0.3 * std::sin(t * 7.0));
            m.accel_std_g = s.accel_g * (1.0 + 0.2 * std::cos(t * 5.0));
            m.has_accel = with_accel;
            f.push_back(m);
        }
    }
    return f;
}

constexpr Segment kMove{0, 0.10, 0.06}, kHold{0, 0.004, 0.015}, kTurn{0, 0.10, 0.04};
Segment seg(Segment s, double seconds) { s.seconds = seconds; return s; }

bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

void sectionProtocol() {
    // Hold 2..12 s, turn 12..42 s (with a 1.2 s lull at 20 s), hold 42..52 s,
    // then a reach the camera barely feels but the accelerometer does.
    const std::vector<FrameMotion> f = series({seg(kMove, 2), seg(kHold, 10), seg(kTurn, 8),
                                               {1.2, 0.004, 0.015}, seg(kTurn, 20.8),
                                               seg(kHold, 10), {2, 0.004, 0.08}});
    const StillOptions o;
    const auto r = azu_rec::findTrim(f, o);
    std::printf("  A: start %.2f s end %.2f s (holds %.2f-%.2f, %.2f-%.2f)\n", r.start_t, r.end_t,
                r.start_run.t0, r.start_run.t1, r.end_run.t0, r.end_run.t1);
    CHECK(r.start_found && r.end_found, "A: both holds found");
    CHECK(near(r.start_t, 2.0 + o.margin, 0.5), "A: start cut at the start hold + margin");
    CHECK(near(r.end_t, 52.0 - o.margin, 0.5), "C: end cut before the reach");
    CHECK(r.start_run.duration() > 9.0, "A: the whole start hold is one run");
    CHECK(r.end_run.duration() > 9.0, "A: the whole end hold is one run");
    CHECK(r.start_run.t1 < 12.5, "B: the mid-turn lull is not merged into or used as the start");
}

void sectionJolt() {
    // A 0.2 s jolt (accelerometer only) in the middle of a 10 s hold. The
    // tool's accelerometer std is a centred 0.5 s window, so the jolt raises it
    // for ~0.7 s of frames; the series models that directly.
    const std::vector<FrameMotion> f =
        series({seg(kMove, 2), seg(kHold, 5), {0.7, 0.004, 0.12}, seg(kHold, 4.3), seg(kTurn, 20),
                seg(kHold, 10), seg(kMove, 2)});
    const auto r = azu_rec::findTrim(f, StillOptions{});
    CHECK(r.start_found && r.start_run.duration() > 9.0, "D: a jolt inside a hold does not split it");
}

void sectionNotFound() {
    // Only 1.5 s of hold at each end.
    const std::vector<FrameMotion> f = series({seg(kMove, 2), seg(kHold, 1.5), seg(kTurn, 20),
                                               seg(kHold, 1.5), seg(kMove, 2)});
    const auto r = azu_rec::findTrim(f, StillOptions{});
    std::printf("  E: best start %.2f-%.2f, best end %.2f-%.2f\n", r.start_run.t0, r.start_run.t1,
                r.end_run.t0, r.end_run.t1);
    CHECK(!r.start_found && !r.end_found, "E: holds shorter than min_still are not accepted");
    CHECK(r.have_start_run && near(r.start_run.duration(), 1.5, 0.4), "E: best start candidate reported");
    CHECK(r.have_end_run && near(r.end_run.duration(), 1.5, 0.4), "E: best end candidate reported");
    StillOptions shorter;
    shorter.min_still = 1.0;
    const auto r2 = azu_rec::findTrim(f, shorter);
    CHECK(r2.start_found && r2.end_found, "E: --min-still 1.0 accepts them");
}

void sectionNoAccel() {
    // Without accelerometer data the depth-still reach cannot be seen: the end
    // hold runs to the end of the recording.
    const std::vector<FrameMotion> f = series({seg(kMove, 2), seg(kHold, 10), seg(kTurn, 20),
                                               seg(kHold, 10), {2, 0.004, 0.08}},
                                              /*with_accel=*/false);
    const auto r = azu_rec::findTrim(f, StillOptions{});
    CHECK(r.start_found && near(r.start_t, 2.25, 0.5), "F: depth alone finds the start hold");
    // Take is 44 s long; the reach starts at 42 s.
    CHECK(r.end_found && r.end_t > 42.5, "F: depth alone cannot see the reach");
}

void sectionPrimitives() {
    std::vector<uint16_t> a(64 * 48, 800), b(64 * 48, 800);
    CHECK(azu_rec::depthChange(a, b, 64, 48, 1) == 0.0, "G: identical frames change 0");
    for (size_t i = 0; i < b.size() / 4; ++i) b[i] = 820;
    CHECK(near(azu_rec::depthChange(a, b, 64, 48, 1), 0.25, 1e-9), "G: 25% of pixels changed");
    for (size_t i = 0; i < b.size(); ++i) b[i] = static_cast<uint16_t>(802);
    CHECK(azu_rec::depthChange(a, b, 64, 48, 1) == 0.0, "G: a 2-code flicker is not motion");
    b[0] = 0;
    b[1] = 2047;
    CHECK(azu_rec::depthChange(a, b, 64, 48, 1) == 0.0, "G: invalid pixels are ignored");

    std::vector<azu_rec::AccelSample> s;
    for (int i = 0; i < 300; ++i) s.push_back({i / 300.0, 0.0, (i % 2) ? 1.01 : 0.99, 0.0});
    CHECK(near(azu_rec::accelStd(s, 0.5, 0.25), 0.01, 1e-6), "G: accelerometer std in g");
    CHECK(azu_rec::accelStd(s, 5.0, 0.25) < 0.0, "G: no samples in the window -> -1");
}

}  // namespace

int main() {
    sectionProtocol();
    sectionJolt();
    sectionNotFound();
    sectionNoAccel();
    sectionPrimitives();
    if (g_failures == 0) {
        std::printf("recording_trim_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("recording_trim_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
