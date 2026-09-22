// depth_ema_determinism_contract (big-fix Todo 20): CPU-only contract that the
// temporal depth EMA reads a source snapshot and writes a destination, so its
// output is a pure function of (input frame, per-pixel history, band). Public
// CPU seam only (AZU_PIPELINE_TEST_SEAM applyDepthEmaForTests + emaStateForTests
// + sensor/DepthValidity.h): no device, display, GPU, sensor, thread timing,
// sleep or filesystem.
//
// The fixture is deliberately asymmetric between a pixel's own history and its
// neighbours' history, which is the only shape that can distinguish a
// source-buffer read from a destination-buffer read:
//
//   frame 1: wall raw 500 everywhere, centre raw 548
//            -> history wall  0.556979 m, history centre 0.606801 m
//   frame 2: wall raw 548 everywhere, centre raw 586
//            wall:   delta = |0.606801 - 0.556979| = 0.049822 <= 0.05 -> no jump
//                    reset; neighbour mean = 0.606801 -> no reset;
//                    out = 0.7*0.606801 + 0.3*0.556979 = 0.591855 -> raw 534
//            centre: delta = |0.653047 - 0.606801| = 0.046245 -> no jump reset,
//                    neighbour mean (SOURCE) = 0.606801, |0.653047 - 0.606801|
//                    = 0.046245 <= 0.05 -> no reset;
//                    out = 0.7*0.653047 + 0.3*0.606801 = 0.639173 -> raw 575
//
// If the EMA reads its own destination instead of the snapshot, some of those
// eight neighbours have already become 0.591855. Any mixture x of 0.591855 and
// 0.606801 with weight >= ~0.31 on the updated value puts |0.653047 - x| above
// 0.05, which flips the centre to the reset branch and emits raw 586 instead of
// 575. With one thread the four lower-index neighbours are always updated first
// (mean 0.599328, |diff| 0.053719 > 0.05), so the pre-fix code misses raw 575
// deterministically, not occasionally.
//
// Sections:
//   A  the two-frame oracle: every pixel of the 640x480 destination equals an
//      independent double-precision implementation of the canonical rule, and
//      the hand-derived literals (534 / 575) hold
//   B  the anti-contamination literal: centre is raw 575 and explicitly not 586
//   C  repeated runs from a fresh conditioner are byte-identical (raw buffer and
//      float state, FNV-1a digests)
//   D  OMP_NUM_THREADS-equivalent 1 / 2 / 4 reproduce the same digests
//   E  reset lifecycle: resetEMA() and reset() both clear history to 0.0f, and
//      the frame after a reset is the identity on in-band raws
#include "sensor/DepthValidity.h"
#include "sensor/FrameData.h"
#include "sensor/SignalConditioner.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef AZU_PIPELINE_TEST_SEAM
#error "depth_ema_determinism_contract must be compiled with AZU_PIPELINE_TEST_SEAM"
#endif

namespace {

using kfusion::sensor::cpuDepthMeters;
using kfusion::sensor::cpuDepthMetersToRaw;
using kfusion::sensor::FRAME_H;
using kfusion::sensor::FRAME_W;
using kfusion::sensor::SignalConditioner;

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, what)                                                           \
    do {                                                                            \
        ++g_checks;                                                                 \
        if (!(cond)) {                                                              \
            std::printf("FAIL: %s  [%s:%d]\n", std::string(what).c_str(), __FILE__, \
                        __LINE__);                                                  \
            ++g_failures;                                                           \
        }                                                                           \
    } while (false)

constexpr double kCurveA = -0.0030711016;
constexpr double kCurveB = 3.3309495161;
constexpr double kJumpReset = 0.05;    // documented kEmaJumpResetMeters
constexpr float  kLo = 0.30f, kHi = 2.50f;

constexpr int   kWall1 = 500;
constexpr int   kWall2 = 548;
constexpr int   kCentre = 586;
constexpr int   kCx = 320, kCy = 240;
constexpr int   kExpectWall = 534;
constexpr int   kExpectCentre = 575;
constexpr int   kContaminatedCentre = 586;

double oracleMeters(int raw) { return 1.0 / (static_cast<double>(raw) * kCurveA + kCurveB); }
bool   oracleInvalid(int raw) { return raw == 0 || raw >= 2047; }
bool   oracleBand(double m) { return std::isfinite(m) && m >= kLo && m <= kHi; }
double oracleUsableMeters(int raw) {
    return oracleInvalid(raw) ? 0.0 : (oracleBand(oracleMeters(raw)) ? oracleMeters(raw) : 0.0);
}
int    oracleEncode(double m) {
    if (!oracleBand(m)) return 0;
    const double raw = (1.0 / m - kCurveB) / kCurveA;
    const long c = std::lround(raw);
    return (c >= 1 && c <= 2046) ? static_cast<int>(c) : 0;
}
int    reflect(int v, int max_v) {
    if (v < 0) return -v - 1;
    if (v >= max_v) return 2 * max_v - v - 1;
    return v;
}

// Independent double oracle for one EMA pass over the whole frame.
struct EmaResult {
    std::vector<uint16_t> out;
    std::vector<double>   state;
};

EmaResult oracleEma(const std::vector<uint16_t>& in, const std::vector<double>& state_in) {
    EmaResult r;
    r.out.assign(static_cast<size_t>(FRAME_W) * FRAME_H, 0);
    r.state.assign(static_cast<size_t>(FRAME_W) * FRAME_H, 0.0);
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            const int i = y * FRAME_W + x;
            const double m = oracleUsableMeters(in[static_cast<size_t>(i)]);
            if (m == 0.0) {
                r.state[static_cast<size_t>(i)] = 0.0;
                r.out[static_cast<size_t>(i)] = in[static_cast<size_t>(i)];  // passthrough
                continue;
            }
            const double prev = state_in[static_cast<size_t>(i)];
            const double delta = std::abs(m - prev);
            bool reset = !(prev > 0.0) || delta > kJumpReset;
            if (!reset && delta > kJumpReset * 0.5) {
                double sum = 0.0;
                int n = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        const int nx = reflect(x + dx, FRAME_W);
                        const int ny = reflect(y + dy, FRAME_H);
                        const double nm = oracleUsableMeters(in[static_cast<size_t>(ny) * FRAME_W + nx]);
                        if (nm != 0.0) { sum += nm; ++n; }
                    }
                }
                if (n > 0 && std::abs(m - sum / n) > kJumpReset) reset = true;
            }
            const double filtered = reset ? m : (0.7 * m + 0.3 * prev);
            const int code = oracleEncode(filtered);
            r.state[static_cast<size_t>(i)] = (code != 0) ? filtered : 0.0;
            r.out[static_cast<size_t>(i)] = static_cast<uint16_t>(code);
        }
    }
    return r;
}

std::vector<uint16_t> makeFrame(int wall, int centre_raw, int cx = kCx, int cy = kCy) {
    std::vector<uint16_t> d(static_cast<size_t>(FRAME_W) * FRAME_H,
                            static_cast<uint16_t>(wall));
    d[static_cast<size_t>(cy) * FRAME_W + cx] = static_cast<uint16_t>(centre_raw);
    return d;
}

// FNV-1a over the destination bytes then the float state bits.
uint64_t fnv(const std::vector<uint16_t>& buf, const std::vector<float>& state) {
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint8_t b) { h ^= b; h *= 1099511628211ull; };
    for (uint16_t v : buf) { mix(static_cast<uint8_t>(v & 0xFF)); mix(static_cast<uint8_t>(v >> 8)); }
    for (float f : state) {
        uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(bits));
        for (int s = 0; s < 32; s += 8) mix(static_cast<uint8_t>((bits >> s) & 0xFF));
    }
    return h;
}

// Runs the whole two-frame sequence on a fresh conditioner and digests both the
// final destination buffer and the retained float state.
uint64_t runSequence(int& centre_out, int& wall_out, std::vector<float>* state_out) {
    SignalConditioner sc;
    std::vector<uint16_t> f1 = makeFrame(kWall1, kWall2);
    sc.applyDepthEmaForTests(f1, kLo, kHi);
    std::vector<uint16_t> f2 = makeFrame(kWall2, kCentre);
    sc.applyDepthEmaForTests(f2, kLo, kHi);
    const std::vector<float>& st = sc.emaStateForTests();
    centre_out = f2[static_cast<size_t>(kCy) * FRAME_W + kCx];
    wall_out   = f2[static_cast<size_t>(kCy) * FRAME_W + (kCx + 40)];
    if (state_out) *state_out = st;
    return fnv(f2, st);
}

// ---- OpenMP thread control (restore prior state after the test) -------------
int g_savedThreads = 0;
void pinThreads(int n) {
#ifdef _OPENMP
    g_savedThreads = omp_get_max_threads();
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}
void restoreThreads() {
#ifdef _OPENMP
    omp_set_num_threads(g_savedThreads);
#endif
}

// ---------------------------------------------------------------------------
// A + B  oracle over the full frame, and the anti-contamination literal
// ---------------------------------------------------------------------------
void sectionOracle() {
    // The oracle's own history comes from running frame 1 through the same
    // independent rule starting from a zero state.
    const std::vector<uint16_t> f1 = makeFrame(kWall1, kWall2);
    const std::vector<double> zero_state(static_cast<size_t>(FRAME_W) * FRAME_H, 0.0);
    const EmaResult after1 = oracleEma(f1, zero_state);

    // Asymmetric history is what makes the section meaningful at all.
    const double h_wall = after1.state[static_cast<size_t>(kCy) * FRAME_W + (kCx + 40)];
    const double h_centre = after1.state[static_cast<size_t>(kCy) * FRAME_W + kCx];
    CHECK(std::abs(h_wall - oracleMeters(kWall1)) < 1e-9, "oracle history: wall holds frame-1 meters");
    CHECK(std::abs(h_centre - oracleMeters(kWall2)) < 1e-9, "oracle history: centre holds its own frame-1 meters");
    CHECK(std::abs(h_wall - h_centre) > 0.04, "fixture history is asymmetric between centre and neighbours");

    const std::vector<uint16_t> f2 = makeFrame(kWall2, kCentre);
    const EmaResult expect2 = oracleEma(f2, after1.state);

    // The oracle comparison runs single-threaded on purpose: single-threaded
    // schedule order is the worst case for an EMA that reads its own
    // destination, so a destination-reading implementation misses here too
    // instead of hiding behind a thread-dependent neighbour order.
    pinThreads(1);
    SignalConditioner sc;
    std::vector<uint16_t> p1 = f1;
    sc.applyDepthEmaForTests(p1, kLo, kHi);
    std::vector<uint16_t> p2 = f2;
    sc.applyDepthEmaForTests(p2, kLo, kHi);
    restoreThreads();

    int mismatches = 0, first_bad = -1;
    for (size_t i = 0; i < p2.size(); ++i) {
        if (p2[i] != expect2.out[i]) {
            ++mismatches;
            if (first_bad < 0) {
                first_bad = static_cast<int>(i);
                std::printf("  first mismatch at x=%d y=%d: product %u != oracle %u\n",
                            static_cast<int>(i) % FRAME_W, static_cast<int>(i) / FRAME_W,
                            static_cast<unsigned>(p2[i]), static_cast<unsigned>(expect2.out[i]));
            }
        }
    }
    CHECK(mismatches == 0, "EMA destination matches the independent oracle over all 307200 pixels");

    int state_mismatches = 0;
    for (size_t i = 0; i < expect2.state.size(); ++i) {
        const double got = sc.emaStateForTests()[i];
        if (std::abs(got - expect2.state[i]) > 1e-6) ++state_mismatches;
    }
    CHECK(state_mismatches == 0, "retained EMA history matches the independent oracle");

    const int centre = p2[static_cast<size_t>(kCy) * FRAME_W + kCx];
    const int wall = p2[static_cast<size_t>(kCy) * FRAME_W + (kCx + 40)];
    CHECK(wall == kExpectWall, "wall pixel blends to raw 534 (hand-derived)");
    CHECK(centre == kExpectCentre, "centre blends to raw 575 (hand-derived, source-buffer neighbours)");
    CHECK(centre != kContaminatedCentre,
          "centre is NOT raw 586: it did not reset off already-updated neighbour outputs");
    CHECK(std::abs(oracleMeters(kCentre) - oracleUsableMeters(kWall2)) <= kJumpReset,
          "fixture precondition: centre vs SOURCE neighbours stays under the reset threshold");

    // No EMA output is ever a sentinel or an out-of-band code.
    int invalid_out = 0;
    for (uint16_t v : p2) {
        if (v == 0) continue;
        if (!kfusion::sensor::isValidRawDepth(v)) { ++invalid_out; continue; }
        if (cpuDepthMeters(v, kLo, kHi) == 0.0f) ++invalid_out;
    }
    CHECK(invalid_out == 0, "EMA never emits an invalid or out-of-band raw code");
}

// ---------------------------------------------------------------------------
// C  repeated runs are bit-identical
// ---------------------------------------------------------------------------
void sectionRepeats() {
    uint64_t first = 0;
    bool same = true;
    for (int run = 0; run < 4; ++run) {
        int centre = 0, wall = 0;
        std::vector<float> state;
        const uint64_t digest = runSequence(centre, wall, &state);
        if (run == 0) first = digest;
        if (digest != first) same = false;
        if (centre != kExpectCentre || wall != kExpectWall) same = false;
    }
    CHECK(same, "four fresh two-frame runs are byte-identical (destination + float state digests)");
    std::printf("  ema digest (all threads, all repeats): %016llx\n",
                static_cast<unsigned long long>(first));
}

// ---------------------------------------------------------------------------
// D  thread-count equivalence
// ---------------------------------------------------------------------------
void sectionThreads() {
    uint64_t baseline = 0;
    bool identical = true;
    for (int n : {1, 2, 4, 1}) {
        pinThreads(n);
        int centre = 0, wall = 0;
        std::vector<float> state;
        const uint64_t digest = runSequence(centre, wall, &state);
        restoreThreads();
        std::printf("  OMP threads %d -> digest %016llx (centre %d, wall %d)\n", n,
                    static_cast<unsigned long long>(digest), centre, wall);
        if (n == 1 && baseline == 0) baseline = digest;
        if (digest != baseline) identical = false;
        if (centre != kExpectCentre || wall != kExpectWall) identical = false;
    }
    CHECK(identical, "threads 1 / 2 / 4 produce identical destination and state digests");
}

// ---------------------------------------------------------------------------
// E  reset lifecycle
// ---------------------------------------------------------------------------
void sectionResetLifecycle() {
    auto expect_identity_on_reset = [](SignalConditioner& sc, const char* what) {
        std::vector<uint16_t> f = makeFrame(kWall2, kCentre);
        const std::vector<uint16_t> input = f;
        sc.applyDepthEmaForTests(f, kLo, kHi);
        int bad = 0;
        for (size_t i = 0; i < f.size(); ++i) {
            if (f[i] != input[i]) ++bad;
        }
        CHECK(bad == 0, what);
    };

    SignalConditioner a;
    std::vector<uint16_t> warm = makeFrame(kWall1, kWall2);
    a.applyDepthEmaForTests(warm, kLo, kHi);
    bool history_built = false;
    for (float v : a.emaStateForTests()) {
        if (v > 0.0f) { history_built = true; break; }
    }
    CHECK(history_built, "a warm run leaves non-zero per-pixel history");

    a.resetEMA();
    bool all_zero = true;
    for (float v : a.emaStateForTests()) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        if (bits != 0u) { all_zero = false; break; }
    }
    CHECK(all_zero, "resetEMA() clears every history slot to exactly +0.0f");
    expect_identity_on_reset(a, "after resetEMA() the next frame is the identity on in-band raws");

    SignalConditioner b;
    b.applyDepthEmaForTests(warm, kLo, kHi);
    b.reset();
    bool all_zero_after_reset = true;
    for (float v : b.emaStateForTests()) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        if (bits != 0u) { all_zero_after_reset = false; break; }
    }
    CHECK(all_zero_after_reset, "reset() clears the EMA history as well");
    expect_identity_on_reset(b, "after reset() the next frame is the identity on in-band raws");

    // A third path: an invalid frame must wipe history at those pixels only.
    SignalConditioner c;
    std::vector<uint16_t> f = makeFrame(kWall2, kCentre);
    c.applyDepthEmaForTests(f, kLo, kHi);
    std::vector<uint16_t> poison = makeFrame(kWall2, kCentre);
    poison[static_cast<size_t>(kCy) * FRAME_W + kCx] = 0;      // hole
    poison[static_cast<size_t>(kCy) * FRAME_W + (kCx + 5)] = 2047;  // sentinel
    poison[static_cast<size_t>(kCy) * FRAME_W + (kCx + 9)] = 1500;  // out of band
    c.applyDepthEmaForTests(poison, kLo, kHi);
    CHECK(c.emaStateForTests()[static_cast<size_t>(kCy) * FRAME_W + kCx] == 0.0f,
          "a raw 0 pixel resets its own history to 0.0f");
    CHECK(c.emaStateForTests()[static_cast<size_t>(kCy) * FRAME_W + (kCx + 5)] == 0.0f,
          "a raw 2047 pixel resets its own history to 0.0f");
    CHECK(c.emaStateForTests()[static_cast<size_t>(kCy) * FRAME_W + (kCx + 9)] == 0.0f,
          "an out-of-band pixel resets its own history to 0.0f");
    CHECK(poison[static_cast<size_t>(kCy) * FRAME_W + kCx] == 0,
          "a raw 0 pixel passes through as raw 0 (never invented)");
    CHECK(poison[static_cast<size_t>(kCy) * FRAME_W + (kCx + 5)] == 2047,
          "a raw 2047 pixel passes through as raw 2047 (never demoted to a hole code)");
    CHECK(poison[static_cast<size_t>(kCy) * FRAME_W + (kCx + 9)] == 1500,
          "an out-of-band raw passes through unchanged (the band gate lives at the meters boundary)");
    CHECK(c.emaStateForTests()[static_cast<size_t>(kCy) * FRAME_W + (kCx + 40)] > 0.0f,
          "unrelated pixels keep their history");
}

} // namespace

int main() {
    sectionOracle();
    sectionRepeats();
    sectionThreads();
    sectionResetLifecycle();

    if (g_failures == 0) {
        std::printf("depth_ema_determinism_contract: PASS (%d checks, centre raw %d, wall raw %d)\n",
                    g_checks, kExpectCentre, kExpectWall);
        return 0;
    }
    std::printf("depth_ema_determinism_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
