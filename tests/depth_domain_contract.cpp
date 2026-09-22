// depth_domain_contract (big-fix Todo 20): CPU-only contract for the raw depth
// predicate, the configured meter band, the reciprocal-pole hazards and the
// hole-fill sentinel rule. Public CPU API only (sensor/DepthValidity.h +
// buildFrameData + the AZU_PIPELINE_TEST_SEAM depth stages + the
// FusionHyperparams owner): no device, display, GPU, sensor, thread timing,
// sleep or filesystem.
//
// Every expected number is an independent oracle written in double precision in
// this file from the canonical formula in docs/CANONICAL_SEMANTICS.md, never a
// product-code snapshot and never the product helper re-called. Sections:
//   A  configured band owner: min_depth > 0 (the second half of the
//      "negative meters can never reach geometry" guarantee) and min < max
//   B  raw predicate over the whole uint16 domain: invalid iff raw == 0 ||
//      raw >= 2047, and an invalid raw converts to exactly +0.0f
//   C  band gate over every raw 1..2046: the ACCEPTED SET is exactly the raws
//      whose double-precision meters land in [min,max] (954 of them, raw
//      1..954 for the default band), each accepted value matches the double
//      oracle, every rejected value is exactly 0.0f
//   D  band movement: below-minimum and above-maximum conversions are rejected
//      at the new band, and neither is snapped to a wall (a wall value is
//      asserted ABSENT, not merely "not equal to the oracle")
//   E  pole/negative sweep: raw 1084 (+533 m), 1085 (-836 m), 1500, 2046 all
//      reject; across all 2046 raws the output is never negative, never NaN,
//      never infinite
//   F  buildFrameData (the boundary that feeds vertices/normals/pyramid/ICP/
//      TSDF): rejected pixels are exactly 0.0f with a zero vertex, accepted
//      pixels match the oracle and the documented back-projection
//   G  the meters -> raw encoder: out-of-band and non-finite input yields raw 0
//      instead of the old clamp (raw 1 / raw 2046 are asserted ABSENT), and the
//      in-band round trip is the identity over every accepted raw
//   H  hole fill: only raw 0 is fillable; raw 2047 and an out-of-band measured
//      raw are untouched; a raw 0 hole ringed only by sentinels stays unfilled;
//      a filled value equals an independent inverse-distance-weighted oracle;
//      no non-hole pixel is modified
#include "app/FusionHyperparams.h"
#include "sensor/DepthValidity.h"
#include "sensor/FrameData.h"
#include "sensor/KinectSensor.h"
#include "sensor/SignalConditioner.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "depth_domain_contract must be compiled with AZU_PIPELINE_TEST_SEAM"
#endif

namespace {

using kfusion::app::FusionHyperparams;
using kfusion::sensor::buildFrameData;
using kfusion::sensor::cpuDepthMeters;
using kfusion::sensor::cpuDepthMetersToRaw;
using kfusion::sensor::FRAME_H;
using kfusion::sensor::FRAME_W;
using kfusion::sensor::isDepthMetersInBand;
using kfusion::sensor::isInvalidRawDepth;
using kfusion::sensor::isUsableRawDepth;
using kfusion::sensor::isValidRawDepth;
using kfusion::sensor::rawDepthToMeters;
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

// Independent double-precision oracle for the canonical Kinect v1 curve,
// written from docs/CANONICAL_SEMANTICS.md rather than from the product header.
constexpr double kCurveA = -0.0030711016;
constexpr double kCurveB = 3.3309495161;

double oracleMeters(int raw) {
    return 1.0 / (static_cast<double>(raw) * kCurveA + kCurveB);
}

bool oracleRawInvalid(int raw) { return raw == 0 || raw >= 2047; }

bool oracleInBand(double meters, double lo, double hi) {
    return std::isfinite(meters) && meters >= lo && meters <= hi;
}

// Exact +0.0f (bit pattern 0), so a -0.0f or a denormal "close enough" cannot
// pass as the canonical rejected value.
bool isExactPositiveZero(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits == 0u;
}

uint32_t bitsOf(float v) {
    uint32_t b = 0;
    std::memcpy(&b, &v, sizeof(b));
    return b;
}

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();

// ---------------------------------------------------------------------------
// A  the configured band owner is a positive, ordered interval
// ---------------------------------------------------------------------------
void sectionBandOwner(float& lo, float& hi) {
    const FusionHyperparams hp = FusionHyperparams::defaults();
    lo = hp.min_depth;
    hi = hp.max_depth;
    CHECK(lo > 0.0f, "configured min_depth is strictly positive (a negative band would admit negative meters)");
    CHECK(hi > lo, "configured max_depth exceeds min_depth");
    CHECK(std::isfinite(lo) && std::isfinite(hi), "configured band is finite");

    // The predicate itself, in isolation, over a hand-built table: the
    // conjunction finite AND >= min AND <= max.
    CHECK(isDepthMetersInBand(lo, lo, hi), "band is inclusive at min");
    CHECK(isDepthMetersInBand(hi, lo, hi), "band is inclusive at max");
    CHECK(!isDepthMetersInBand(lo - 1e-6f, lo, hi), "just below min is out of band");
    CHECK(!isDepthMetersInBand(hi + 1e-6f, lo, hi), "just above max is out of band");
    CHECK(!isDepthMetersInBand(kNaN, lo, hi), "NaN is never in band");
    CHECK(!isDepthMetersInBand(kInf, lo, hi), "+Inf is never in band");
    CHECK(!isDepthMetersInBand(-kInf, lo, hi), "-Inf is never in band");
    CHECK(!isDepthMetersInBand(-0.5f, lo, hi), "a negative distance is never in band");
}

// ---------------------------------------------------------------------------
// B  raw predicate over the whole uint16 domain
// ---------------------------------------------------------------------------
void sectionRawPredicate() {
    int mismatch = 0;
    for (int raw = 0; raw <= 65535; ++raw) {
        const bool invalid = isInvalidRawDepth(static_cast<uint16_t>(raw));
        if (invalid != oracleRawInvalid(raw)) ++mismatch;
        if (isValidRawDepth(static_cast<uint16_t>(raw)) != !oracleRawInvalid(raw)) ++mismatch;
        if (isUsableRawDepth(static_cast<uint16_t>(raw), 0.30f, 2.50f) && oracleRawInvalid(raw)) ++mismatch;
    }
    CHECK(mismatch == 0, "raw predicate == (raw == 0 || raw >= 2047) across all 65536 codes");

    CHECK(isExactPositiveZero(rawDepthToMeters(0)), "raw 0 -> exactly +0.0f");
    CHECK(isExactPositiveZero(rawDepthToMeters(2047)), "raw 2047 -> exactly +0.0f");
    CHECK(isExactPositiveZero(rawDepthToMeters(2048)), "raw 2048 -> exactly +0.0f");
    CHECK(isExactPositiveZero(rawDepthToMeters(65535)), "raw 65535 -> exactly +0.0f");
    CHECK(isExactPositiveZero(cpuDepthMeters(0, 0.3f, 2.5f)), "band gate: raw 0 -> exactly +0.0f");
    CHECK(isExactPositiveZero(cpuDepthMeters(2047, 0.3f, 2.5f)), "band gate: raw 2047 -> exactly +0.0f");
    CHECK(isExactPositiveZero(cpuDepthMeters(2048, 0.3f, 2.5f)), "band gate: raw > 2047 -> exactly +0.0f");

    // Valid raw boundary codes still convert (they are codes, not sentinels).
    CHECK(rawDepthToMeters(1) > 0.0f, "raw 1 still converts");
    CHECK(rawDepthToMeters(2) > 0.0f, "raw 2 still converts");
    CHECK(rawDepthToMeters(2046) != 0.0f, "raw 2046 still converts (its meters are negative, see section E)");
    CHECK(rawDepthToMeters(1) < rawDepthToMeters(2), "curve is monotonic below the pole");
    CHECK(std::abs(rawDepthToMeters(1) - static_cast<float>(oracleMeters(1))) < 1e-6f,
          "raw 1 meters match the double oracle");
}

// ---------------------------------------------------------------------------
// C  band gate over every raw 1..2046
// ---------------------------------------------------------------------------
void sectionBandGate(float lo, float hi) {
    int accepted = 0;
    int set_mismatch = 0;
    int value_mismatch = 0;
    int reject_mismatch = 0;
    int first_accepted = -1, last_accepted = -1;
    for (int raw = 1; raw < 2047; ++raw) {
        const double expect = oracleMeters(raw);
        const bool should_accept = oracleInBand(expect, lo, hi);
        const float got = cpuDepthMeters(static_cast<uint16_t>(raw), lo, hi);
        const bool accepted_now = (got != 0.0f);
        if (accepted_now != should_accept) ++set_mismatch;
        if (accepted_now) {
            ++accepted;
            if (first_accepted < 0) first_accepted = raw;
            last_accepted = raw;
            // float32 evaluation of the same curve: relative tolerance 1e-6.
            const double tol = 1e-6 * (std::abs(expect) + 1.0);
            if (std::abs(static_cast<double>(got) - expect) > tol) ++value_mismatch;
        } else if (!isExactPositiveZero(got)) {
            ++reject_mismatch;
        }
    }
    CHECK(set_mismatch == 0, "accepted raw set == { raw : oracleMeters(raw) in [min,max] }");
    CHECK(value_mismatch == 0, "every accepted conversion matches the independent double oracle");
    CHECK(reject_mismatch == 0, "every rejected conversion is exactly +0.0f");
    CHECK(accepted == 954, "the default band accepts exactly raw 1..954 (954 codes)");
    CHECK(first_accepted == 1 && last_accepted == 954, "accepted raw span is 1..954");

    // Spot values, hand-computed from the oracle.
    const double m954 = oracleMeters(954);
    const double m955 = oracleMeters(955);
    CHECK(m954 <= 2.50 && m954 > 2.49, "oracle: raw 954 is the last code inside the default band");
    CHECK(m955 > 2.50, "oracle: raw 955 has already left the default band");
    CHECK(cpuDepthMeters(954, lo, hi) != 0.0f, "raw 954 accepted");
    CHECK(isExactPositiveZero(cpuDepthMeters(955, lo, hi)), "raw 955 rejected (above max_depth)");
}

// ---------------------------------------------------------------------------
// D  band movement, with wall values asserted absent
// ---------------------------------------------------------------------------
void sectionNoWallClamp() {
    const float lo = 0.30f, hi = 2.50f;

    // Below minimum at a tighter band: raw 1 (0.30049 m) leaves [0.31, 2.5].
    const float below = cpuDepthMeters(1, 0.31f, hi);
    CHECK(isExactPositiveZero(below), "below-minimum conversion is rejected");
    CHECK(below != 0.31f, "below-minimum output is NOT clamped up to min_depth");
    CHECK(below != lo, "below-minimum output is NOT clamped to the default min_depth");
    CHECK(bitsOf(below) != bitsOf(rawDepthToMeters(1)), "below-minimum output is not the raw conversion either");

    // Above maximum: raw 955 (2.5123 m) leaves the default band.
    const float above = cpuDepthMeters(955, lo, hi);
    CHECK(isExactPositiveZero(above), "above-maximum conversion is rejected");
    CHECK(above != hi, "above-maximum output is NOT clamped down to max_depth");

    // The historical wall codes: raw 1 and raw 2046 are what the old clamp
    // produced for a low-side and a high-side overflow. Neither may come back.
    CHECK(cpuDepthMetersToRaw(0.29f, lo, hi) != 1, "just-below-min meters do NOT encode to raw 1");
    CHECK(cpuDepthMetersToRaw(2.60f, lo, hi) != 2046, "just-above-max meters do NOT encode to raw 2046");
    CHECK(isExactPositiveZero(rawDepthToMeters(2046)) == false,
          "raw 2046 is a code, not a sentinel: its meters are non-zero and negative");

    // A tighter band rejects more raws and never invents a value.
    int tightened = 0, leaked = 0;
    for (int raw = 1; raw < 955; ++raw) {
        const float v = cpuDepthMeters(static_cast<uint16_t>(raw), 1.0f, 1.5f);
        if (v == 0.0f) continue;
        ++tightened;
        if (v < 1.0f || v > 1.5f) ++leaked;
    }
    const double lo15 = oracleMeters(800);
    CHECK(lo15 > 1.0 && lo15 < 1.5, "oracle: raw 800 sits inside the tight [1.0,1.5] band");
    CHECK(oracleMeters(700) < 1.0, "oracle: raw 700 falls below the tight band and must be rejected");
    CHECK(tightened > 0, "the tight band still accepts something");
    CHECK(leaked == 0, "no accepted value ever escapes the tight band");
}

// ---------------------------------------------------------------------------
// E  reciprocal pole and negative-meter sweep
// ---------------------------------------------------------------------------
void sectionPoleAndNegatives(float lo, float hi) {
    CHECK(oracleMeters(1084) > 500.0, "oracle: raw 1084 is +533 m, the runaway approach to the pole");
    CHECK(oracleMeters(1085) < -800.0, "oracle: raw 1085 is -836 m, negative right after the pole");
    CHECK(oracleMeters(2046) < 0.0, "oracle: raw 2046 is still negative");
    CHECK(isExactPositiveZero(cpuDepthMeters(1084, lo, hi)), "near-pole +533 m rejected");
    CHECK(isExactPositiveZero(cpuDepthMeters(1085, lo, hi)), "near-pole -836 m rejected");
    CHECK(isExactPositiveZero(cpuDepthMeters(1500, lo, hi)), "raw 1500 (-0.784 m) rejected");
    CHECK(isExactPositiveZero(cpuDepthMeters(2046, lo, hi)), "raw 2046 (-0.339 m) rejected");

    int negative = 0, nonfinite = 0, above_max = 0, below_min = 0;
    for (int raw = 0; raw <= 65535; ++raw) {
        const float v = cpuDepthMeters(static_cast<uint16_t>(raw), lo, hi);
        if (!std::isfinite(v)) ++nonfinite;
        if (std::signbit(v)) ++negative;  // -0.0f counts here and is also a defect
        if (v > hi) ++above_max;
        if (v != 0.0f && v < lo) ++below_min;
    }
    CHECK(nonfinite == 0, "no NaN/Inf depth over the whole uint16 raw domain");
    CHECK(negative == 0, "no negative (or -0.0f) depth over the whole uint16 raw domain");
    CHECK(above_max == 0, "nothing exceeds max_depth");
    CHECK(below_min == 0, "nothing lands in (0, min_depth)");
}

// ---------------------------------------------------------------------------
// F  buildFrameData boundary
// ---------------------------------------------------------------------------
void sectionFrameDataBoundary(float lo, float hi) {
    struct Case { uint16_t raw; bool expect_ok; const char* what; };
    const Case cases[] = {
        {0, false, "raw 0"},          {2047, false, "raw 2047"},
        {2048, false, "raw 2048"},    {1, true, "raw 1"},
        {954, true, "raw 954"},       {955, false, "raw 955 above max"},
        {1084, false, "raw 1084 pole"}, {1085, false, "raw 1085 negative"},
        {1500, false, "raw 1500 negative"}, {2046, false, "raw 2046 negative"},
        {700, true, "raw 700"},       {500, true, "raw 500"},
    };
    const int nc = static_cast<int>(sizeof(cases) / sizeof(cases[0]));

    std::vector<uint16_t> raw_depth(static_cast<size_t>(FRAME_W) * FRAME_H, 700);
    std::vector<uint8_t> raw_rgb(static_cast<size_t>(FRAME_W) * FRAME_H * 3, 0);
    for (int i = 0; i < nc; ++i) raw_depth[static_cast<size_t>(i)] = cases[i].raw;

    kfusion::sensor::FrameData fd;
    buildFrameData(raw_depth.data(), raw_rgb.data(), fd, lo, hi);

    int bad = 0;
    for (int i = 0; i < nc; ++i) {
        const Case& c = cases[i];
        const float d = fd.depth_meters[static_cast<size_t>(i)];
        const bool ok = (d != 0.0f);
        if (ok != c.expect_ok) {
            ++bad;
            std::printf("  case %s raw %u -> %.9f (ok=%d expected=%d)\n", c.what,
                        static_cast<unsigned>(c.raw), d, static_cast<int>(ok),
                        static_cast<int>(c.expect_ok));
            continue;
        }
        if (!ok) {
            // Rejected: exactly zero, and explicitly NOT a wall value.
            if (!isExactPositiveZero(d)) { ++bad; std::printf("  %s not exact +0.0f\n", c.what); }
            if (d == lo || d == hi) { ++bad; std::printf("  %s was clamped to a wall\n", c.what); }
            if (!fd.vertices[static_cast<size_t>(i)].isZero()) {
                ++bad; std::printf("  %s left a non-zero vertex\n", c.what);
            }
            if (fd.isValid(static_cast<int>(i), 0)) { ++bad; std::printf("  %s reported valid\n", c.what); }
        } else {
            const double expect = oracleMeters(c.raw);
            const double tol = 1e-6 * (std::abs(expect) + 1.0);
            if (std::abs(static_cast<double>(d) - expect) > tol) {
                ++bad; std::printf("  %s meters %.9f != oracle %.9f\n", c.what, d, expect);
            }
            const int x = i % FRAME_W, y = i / FRAME_W;
            const double ex = (x - kfusion::sensor::CX) / static_cast<double>(kfusion::sensor::FX) * expect;
            const double ey = (y - kfusion::sensor::CY) / static_cast<double>(kfusion::sensor::FY) * expect;
            const Eigen::Vector3f& v = fd.vertices[static_cast<size_t>(i)];
            if (std::abs(static_cast<double>(v.x()) - ex) > 1e-5 ||
                std::abs(static_cast<double>(v.y()) - ey) > 1e-5 ||
                std::abs(static_cast<double>(v.z()) - expect) > 1e-6) {
                ++bad; std::printf("  %s vertex does not match back-projection\n", c.what);
            }
        }
    }
    CHECK(bad == 0, "buildFrameData honors the raw predicate + band with exact-zero rejection");

    int negative = 0;
    for (size_t i = 0; i < fd.depth_meters.size(); ++i) {
        if (std::signbit(fd.depth_meters[i]) || fd.depth_meters[i] < 0.0f) ++negative;
    }
    CHECK(negative == 0, "buildFrameData emits no negative depth into the geometry path");
}

// ---------------------------------------------------------------------------
// G  meters -> raw encoder: rejection, no wall, exact round trip
// ---------------------------------------------------------------------------
void sectionEncoder() {
    const float lo = 0.30f, hi = 2.50f;
    const float rejects[] = {-1.0f, -0.3387f, 0.0f, 0.01f, 0.29f, 2.51f, 3.0f, 1000.0f,
                             kNaN, kInf, -kInf, std::numeric_limits<float>::lowest(),
                             std::numeric_limits<float>::max()};
    int bad = 0;
    for (float v : rejects) {
        const uint16_t code = cpuDepthMetersToRaw(v, lo, hi);
        if (code != 0) { ++bad; std::printf("  %g encoded to %u\n", v, static_cast<unsigned>(code)); }
    }
    CHECK(bad == 0, "out-of-band / non-finite meters encode to raw 0, never to a code");

    // In-band round trip must be the identity over every accepted raw code.
    int rt_bad = 0, checked = 0;
    for (int raw = 1; raw < 2047; ++raw) {
        const float m = cpuDepthMeters(static_cast<uint16_t>(raw), lo, hi);
        if (m == 0.0f) continue;
        ++checked;
        const uint16_t back = cpuDepthMetersToRaw(m, lo, hi);
        if (back != static_cast<uint16_t>(raw)) {
            ++rt_bad;
            if (rt_bad < 4) std::printf("  raw %d -> %.9f -> %u\n", raw, m, static_cast<unsigned>(back));
        }
    }
    CHECK(checked == 954, "round-trip sweep covers the 954 in-band codes");
    CHECK(rt_bad == 0, "encode(decode(raw)) == raw for every in-band raw");

    // The encoder can never emit the sentinel, whatever it is handed.
    int sentinel_hits = 0;
    for (int i = -2000; i <= 4000; ++i) {
        const float v = static_cast<float>(i) * 0.001f;
        const uint16_t code = cpuDepthMetersToRaw(v, lo, hi);
        if (code >= 2047) ++sentinel_hits;
        if (code != 0 && !isValidRawDepth(code)) ++sentinel_hits;
        if (code != 0) {
            const float decoded = cpuDepthMeters(code, lo, hi);
            if (decoded == 0.0f) ++sentinel_hits;  // encoded something that decodes out of band
        }
    }
    CHECK(sentinel_hits == 0, "encoder never emits an invalid raw code nor an out-of-band code");
}

// ---------------------------------------------------------------------------
// H  hole fill
// ---------------------------------------------------------------------------
// Independent oracle for the documented fill rule: inverse-distance weighted
// mean over the radius-2 window (centre excluded, reflect boundary), using only
// neighbours that are usable depth at the configured band, re-encoded through
// the same canonical curve. Written from the rule, not from the product code.
uint16_t oracleFill(const std::vector<uint16_t>& depth, int x, int y, int radius,
                    float lo, float hi) {
    auto reflect = [](int v, int max_v) {
        if (v < 0) return -v - 1;
        if (v >= max_v) return 2 * max_v - v - 1;
        return v;
    };
    double wsum = 0.0, acc = 0.0;
    int n = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const int sx = reflect(x + dx, FRAME_W);
            const int sy = reflect(y + dy, FRAME_H);
            const uint16_t raw = depth[static_cast<size_t>(sy) * FRAME_W + sx];
            if (oracleRawInvalid(raw)) continue;
            const double m = oracleMeters(raw);
            if (!oracleInBand(m, lo, hi)) continue;
            const double w = 1.0 / static_cast<double>(dx * dx + dy * dy + 1);
            acc += w * m;
            wsum += w;
            ++n;
        }
    }
    if (n == 0 || wsum <= 1e-6) return 0;  // nothing fillable
    const double mean = acc / wsum;
    if (!oracleInBand(mean, lo, hi)) return 0;
    const double raw = (1.0 / mean - kCurveB) / kCurveA;
    const long code = std::lround(raw);
    return (code >= 1 && code <= 2046) ? static_cast<uint16_t>(code) : 0;
}

void sectionHoleFill(float lo, float hi) {
    const int kHoleRadius = 2;  // documented kHoleFillRadius

    // Fixture: uniform valid wall, then four features placed far apart so their
    // radius-2 windows never overlap.
    auto makeFrame = []() {
        return std::vector<uint16_t>(static_cast<size_t>(FRAME_W) * FRAME_H, static_cast<uint16_t>(700));
    };
    const int hx = 100, hy = 100;   // plain raw-0 hole in a 700 wall
    const int sx = 200, sy = 100;   // raw 2047 sentinel in a 700 wall
    const int ox = 300, oy = 100;   // raw 1500 (valid code, out-of-band meters)
    const int ux = 400, uy = 100;   // raw-0 hole ringed only by sentinels / out-of-band
    const int mx = 500, my = 200;   // raw-0 hole with a deterministic usable/sentinel mix

    // (1) a raw 0 hole in a uniform wall is filled with the wall value
    {
        std::vector<uint16_t> d = makeFrame();
        d[static_cast<size_t>(hy) * FRAME_W + hx] = 0;
        std::vector<uint16_t> before = d;
        SignalConditioner sc;
        sc.fillDepthHolesForTests(d, lo, hi);
        const uint16_t expect = oracleFill(before, hx, hy, kHoleRadius, lo, hi);
        CHECK(expect == 700, "oracle: a hole in a uniform raw-700 wall refills to raw 700");
        CHECK(d[static_cast<size_t>(hy) * FRAME_W + hx] == expect, "raw 0 hole is filled with the oracle value");
        int touched = 0;
        for (size_t i = 0; i < d.size(); ++i) {
            if (static_cast<int>(i) == hy * FRAME_W + hx) continue;
            if (d[i] != before[i]) ++touched;
        }
        CHECK(touched == 0, "hole fill modifies no pixel other than the hole");
    }

    // (2) raw 2047 is never interpolated over, even though its meters are 0.0
    {
        std::vector<uint16_t> d = makeFrame();
        d[static_cast<size_t>(sy) * FRAME_W + sx] = 2047;
        SignalConditioner sc;
        sc.fillDepthHolesForTests(d, lo, hi);
        CHECK(d[static_cast<size_t>(sy) * FRAME_W + sx] == 2047, "raw 2047 is left untouched by hole fill");
    }

    // (3) a measured raw whose meters are out of band is not a hole either
    {
        std::vector<uint16_t> d = makeFrame();
        d[static_cast<size_t>(oy) * FRAME_W + ox] = 1500;  // -0.784 m
        SignalConditioner sc;
        sc.fillDepthHolesForTests(d, lo, hi);
        CHECK(d[static_cast<size_t>(oy) * FRAME_W + ox] == 1500,
              "an out-of-band measured raw is not treated as a fillable hole");
    }

    // (4) a hole whose whole window is sentinel / out-of-band stays a hole
    {
        std::vector<uint16_t> d = makeFrame();
        for (int dy = -3; dy <= 3; ++dy) {
            for (int dx = -3; dx <= 3; ++dx) {
                const int yy = uy + dy, xx = ux + dx;
                d[static_cast<size_t>(yy) * FRAME_W + xx] =
                    ((std::abs(dx) + std::abs(dy)) % 2) ? static_cast<uint16_t>(2047)
                                                          : static_cast<uint16_t>(1500);
            }
        }
        d[static_cast<size_t>(uy) * FRAME_W + ux] = 0;
        std::vector<uint16_t> before = d;
        SignalConditioner sc;
        sc.fillDepthHolesForTests(d, lo, hi);
        CHECK(oracleFill(before, ux, uy, kHoleRadius, lo, hi) == 0,
              "oracle: a sentinel-only window has no fill source");
        CHECK(d[static_cast<size_t>(uy) * FRAME_W + ux] == 0,
              "a hole ringed only by raw 2047 / out-of-band raws stays unfilled");
    }

    // (5) mixed usable + sentinel + hole window: value equals the oracle over
    //     the usable neighbours only (a hand-checked 4-neighbour case).
    {
        std::vector<uint16_t> d = makeFrame();
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                const int yy = my + dy, xx = mx + dx;
                static const uint16_t pattern[6] = {700, 500, 2047, 0, 1500, 548};
                d[static_cast<size_t>(yy) * FRAME_W + xx] =
                    pattern[(std::abs(dx) + std::abs(dy)) % 6];
            }
        }
        d[static_cast<size_t>(my) * FRAME_W + mx] = 0;
        std::vector<uint16_t> before = d;
        SignalConditioner sc;
        sc.fillDepthHolesForTests(d, lo, hi);
        const uint16_t expect = oracleFill(before, mx, my, kHoleRadius, lo, hi);
        CHECK(expect == 500, "oracle: the mixed window fills to raw 500 from its 4 usable neighbours");
        CHECK(d[static_cast<size_t>(my) * FRAME_W + mx] == expect,
              "mixed-window fill matches the independent inverse-distance-weighted oracle");
    }

    // (6) a checkerboard of two in-band walls fills to the weighted mean of both
    {
        std::vector<uint16_t> d = makeFrame();
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                const int yy = 300 + dy, xx = 100 + dx;
                d[static_cast<size_t>(yy) * FRAME_W + xx] =
                    ((std::abs(dx) + std::abs(dy)) % 2) ? static_cast<uint16_t>(548)
                                                         : static_cast<uint16_t>(500);
            }
        }
        d[300 * FRAME_W + 100] = 0;
        std::vector<uint16_t> before = d;
        SignalConditioner sc;
        sc.fillDepthHolesForTests(d, lo, hi);
        const uint16_t expect = oracleFill(before, 100, 300, kHoleRadius, lo, hi);
        CHECK(expect == 528, "oracle: the 500/548 checkerboard fills to raw 528");
        CHECK(d[300 * FRAME_W + 100] == expect, "checkerboard fill matches the oracle");
    }

    // (7) every value the fill WRITES is a valid in-band code, and the invalid
    //     codes the fixture planted (2047 holes-of-sentinels) survive untouched
    {
        std::vector<uint16_t> d = makeFrame();
        for (int y = 50; y < 60; ++y)
            for (int x = 50; x < 60; ++x)
                d[static_cast<size_t>(y) * FRAME_W + x] = ((x * 7 + y * 3) % 5 == 0) ? 0 : 2047;
        const std::vector<uint16_t> before = d;
        SignalConditioner sc;
        sc.fillDepthHolesForTests(d, lo, hi);
        int bad_writes = 0, planted_lost = 0, filled = 0;
        for (size_t i = 0; i < d.size(); ++i) {
            if (d[i] != before[i]) {
                // Only raw 0 pixels may move, and only to an in-band code.
                if (before[i] != 0) ++bad_writes;
                else if (!isValidRawDepth(d[i]) || cpuDepthMeters(d[i], lo, hi) == 0.0f) ++bad_writes;
                else ++filled;
            } else if (before[i] != 0 && !isValidRawDepth(before[i]) && isValidRawDepth(d[i])) {
                ++planted_lost;
            }
        }
        CHECK(bad_writes == 0, "hole fill only turns raw 0 into a valid in-band raw code");
        CHECK(planted_lost == 0, "hole fill leaves the planted 2047 sentinels exactly as it found them");
        CHECK(filled > 0, "holes bordering the raw 700 wall do get filled (section is not vacuous)");
    }
}

} // namespace

int main() {
    float lo = 0.0f, hi = 0.0f;
    sectionBandOwner(lo, hi);
    sectionRawPredicate();
    sectionBandGate(lo, hi);
    sectionNoWallClamp();
    sectionPoleAndNegatives(lo, hi);
    sectionFrameDataBoundary(lo, hi);
    sectionEncoder();
    sectionHoleFill(lo, hi);

    if (g_failures == 0) {
        std::printf("depth_domain_contract: PASS (%d checks, band [%.2f, %.2f], "
                    "954 in-band raws, 0 negative outputs)\n",
                    g_checks, lo, hi);
        return 0;
    }
    std::printf("depth_domain_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
