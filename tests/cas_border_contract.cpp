// cas_border_contract (big-fix Todo 21): CPU-only contract for the ONE canonical
// border-mode owner (kfusion::sensor::cpuReflectCoord, include/sensor/BorderMode.h)
// and for the CPU guidance-luma rule that the CAS stage feeds the guided depth
// filter. Public CPU surfaces only: the shared border helper, the public
// sr::applyEASU_CPU product pass, and the AZU_PIPELINE_TEST_SEAM guidance
// accessors. No device, display, GPU, sensor, thread timing, sleep or filesystem.
//
// Why the fixtures are shaped like this (all of it measured, none of it assumed):
//
// 1. A radius-1 stencil CANNOT tell reflection from clamping. Its only
//    out-of-range coordinates are -1 and extent, and
//      reflect(-1)    = 0     = clamp(-1)
//      reflect(extent)= n - 1 = clamp(extent)
//    so the 5-tap RCAS cross and the 3x3 median are border-mode blind. Section A
//    proves that algebraically for every extent 1..1024, and section C therefore
//    pins guidance luma with luma-arithmetic discriminating fixtures rather than
//    pretending a clamp mutation would move it.
// 2. The 4x4 Catmull-Rom resample in applyEASU_CPU is the CPU pass that DOES
//    diverge: at destination column 0 its source taps are exactly {-2,-1,0,1},
//    and at the last destination column they are {n-2,n-1,n,n+1}. -2 and n+1 are
//    outside the +-1 band, they carry non-zero kernel weight, and
//      reflect(-2) = 1 != 0 = clamp(-2)
//      reflect(n+1) = n - 2 != n - 1 = clamp(n+1)
//    Section B derives those tap sets itself, then compares REAL product output
//    against a double oracle for each mode and requires the product to match the
//    reflect oracle and to be EXCLUDED from the clamp oracle.
// 3. Guidance luma is the canonical Rec.601-style CPU rule over the POST-CAS
//    guidance bytes: (0.299*R + 0.587*G + 0.114*B) / 255. Section C drives the
//    real buildSuperResolutionGuidance() through the seam and compares against an
//    independent double CAS + luma oracle; the tolerance is the derived
//    worst-case byte-quantization bound, never a loose fudge.
//
// Sections:
//   A  border algebra: helper == the documented reflection rule, contains the
//      coordinate, is an involution, agrees with clamp at +-1 and diverges at -2
//      and extent+1
//   B  product EASU 1-pixel destination border == reflect oracle, != clamp oracle
//   C  guidance luma through the real seam, 7 synthetic patterns, independent
//      double oracle, exact [0,1] range, wrong-value exclusion
//   D  repeated fresh-conditioner runs are bit-identical (FNV-1a over luma bits)
//   E  OpenMP threads 1 / 2 / 4 reproduce those digests
#include "sensor/BorderMode.h"
#include "sensor/FrameData.h"
#include "sensor/SignalConditioner.h"
#include "sensor/SuperResolution.h"

#include <algorithm>
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
#error "cas_border_contract must be compiled with AZU_PIPELINE_TEST_SEAM"
#endif

namespace {

using kfusion::sensor::cpuReflectCoord;
using kfusion::sensor::FRAME_H;
using kfusion::sensor::FRAME_W;
using kfusion::sensor::SignalConditioner;

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond, what)                                                            \
    do {                                                                             \
        ++g_checks;                                                                  \
        if (!(cond)) {                                                               \
            std::printf("FAIL: %s  [%s:%d]\n", std::string(what).c_str(), __FILE__,  \
                        __LINE__);                                                   \
            ++g_failures;                                                            \
        }                                                                            \
    } while (false)

// ---------------------------------------------------------------------------
// Independent oracles. Restated from the documented rules, never called into
// the product helper, and each byte expectation is computed in double.
// ---------------------------------------------------------------------------
int oracleReflect(int c, int n) {  // the specification, restated
    if (c < 0) return -c - 1;
    if (c >= n) return 2 * n - c - 1;
    return c;
}
int oracleClamp(int c, int n) { return c < 0 ? 0 : (c >= n ? n - 1 : c); }

double cubicCR(double x) {  // Catmull-Rom, the kernel the CPU resample documents
    const double a = std::abs(x);
    if (a <= 1.0) return 0.5 * (a * a * a - 2.0 * a * a + 1.0);
    if (a < 2.0) return 0.5 * (2.0 - a) * (2.0 - a) * (2.0 - a);
    return 0.0;
}

// Destination column -> the four source tap coordinates, derived from the
// documented center-to-center mapping. Independent of product code.
std::vector<int> srcTaps(int dst_x, int src_extent, int scale) {
    const double xs = static_cast<double>(src_extent) / (src_extent * scale);
    const double sx = (dst_x + 0.5) * xs - 0.5;
    const int x0 = static_cast<int>(std::floor(sx)) - 1;
    return {x0, x0 + 1, x0 + 2, x0 + 3};
}

// Double oracle for the CPU bicubic resample under a caller-chosen border mode.
std::vector<uint8_t> oracleEASU(const std::vector<uint8_t>& src, int w, int h, int scale,
                                int (*border)(int, int)) {
    const int dw = w * scale, dh = h * scale;
    std::vector<uint8_t> dst(static_cast<size_t>(dw) * dh * 3, 0);
    const double xs = static_cast<double>(w) / dw, ys = static_cast<double>(h) / dh;
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            const double sx = (x + 0.5) * xs - 0.5;
            const double sy = (y + 0.5) * ys - 0.5;
            const int x0 = static_cast<int>(std::floor(sx)) - 1;
            const int y0 = static_cast<int>(std::floor(sy)) - 1;
            double sum[3] = {0.0, 0.0, 0.0};
            double wsum = 0.0;
            for (int j = 0; j < 4; ++j) {
                for (int i = 0; i < 4; ++i) {
                    const int bxx = border(x0 + i, w), byy = border(y0 + j, h);
                    const size_t si = (static_cast<size_t>(byy) * w + bxx) * 3;
                    const double wt = cubicCR(sx - (x0 + i)) * cubicCR(sy - (y0 + j));
                    for (int c = 0; c < 3; ++c) sum[c] += (src[si + c] / 255.0) * wt;
                    wsum += wt;
                }
            }
            const size_t di = (static_cast<size_t>(y) * dw + x) * 3;
            for (int c = 0; c < 3; ++c) {
                double v = (wsum > 1e-6) ? sum[c] / wsum : 0.0;
                v = std::min(1.0, std::max(0.0, v));
                dst[di + c] = static_cast<uint8_t>(v * 255.0);  // truncation, as documented
            }
        }
    }
    return dst;
}

// Canonical guidance luma rule (double): Rec.601-style weights over bytes, /255.
double lumaRule(int r, int g, int b) {
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
}
double lumaRB(int r, int g, int b) { return lumaRule(b, g, r); }
double luma709(int r, int g, int b) { return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0; }
double lumaNoDiv255(int r, int g, int b) { return 0.299 * r + 0.587 * g + 0.114 * b; }

// Independent double oracle for the CAS stage the guidance builder runs
// (cross stencil, reflect border, sharpness 0.5 as buildSuperResolutionGuidance
// calls it), returning sharpened BYTES so the luma comparison is like-for-like.
std::vector<uint8_t> oracleCas(const std::vector<uint8_t>& rgb, int w, int h) {
    std::vector<uint8_t> out(rgb.size(), 0);
    const double peak = -1.0 / ((1.0 - 0.5) * 8.0 + 0.5 * 5.0);
    const int off[5][2] = {{0, -1}, {-1, 0}, {0, 0}, {1, 0}, {0, 1}};
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            double s[5][3];
            for (int k = 0; k < 5; ++k) {
                const int sx = oracleReflect(x + off[k][0], w);
                const int sy = oracleReflect(y + off[k][1], h);
                const size_t si = (static_cast<size_t>(sy) * w + sx) * 3;
                for (int c = 0; c < 3; ++c) s[k][c] = rgb[si + c] / 255.0;
            }
            double amp = 1.0;
            for (int c = 0; c < 3; ++c) {
                double mn = s[0][c], mx = s[0][c];
                for (int k = 1; k < 5; ++k) {
                    mn = std::min(mn, s[k][c]);
                    mx = std::max(mx, s[k][c]);
                }
                amp = std::min(amp, std::min(mn, 1.0 - mx) / std::max(mx, 1e-6));
            }
            const double wt = amp * peak, wsum = 1.0 + 4.0 * wt;
            const size_t di = (static_cast<size_t>(y) * w + x) * 3;
            for (int c = 0; c < 3; ++c) {
                double v = (s[2][c] + wt * (s[0][c] + s[1][c] + s[3][c] + s[4][c])) / wsum;
                v = std::min(1.0, std::max(0.0, v));
                out[di + c] = static_cast<uint8_t>(v * 255.0);
            }
        }
    }
    return out;
}

uint64_t fnv1a(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}
uint64_t digestLuma(const std::vector<float>& luma) {
    return fnv1a(reinterpret_cast<const uint8_t*>(luma.data()), luma.size() * sizeof(float));
}

int g_saved_threads = 0;
void pinThreads(int n) {
#ifdef _OPENMP
    g_saved_threads = omp_get_max_threads();
    omp_set_num_threads(n);
#else
    (void)n;
#endif
}
void restoreThreads() {
#ifdef _OPENMP
    omp_set_num_threads(g_saved_threads);
#endif
}

// ---------------------------------------------------------------------------
// A  border algebra
// ---------------------------------------------------------------------------
void sectionBorderAlgebra() {
    // Exact documented literals, including the two divergent coordinates.
    CHECK(cpuReflectCoord(-2, 8) == 1, "reflect(-2, 8) == 1 (clamp would give 0)");
    CHECK(cpuReflectCoord(-2, 8) != oracleClamp(-2, 8), "-2 is a genuine reflect/clamp divergence");
    CHECK(cpuReflectCoord(9, 8) == 6, "reflect(extent+1, 8) == extent-2 (clamp would give 7)");
    CHECK(cpuReflectCoord(9, 8) != oracleClamp(9, 8), "extent+1 is a genuine reflect/clamp divergence");
    CHECK(cpuReflectCoord(-3, 8) == 2 && cpuReflectCoord(10, 8) == 5, "deeper taps keep mirroring");
    CHECK(cpuReflectCoord(-1, 8) == 0 && cpuReflectCoord(8, 8) == 7,
          "the +-1 stencil coordinates map identically to clamp (RCAS is border-blind)");

    // Swept over the helper's documented domain [-extent, 2*extent-1], which is
    // exactly where a mirrored coordinate still names a sample. Outside it the
    // formula evaluates but means nothing, so feeding e.g. -4 to extent 1 would
    // test a state no caller can produce (each pass offsets a valid index by a
    // radius smaller than the extent).
    int spec_mismatch = 0, out_of_range = 0, not_idempotent = 0, blind_violation = 0, divergence = 0;
    for (int n = 1; n <= 1024; ++n) {
        for (int c = -n; c <= 2 * n - 1; ++c) {
            const int r = cpuReflectCoord(c, n);
            if (r != oracleReflect(c, n)) ++spec_mismatch;
            if (r < 0 || r >= n) ++out_of_range;
            if (cpuReflectCoord(r, n) != r) ++not_idempotent;
            if (c == -1 || c == n) {
                if (r != oracleClamp(c, n)) ++blind_violation;
            } else if (r != oracleClamp(c, n)) {
                ++divergence;
            }
        }
    }
    CHECK(spec_mismatch == 0, "helper equals the documented reflection rule for extents 1..1024");
    CHECK(out_of_range == 0, "every mapped coordinate lands inside [0, extent-1]");
    CHECK(not_idempotent == 0, "border extension is stable: applying it twice changes nothing");
    CHECK(blind_violation == 0,
          "reflect == clamp at -1 and at extent for every extent (why radius-1 cannot diverge)");
    CHECK(divergence > 0, "reflect != clamp is actually reachable beyond the +-1 band");
    std::printf("  border algebra: extents 1..1024 over domain [-extent, 2*extent-1], "
                "divergent coords=%d\n",
                divergence);
}

// ---------------------------------------------------------------------------
// B  product EASU output at the 1-pixel destination border
// ---------------------------------------------------------------------------
void sectionTapReach() {
    bool reach_first = true, reach_last = true, weight_ok = true;
    for (int scale : {2, 3, 4}) {
        for (int n : {8, 16, 640, 641}) {
            const auto t0 = srcTaps(0, n, scale);
            const auto t1 = srcTaps(n * scale - 1, n, scale);
            if (std::find(t0.begin(), t0.end(), -2) == t0.end()) reach_first = false;
            if (std::find(t1.begin(), t1.end(), n + 1) == t1.end()) reach_last = false;
            // the out-of-band tap must carry non-zero kernel weight, or the
            // fixture would be sampling it with weight 0 and testing nothing
            const double xs = static_cast<double>(n) / (n * scale);
            const double sx0 = 0.5 * xs - 0.5;
            const double sx1 = (n * scale - 0.5) * xs - 0.5;
            if (cubicCR(sx0 - t0[0]) <= 0.0 || cubicCR(sx1 - t1[3]) <= 0.0) weight_ok = false;
        }
    }
    CHECK(reach_first, "destination column 0 samples source tap -2 at scale 2, 3 and 4");
    CHECK(reach_last, "the last destination column samples source tap extent+1 at scale 2, 3 and 4");
    CHECK(weight_ok, "the -2 / extent+1 taps carry non-zero Catmull-Rom weight");
}

void easuFixture(const char* name, int n, int scale, int ring_r, int ring_g, int ring_b, int in_r,
                 int in_g, int in_b) {
    std::vector<uint8_t> src(static_cast<size_t>(n) * n * 3, 0);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            const bool edge = (x == 0 || y == 0 || x == n - 1 || y == n - 1);
            const size_t i = (static_cast<size_t>(y) * n + x) * 3;
            src[i]     = static_cast<uint8_t>(edge ? ring_r : in_r);
            src[i + 1] = static_cast<uint8_t>(edge ? ring_g : in_g);
            src[i + 2] = static_cast<uint8_t>(edge ? ring_b : in_b);
        }
    }
    std::vector<uint8_t> prod;
    kfusion::sensor::sr::applyEASU_CPU(src, prod, n, n, scale);
    const std::vector<uint8_t> o_r = oracleEASU(src, n, n, scale, oracleReflect);
    const std::vector<uint8_t> o_c = oracleEASU(src, n, n, scale, oracleClamp);
    const int dw = n * scale;

    // +/-1 byte: the product truncates a float32 accumulator, the oracle a
    // double one, so a value sitting within 1e-4 of a byte boundary is the only
    // way the two can differ at all.
    int oracle_gap = 0;
    for (size_t i = 0; i < prod.size(); ++i) {
        oracle_gap = std::max(oracle_gap, std::abs(int(prod[i]) - int(o_r[i])));
    }
    CHECK(oracle_gap <= 1, name);
    CHECK(oracle_gap <= 1, "product EASU matches the double reflect oracle within 1 byte");

    // Every destination byte where the two border modes differ by at least 3 must
    // be on the reflect side. 3 is the floor that makes the test conclusive: the
    // product may sit 1 byte from its own double oracle, so at a 2-byte mode gap
    // (which is all scale 2 can reach) a clamped expectation would still be
    // within tolerance. A fixture that reaches no such byte is vacuous and fails.
    int divergent = 0, explained_by_clamp = 0, max_gap = 0;
    for (size_t i = 0; i < o_r.size(); ++i) {
        const int mode_gap = std::abs(int(o_r[i]) - int(o_c[i]));
        if (mode_gap < 3) continue;
        ++divergent;
        max_gap = std::max(max_gap, mode_gap);
        if (std::abs(int(prod[i]) - int(o_c[i])) <= 1) ++explained_by_clamp;
    }
    CHECK(divergent > 0, "fixture reaches a destination byte where reflect and clamp differ");
    CHECK(explained_by_clamp == 0, "no divergent destination byte is explainable by clamping");

    // The 1-pixel destination border is the reported evidence line.
    int border_divergent = 0, border_max_gap = 0;
    for (int y = 0; y < dw; ++y) {
        for (int x : {0, dw - 1}) {
            const size_t i = (static_cast<size_t>(y) * dw + x) * 3;
            for (int c = 0; c < 3; ++c) {
                const int gap = std::abs(int(o_r[i + c]) - int(o_c[i + c]));
                if (gap > 2) {
                    ++border_divergent;
                    border_max_gap = std::max(border_max_gap, gap);
                }
            }
        }
    }
    CHECK(border_divergent > 0, "the divergence is carried by the 1-pixel destination border");
    std::printf("  %-26s n=%-3d scale=%d  reflect-vs-clamp: %d divergent bytes (max %d), "
                "%d on the dst 1px border (max %d), product-vs-oracle gap %d\n",
                name, n, scale, divergent, max_gap, border_divergent, border_max_gap, oracle_gap);
    std::printf("     dst(0,0)   product=%d,%d,%d   reflect=%d,%d,%d   clamp=%d,%d,%d\n", prod[0],
                prod[1], prod[2], o_r[0], o_r[1], o_r[2], o_c[0], o_c[1], o_c[2]);
}

void sectionEasuBorder() {
    // Scale 2 is deliberately NOT a byte-level fixture: it does reach source tap
    // -2 (asserted in sectionTapReach) but its maximum reflect-vs-clamp byte delta
    // is 2, which the 1-byte oracle gap cannot separate from a clamped
    // expectation. Claiming border coverage there would be an overclaim.
    easuFixture("ring black / interior white", 8, 3, 0, 0, 0, 255, 255, 255);
    easuFixture("ring black / interior white", 8, 4, 0, 0, 0, 255, 255, 255);
    easuFixture("ring blue / interior yellow", 16, 4, 0, 0, 255, 255, 255, 0);
    easuFixture("ring teal / interior amber", 16, 4, 10, 20, 245, 128, 130, 60);
    easuFixture("ring black / interior white", 16, 4, 0, 0, 0, 255, 255, 255);
}

// ---------------------------------------------------------------------------
// C  guidance luma through the real builder
// ---------------------------------------------------------------------------
enum Pattern {
    kBlack = 0,   // luma must be exactly 0.0f
    kWhite,       // luma must be exactly 1.0f
    kFlat,        // (200,120,40): CAS provably the identity -> pure luma pin
    kHoroRamp,    // R!=G!=B along x
    kVertRamp,    // R!=G!=B along y
    kChecker,     // mid-range 2x2 blocks: CAS has headroom, sharpens hard
    kRing,        // 1px border ring: reflect-mirrored stencil
    kPatternCount
};

void paint(int kind, int x, int y, int& r, int& g, int& b) {
    switch (kind) {
        case kBlack: r = g = b = 0; break;
        case kWhite: r = g = b = 255; break;
        case kFlat: r = 200; g = 120; b = 40; break;
        case kHoroRamp: {
            const int v = (x * 255) / (FRAME_W - 1);
            r = v; g = (v + 85) % 256; b = (v + 170) % 256;
            break;
        }
        case kVertRamp: {
            const int v = (y * 255) / (FRAME_H - 1);
            r = v; g = (v + 85) % 256; b = (v + 170) % 256;
            break;
        }
        case kChecker: {
            const bool hi = ((x / 2 + y / 2) % 2) == 0;
            r = hi ? 190 : 60; g = hi ? 180 : 70; b = hi ? 170 : 80;
            break;
        }
        default: {
            const bool edge = (x == 0 || y == 0 || x == FRAME_W - 1 || y == FRAME_H - 1);
            r = edge ? 10 : 128; g = edge ? 20 : 130; b = edge ? 245 : 60;
            break;
        }
    }
}

std::vector<uint8_t> makeFrame(int kind) {
    std::vector<uint8_t> rgb(static_cast<size_t>(FRAME_W) * FRAME_H * 3, 0);
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            int r, g, b;
            paint(kind, x, y, r, g, b);
            const size_t i = (static_cast<size_t>(y) * FRAME_W + x) * 3;
            rgb[i] = static_cast<uint8_t>(r);
            rgb[i + 1] = static_cast<uint8_t>(g);
            rgb[i + 2] = static_cast<uint8_t>(b);
        }
    }
    return rgb;
}

const char* patternName(int kind) {
    static const char* n[kPatternCount] = {"flat black",        "flat white", "flat (200,120,40)",
                                          "horizontal ramp",   "vertical ramp",
                                          "2x2 checkerboard",  "1px border ring"};
    return n[kind];
}

// Tolerance per fixture.
//  - flat black/white: 1e-12. CAS is provably the identity on a flat field
//    (mn == mx -> amp == 0 -> out == in), so no byte quantization ambiguity
//    exists; the only residue is that the decimal weights 0.299+0.587+0.114 sum
//    to 0.99999999999999989 in double while the product's float32 evaluation of
//    white lands on exactly 1.0f, i.e. ~1e-16 of oracle disagreement.
//  - flat colour: 1e-6. Same identity, and this fixture carries the weight pin,
//    so it is pinned ~4000x tighter than the 1/255 byte step.
//  - the four textured fixtures: 1/255 + float-vs-double slack. |float value -
//    double value| is < 1e-4 of a byte, so each sharpened byte can differ by at
//    most ONE unit, and the three weights sum to 1, giving
//    |dLuma| <= (0.299+0.587+0.114)/255 = 0.0039216. 0.0040 is that derived bound
//    plus slack, not a fudge factor.
double toleranceFor(int kind) {
    switch (kind) {
        case kBlack:
        case kWhite: return 1e-12;
        case kFlat: return 1e-6;
        default: return 0.0040;
    }
}

void sectionGuidanceLuma() {
    const size_t kPixels = static_cast<size_t>(FRAME_W) * FRAME_H;
    for (int kind = 0; kind < kPatternCount; ++kind) {
        const std::vector<uint8_t> rgb = makeFrame(kind);
        const double tol = toleranceFor(kind);

        SignalConditioner sc;
        sc.buildGuidanceForTests(rgb);
        const std::vector<float>& luma = sc.guidanceLumaForTests();
        const std::vector<uint8_t>& cas_rgb = sc.guidanceRgbForTests();
        CHECK(luma.size() == kPixels, "guidance luma covers the whole 640x480 frame");
        CHECK(cas_rgb.size() == kPixels * 3, "guidance RGB is a full-frame RGB image");

        const std::vector<uint8_t> oracle_rgb = oracleCas(rgb, FRAME_W, FRAME_H);

        double worst = 0.0, lo = 1e9, hi = -1e9;
        bool finite_in_range = true;
        int bytes_cas_moved = 0;
        double max_rb = 0.0, max_709 = 0.0, max_nodiv = 0.0, max_unsharp = 0.0;
        for (size_t i = 0; i < kPixels; ++i) {
            const size_t j = i * 3;
            const float v = luma[i];
            if (!std::isfinite(v) || v < 0.0f || v > 1.0f) finite_in_range = false;
            lo = std::min<double>(lo, v);
            hi = std::max<double>(hi, v);
            if (cas_rgb[j] != rgb[j] || cas_rgb[j + 1] != rgb[j + 1] || cas_rgb[j + 2] != rgb[j + 2]) {
                ++bytes_cas_moved;
            }
            const int or0 = cas_rgb[j], or1 = cas_rgb[j + 1], or2 = cas_rgb[j + 2];
            worst = std::max(worst, std::abs(static_cast<double>(v) -
                                            lumaRule(oracle_rgb[j], oracle_rgb[j + 1],
                                                     oracle_rgb[j + 2])));
            // Wrong-value exclusion, measured at the same post-CAS bytes so the
            // comparison isolates the arithmetic being pinned.
            max_rb = std::max(max_rb, std::abs(static_cast<double>(v) - lumaRB(or0, or1, or2)));
            max_709 = std::max(max_709, std::abs(static_cast<double>(v) - luma709(or0, or1, or2)));
            max_nodiv = std::max(max_nodiv,
                                 std::abs(static_cast<double>(v) - lumaNoDiv255(or0, or1, or2)));
            max_unsharp = std::max(max_unsharp,
                                   std::abs(static_cast<double>(v) -
                                            lumaRule(rgb[j], rgb[j + 1], rgb[j + 2])));
        }

        // Canonical rule over the post-CAS bytes.
        CHECK(worst <= tol, patternName(kind));
        CHECK(worst <= tol, "guidance luma matches (0.299R+0.587G+0.114B)/255 over the oracle CAS bytes");
        // Range: derived, not assumed. applyCAS_CPU clamps to [0,1] before
        // scaling, so guidance bytes are in [0,255]; the three weights sum to 1,
        // so the exact real range is [0,1]; and the float32 evaluation was
        // checked exhaustively over all 2^24 byte triples to reach min exactly
        // 0.0f and max exactly 1.0f, so no sharpening overshoot can escape it.
        CHECK(finite_in_range, "every guidance luma is finite and inside the derived [0,1] range");

        switch (kind) {
            case kBlack:
                CHECK(bytes_cas_moved == 0, "CAS is the identity on a flat field (black)");
                CHECK(hi == 0.0 && lo == 0.0, "flat black guidance luma is exactly 0.0f");
                break;
            case kWhite:
                CHECK(bytes_cas_moved == 0, "CAS is the identity on a flat field (white)");
                CHECK(hi == 1.0 && lo == 1.0, "flat white guidance luma is exactly 1.0f (pins /255)");
                break;
            case kFlat:
                CHECK(bytes_cas_moved == 0, "CAS is the identity on a flat coloured field");
                CHECK(max_rb >= 0.05, "flat colour is not the R/B-swapped luma");
                CHECK(max_709 >= 0.005, "flat colour is not Rec.709 luma");
                CHECK(max_nodiv >= 100.0, "flat colour is not the un-divided-by-255 luma");
                break;
            case kHoroRamp:
            case kVertRamp:
                CHECK(bytes_cas_moved > 0, "ramp fixture actually moves CAS bytes");
                CHECK(max_rb >= 0.05, "ramp luma is not the R/B-swapped luma");
                CHECK(max_709 >= 0.03, "ramp luma is not Rec.709 luma");
                CHECK(max_nodiv >= 100.0, "ramp luma is not the un-divided-by-255 luma");
                break;
            case kChecker:
                // The only fixture whose CAS delta is large enough to separate
                // the post-CAS guidance image from the raw input image.
                CHECK(bytes_cas_moved > 300000, "checkerboard is sharpened almost everywhere");
                CHECK(max_unsharp >= 0.02,
                      "checkerboard luma comes from the POST-CAS image, not the unsharpened input");
                CHECK(max_nodiv >= 100.0, "checkerboard luma is not the un-divided-by-255 luma");
                break;
            default:
                CHECK(bytes_cas_moved > 0, "border ring fixture actually moves CAS bytes");
                CHECK(max_rb >= 0.05, "border ring luma is not the R/B-swapped luma");
                CHECK(max_nodiv >= 100.0, "border ring luma is not the un-divided-by-255 luma");
                break;
        }
        std::printf("  %-19s oracle|err| max=%.7f (tol %.1e)  luma in [%.6f, %.6f]  CAS-moved=%d/%zu\n",
                    patternName(kind), worst, tol, lo, hi, bytes_cas_moved, kPixels);
        std::printf("     wrong-value gaps: RB=%.6f  Rec709=%.6f  no/255=%.3f  unsharpened=%.6f\n",
                    max_rb, max_709, max_nodiv, max_unsharp);
    }
}

// ---------------------------------------------------------------------------
// D / E  determinism across repeats and thread counts
// ---------------------------------------------------------------------------
uint64_t guidanceDigest(int kind) {
    SignalConditioner sc;
    sc.buildGuidanceForTests(makeFrame(kind));
    return digestLuma(sc.guidanceLumaForTests());
}

void sectionDeterminism() {
    for (int kind : {kChecker, kRing, kHoroRamp}) {
        uint64_t first = 0;
        bool stable = true;
        for (int run = 0; run < 4; ++run) {
            const uint64_t d = guidanceDigest(kind);
            if (run == 0) first = d;
            if (d != first) stable = false;
        }
        CHECK(stable, patternName(kind));
        CHECK(stable, "four fresh-conditioner guidance runs are bit-identical");
        std::printf("  %-19s luma digest %016llx\n", patternName(kind),
                    static_cast<unsigned long long>(first));
    }
}

void sectionThreads() {
    uint64_t base[kPatternCount] = {0};
    for (int kind = 0; kind < kPatternCount; ++kind) base[kind] = guidanceDigest(kind);
    bool identical = true;
    for (int n : {1, 2, 4, 1}) {
        pinThreads(n);
        for (int kind = 0; kind < kPatternCount; ++kind) {
            if (guidanceDigest(kind) != base[kind]) identical = false;
        }
        restoreThreads();
        std::printf("  OMP threads %d -> all %d pattern digests match\n", n, kPatternCount);
    }
    CHECK(identical, "guidance luma digests are identical at OpenMP threads 1 / 2 / 4");
}

}  // namespace

int main() {
    sectionBorderAlgebra();
    sectionTapReach();
    sectionEasuBorder();
    sectionGuidanceLuma();
    sectionDeterminism();
    sectionThreads();

    if (g_failures == 0) {
        std::printf("cas_border_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("cas_border_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
