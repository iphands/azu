// sr_upscaled_contract (big-fix Todo 22): CPU-only contract for the upscaled-RGB
// path. Public CPU surfaces only: SignalConditioner / Preprocessor availability
// API, the real process() production path, and the public CPU passes
// sr::applyEASU_CPU / sr::applyCAS_CPU. Nothing here reimplements EASU or CAS:
// the freshness check drives the REAL public CPU passes over the frame the
// conditioner actually conditioned and compares bytes, and the sharpness section
// calls the REAL sr::applyCAS_CPU with the product's own sharpness parameter.
// No device, display, GPU, sensor, thread timing, sleep or filesystem.
//
// What is locked, and why each fixture is shaped like it is:
//
//   A  a freshly constructed conditioner reports the buffer UNAVAILABLE even
//      though getSrRgbUpscaled() already contains FRAME_W*FRAME_H*3 zero bytes.
//      That preallocation is the black-texture hazard itself: the getter alone
//      cannot be a validity signal, so this section pins both halves of that
//      statement (bytes present, contract says no).
//   B  a valid CPU frame publishes exactly FRAME_W*scale * FRAME_H*scale * 3
//      bytes for scale in {2,3,4}, carries the frame's own id, and equals the
//      public CPU EASU+RCAS output computed over the image process() left in
//      raw.rgb — i.e. this frame's bytes, not a stale or zero buffer.
//   C  a frame rejected by the size guard invalidates the CURRENT frame: the
//      previous frame's availability and id are gone with it, so a consumer can
//      never texture from bytes it believes are fresh.
//   D  a scale outside [2,4] fails closed. Scale 1 is the load-bearing case:
//      the old code copied an ORIGINAL-resolution image into this buffer, and a
//      buffer left over from a real 2x frame still has exactly the 2x byte count
//      after the scale changes — so this section proves the availability check
//      is geometry-aware and not one stale boolean.
//   E  reset() clears availability and the frame id. Here the surviving 2x byte
//      count proves the opposite half: the boolean is load-bearing too, since
//      geometry alone cannot tell a reset buffer from a published one.
//   F  non-finite sharpness in the real CPU CAS pass. std::clamp returns NaN
//      unchanged, so NaN reached static_cast<uint8_t>(NaN * 255.0f) — undefined
//      behavior whose x86 result was a BLACK frame. Canonical: every non-finite
//      sharpness (NaN, +Inf, -Inf) maps to the softest valid sharpen, 0.0f, and
//      is byte-identical to it. The 0.5f and 1.0f runs are the non-vacuity
//      witness: the fixture demonstrably responds to sharpness, so "identical to
//      0.0f" is a decision, not an accident of a pass that ignores its input.
//   G  the Preprocessor-level contract, through makePreprocessor(): CPU honors
//      production, the CUDA/HIP-labelled preprocessor fails closed. Only its
//      availability queries run — no process() call, so no device path exists.
//   H  published bytes are deterministic across repeats and OpenMP threads
//      1 / 2 / 4 (the buffer is published by move after the whole frame).
#include "sensor/FrameData.h"
#include "sensor/KinectSensor.h"
#include "sensor/Preprocessor.h"
#include "sensor/SignalConditioner.h"
#include "sensor/SuperResolution.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(CUDA_ENABLED) || defined(HIP_ENABLED)
#error "sr_upscaled_contract is the CPU lane; a backend build must not claim CPU upscaled coverage"
#endif

namespace {

using kfusion::sensor::FRAME_H;
using kfusion::sensor::FRAME_W;
using kfusion::sensor::PreprocessBackend;
using kfusion::sensor::Preprocessor;
using kfusion::sensor::RawFrame;
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

constexpr int kScaleMin = 2;
constexpr int kScaleMax = 4;
constexpr float kDepthMinM = 0.3f;
constexpr float kDepthMaxM = 2.5f;
constexpr uint64_t kNoFrame = 0;

// Exact expected size, widened before every multiply (the scale-4 case is
// 2560 * 1920 * 3 = 14,745,600 bytes). Only ever called with a valid scale.
size_t expectedBytes(int scale) {
    return static_cast<size_t>(FRAME_W) * static_cast<size_t>(scale) *
           static_cast<size_t>(FRAME_H) * static_cast<size_t>(scale) * 3u;
}
size_t originalBytes() { return static_cast<size_t>(FRAME_W) * FRAME_H * 3u; }

uint64_t fnv1a(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}
uint64_t digest(const std::vector<uint8_t>& v) { return fnv1a(v.data(), v.size()); }
size_t nonZeroBytes(const std::vector<uint8_t>& v) {
    size_t n = 0;
    for (uint8_t b : v) n += (b != 0) ? 1u : 0u;
    return n;
}

// Structured, non-flat, non-black frame: the border band exercises the reflect
// border of both CPU passes, the ramps give EASU real gradient to resample and
// CAS real contrast headroom, so a black or zero buffer can never match it.
RawFrame makeFrame(uint64_t id) {
    RawFrame f;
    f.frame_id = id;
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            const size_t i = (static_cast<size_t>(y) * FRAME_W + x) * 3;
            const bool border = (x < 2 || y < 2 || x >= FRAME_W - 2 || y >= FRAME_H - 2);
            f.rgb[i]     = static_cast<uint8_t>(border ? 0 : ((x * 7 + y * 3) & 0xFF));
            f.rgb[i + 1] = static_cast<uint8_t>(border ? 0 : ((x * 5 + y * 11 + 64) & 0xFF));
            f.rgb[i + 2] = static_cast<uint8_t>(border ? 255 : (255 - ((x * 3 + y) & 0xFF)));
            // In-band raw codes only, so the depth stages do ordinary work and
            // nothing about the SR contract depends on an invalid-depth path.
            f.depth[static_cast<size_t>(y) * FRAME_W + x] =
                static_cast<uint16_t>(300 + ((x + 2 * y) % 120));
        }
    }
    f.depth_valid = true;
    f.rgb_valid   = true;
    return f;
}

bool allZero(const std::vector<uint8_t>& v) { return nonZeroBytes(v) == 0; }

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
// A  fresh object: bytes present, contract says no
// ---------------------------------------------------------------------------
void sectionFreshObject() {
    SignalConditioner sc;
    CHECK(!sc.srUpscaledAvailable(), "a fresh conditioner reports the upscaled buffer unavailable");
    CHECK(sc.srUpscaledFrameId() == kNoFrame, "a fresh conditioner has no upscaled frame id");
    CHECK(!sc.srUpscaledAvailableForFrame(kNoFrame),
          "frame id 0 is not an implicit publication");
    CHECK(!sc.srUpscaledAvailableForFrame(1), "no frame id matches before any process()");

    const std::vector<uint8_t>& buf = sc.getSrRgbUpscaled();
    CHECK(buf.size() == originalBytes(),
          "the raw getter still hands out the original-resolution preallocation");
    CHECK(allZero(buf), "which is all zero — the black-texture hazard the contract gates");
    CHECK(buf.size() != expectedBytes(sc.getSrScale()),
          "and it is not even a valid upscale at the default scale");
    std::printf("  fresh object: getter size=%zu (all zero), available=%d, scale=%d\n",
                buf.size(), static_cast<int>(sc.srUpscaledAvailable()), sc.getSrScale());
}

// ---------------------------------------------------------------------------
// B  valid CPU production through the real process() path
// ---------------------------------------------------------------------------
void sectionProduction(int scale) {
    const uint64_t kId = 0x1234 + static_cast<uint64_t>(scale);
    RawFrame raw = makeFrame(kId);
    SignalConditioner sc;
    sc.setSrScale(scale);
    CHECK(sc.getSrScale() == scale, "scale accepted by the public setter");

    sc.process(raw, nullptr, kDepthMinM, kDepthMaxM);

    CHECK(sc.srUpscaledAvailable(), "valid CPU frame publishes the upscaled buffer");
    CHECK(sc.srUpscaledFrameId() == kId, "the published id is raw.frame_id");
    CHECK(sc.srUpscaledAvailableForFrame(kId), "available for its own frame");
    CHECK(!sc.srUpscaledAvailableForFrame(kId + 1), "not available for any other frame");

    const std::vector<uint8_t>& buf = sc.getSrRgbUpscaled();
    const size_t want = expectedBytes(scale);
    CHECK(buf.size() == want, "exact upscaled byte count FRAME_W*scale * FRAME_H*scale * 3");
    CHECK(buf.size() != originalBytes(), "the published buffer is not the original resolution");
    CHECK(nonZeroBytes(buf) > 0, "the published buffer is not the preallocated zero buffer");

    // Freshness and correct source, using the product's own public CPU passes on
    // the image process() conditioned (preprocessRgb leaves its result in
    // raw.rgb, which is exactly what the SR stage consumed).
    std::vector<uint8_t> ref;
    kfusion::sensor::sr::applyEASU_CPU(raw.rgb, ref, FRAME_W, FRAME_H, scale);
    CHECK(ref.size() == want, "public CPU EASU produces the same exact size");
    kfusion::sensor::sr::applyCAS_CPU(ref, FRAME_W * scale, FRAME_H * scale, 0.5f);
    CHECK(ref.size() == buf.size() && std::equal(ref.begin(), ref.end(), buf.begin()),
          "published buffer is byte-identical to product EASU+RCAS output for THIS frame");

    std::printf("  scale %dx: %llu bytes (%dx%d), non-zero %zu, matches product passes=%d, id=%llu\n",
                scale, static_cast<unsigned long long>(buf.size()), FRAME_W * scale,
                FRAME_H * scale, nonZeroBytes(buf), static_cast<int>(ref == buf),
                static_cast<unsigned long long>(sc.srUpscaledFrameId()));
}

// ---------------------------------------------------------------------------
// C  an invalid frame invalidates the current frame
// ---------------------------------------------------------------------------
void sectionInvalidGeometry() {
    SignalConditioner sc;
    sc.setSrScale(2);
    RawFrame good = makeFrame(11);
    sc.process(good, nullptr, kDepthMinM, kDepthMaxM);
    CHECK(sc.srUpscaledAvailable(), "fixture: the good frame published");
    CHECK(sc.srUpscaledAvailableForFrame(11), "fixture: good frame id");

    struct BadCase {
        const char* name;
        uint64_t id;
        bool shrink_rgb;
        bool shrink_depth;
        bool clear_rgb;
    };
    const BadCase cases[] = {
        {"rgb 3 bytes short", 12, true, false, false},
        {"depth half length", 13, false, true, false},
        {"rgb empty", 14, false, false, true},
    };
    for (const BadCase& c : cases) {
        RawFrame bad = makeFrame(c.id);
        if (c.shrink_rgb) bad.rgb.resize(bad.rgb.size() - 3);
        if (c.shrink_depth) bad.depth.resize(bad.depth.size() / 2);
        if (c.clear_rgb) bad.rgb.clear();
        sc.process(bad, nullptr, kDepthMinM, kDepthMaxM);
        CHECK(!sc.srUpscaledAvailable(), c.name);
        CHECK(!sc.srUpscaledAvailableForFrame(c.id), "a rejected frame publishes nothing");
        CHECK(!sc.srUpscaledAvailableForFrame(11), "and it retires the PREVIOUS frame too");
        CHECK(sc.srUpscaledFrameId() == kNoFrame, "no stale id survives a rejected frame");
        std::printf("  %-18s -> available=%d, frame id=%llu\n", c.name,
                    static_cast<int>(sc.srUpscaledAvailable()),
                    static_cast<unsigned long long>(sc.srUpscaledFrameId()));
    }
}

// ---------------------------------------------------------------------------
// D  scale policy: only 2..4 can be an upscale
// ---------------------------------------------------------------------------
void sectionScalePolicy() {
    SignalConditioner sc;
    sc.setSrScale(kScaleMin);
    RawFrame good = makeFrame(21);
    sc.process(good, nullptr, kDepthMinM, kDepthMaxM);
    CHECK(sc.srUpscaledAvailable(), "fixture: 2x published");
    const size_t stale2x = sc.getSrRgbUpscaled().size();
    CHECK(stale2x == expectedBytes(2), "fixture: 2x byte count");

    // The geometry layer must catch this on its own: the bytes still fit 2x
    // exactly, so a flag-only reading of the contract would keep them valid.
    sc.setSrScale(1);
    CHECK(!sc.srUpscaledAvailable(),
          "a 2x-sized buffer stops being valid the moment the scale is 1");

    RawFrame g1 = makeFrame(22);
    sc.process(g1, nullptr, kDepthMinM, kDepthMaxM);
    CHECK(!sc.srUpscaledAvailable(), "scale 1 never publishes an upscaled buffer");
    CHECK(sc.srUpscaledFrameId() == kNoFrame, "scale 1 publishes no frame id");
    CHECK(sc.getSrRgbUpscaled().size() == stale2x,
          "scale 1 no longer overwrites the buffer with an original-resolution image");

    for (int s : {0, 5}) {
        sc.setSrScale(s);
        RawFrame g = makeFrame(30 + static_cast<uint64_t>(s));
        sc.process(g, nullptr, kDepthMinM, kDepthMaxM);
        CHECK(!sc.srUpscaledAvailable(), "scale outside [2,4] fails closed");
        CHECK(sc.srUpscaledFrameId() == kNoFrame, "a rejected scale publishes no frame id");
    }
    sc.setSrScale(kScaleMax);
    RawFrame g4 = makeFrame(35);
    sc.process(g4, nullptr, kDepthMinM, kDepthMaxM);
    CHECK(sc.srUpscaledAvailable(), "scale 4 is inside the contract");
    std::printf("  scale policy: 2x bytes %zu survived a scale-1 and 3 rejected scales\n", stale2x);
}

// ---------------------------------------------------------------------------
// E  reset() clears the contract, and geometry alone cannot
// ---------------------------------------------------------------------------
void sectionResetClears() {
    SignalConditioner sc;
    sc.setSrScale(2);
    RawFrame good = makeFrame(41);
    sc.process(good, nullptr, kDepthMinM, kDepthMaxM);
    CHECK(sc.srUpscaledAvailable(), "fixture: published before reset");

    const size_t after_reset_size = sc.getSrRgbUpscaled().size();
    sc.reset();
    CHECK(!sc.srUpscaledAvailable(), "reset() clears availability");
    CHECK(sc.srUpscaledFrameId() == kNoFrame, "reset() clears the frame id");
    CHECK(!sc.srUpscaledAvailableForFrame(41), "reset() retires frame 41");
    CHECK(sc.getSrRgbUpscaled().size() == after_reset_size &&
              after_reset_size == expectedBytes(2),
          "the buffer still fits 2x after reset, so the boolean is load-bearing too");
    CHECK(allZero(sc.getSrRgbUpscaled()), "and it is zeroed, never consumable");
    std::printf("  reset: size stays %zu (2x shape), available=%d, id=%llu\n", after_reset_size,
                static_cast<int>(sc.srUpscaledAvailable()),
                static_cast<unsigned long long>(sc.srUpscaledFrameId()));
}

// ---------------------------------------------------------------------------
// F  non-finite sharpness in the real CPU CAS pass
// ---------------------------------------------------------------------------
constexpr int kCasW = 96;
constexpr int kCasH = 64;

std::vector<uint8_t> casFixture() {
    std::vector<uint8_t> v(static_cast<size_t>(kCasW) * kCasH * 3, 0);
    for (int y = 0; y < kCasH; ++y) {
        for (int x = 0; x < kCasW; ++x) {
            const size_t i = (static_cast<size_t>(y) * kCasW + x) * 3;
            const bool edge = (x == 0 || y == 0 || x == kCasW - 1 || y == kCasH - 1);
            v[i]     = static_cast<uint8_t>(edge ? 8 : ((x * 9 + y * 5) & 0xFF));
            v[i + 1] = static_cast<uint8_t>(edge ? 200 : ((x * 3 + y * 13 + 32) & 0xFF));
            v[i + 2] = static_cast<uint8_t>(edge ? 128 : (255 - ((x + y * 7) & 0xFF)));
        }
    }
    return v;
}

std::vector<uint8_t> runCas(float sharpness) {
    std::vector<uint8_t> v = casFixture();
    kfusion::sensor::sr::applyCAS_CPU(v, kCasW, kCasH, sharpness);
    return v;
}

void sectionSharpnessFinite() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    const std::vector<uint8_t> soft = runCas(0.0f);
    CHECK(nonZeroBytes(soft) > 0, "the 0.0f reference is a real image, not a black frame");
    CHECK(nonZeroBytes(soft) < soft.size(), "the 0.0f reference is not saturated white");

    struct Case {
        const char* name;
        float value;
    };
    const Case cases[] = {
        {"NaN sharpness", nan},
        {"+Inf sharpness", inf},
        {"-Inf sharpness", -inf},
        {"NaN via 0/0-form sign", -nan},
    };
    for (const Case& c : cases) {
        const std::vector<uint8_t> out = runCas(c.value);
        CHECK(out.size() == soft.size(), c.name);
        CHECK(out == soft, "non-finite sharpness is byte-identical to the finite 0.0f output");
        CHECK(digest(out) == digest(soft), "same digest as the 0.0f output");
        CHECK(nonZeroBytes(out) > 0, "non-finite sharpness does not collapse to a black frame");
        std::printf("  %-22s digest=%016llx (0.0f digest=%016llx) identical=%d\n", c.name,
                    static_cast<unsigned long long>(digest(out)),
                    static_cast<unsigned long long>(digest(soft)), static_cast<int>(out == soft));
    }

    // Non-vacuity: the pass responds to sharpness, so mapping every non-finite
    // value onto 0.0f is a decision the fixture could have failed to observe.
    const std::vector<uint8_t> mid = runCas(0.5f);
    const std::vector<uint8_t> hard = runCas(1.0f);
    CHECK(mid != soft, "0.5f sharpness produces different bytes than 0.0f (fixture responds)");
    CHECK(hard != soft && hard != mid, "1.0f is a third distinct result");
    CHECK(digest(runCas(0.0f)) == digest(soft), "0.0f itself repeats bit-identically");

    // Finite out-of-range values keep the documented clamp semantics.
    CHECK(runCas(-1.0f) == soft, "finite -1.0 clamps to 0.0");
    const std::vector<uint8_t> two = runCas(2.0f);
    CHECK(two == hard, "finite 2.0 clamps to 1.0");
    std::printf("  finite clamps: -1.0==0.0 %d, 2.0==1.0 %d\n", 1,
                static_cast<int>(two == hard));
}

// ---------------------------------------------------------------------------
// G  Preprocessor-level contract through the public factory
// ---------------------------------------------------------------------------
void sectionPreprocessorContract() {
    PreprocessBackend actual = PreprocessBackend::Auto;
    std::unique_ptr<Preprocessor> cpu =
        kfusion::sensor::makePreprocessor(PreprocessBackend::CPU, false, nullptr, &actual);
    CHECK(cpu != nullptr, "CPU preprocessor created");
    CHECK(actual == PreprocessBackend::CPU, "requested CPU backend honored");
    CHECK(cpu->srUpscaledAvailable() == false, "CPU preprocessor starts unavailable");

    RawFrame good = makeFrame(51);
    cpu->setSrScale(3);
    cpu->process(good, kDepthMinM, kDepthMaxM);
    CHECK(cpu->srUpscaledAvailable(), "CPU preprocessor publishes after a valid frame");
    CHECK(cpu->srUpscaledFrameId() == 51, "CPU preprocessor reports the frame id");
    CHECK(cpu->srUpscaledAvailableForFrame(51), "frame-pinned query agrees");
    CHECK(cpu->getSrRgbUpscaled().size() == expectedBytes(3), "and the exact 3x size");
    cpu->reset();
    CHECK(!cpu->srUpscaledAvailable(), "reset() propagates through the interface");

    // The CUDA/HIP-labelled preprocessor is fail-closed by contract. It is only
    // constructed (a stream pointer is stored) and queried — process() is never
    // called, so no device path is reached and no backend code is compiled.
    std::unique_ptr<Preprocessor> gpu =
        kfusion::sensor::makePreprocessor(PreprocessBackend::HIP, true, nullptr, &actual);
    CHECK(gpu != nullptr, "HIP-labelled preprocessor constructed without a device");
    CHECK(actual == PreprocessBackend::HIP, "the factory resolved the HIP request");
    // backend() is NOT asserted to be HIP: the GPU-labelled class reports CUDA for
    // a HIP selection, which is the deferred naming defect sensor:S-17. What the
    // upscaled contract needs is that this is not the CPU lane.
    CHECK(gpu->backend() != PreprocessBackend::CPU, "the GPU-labelled preprocessor is not CPU");
    gpu->setSrScale(2);
    CHECK(!gpu->srUpscaledAvailable(), "GPU/HIP path reports the upscaled buffer unavailable");
    CHECK(gpu->srUpscaledFrameId() == kNoFrame, "GPU/HIP path has no upscaled frame id");
    CHECK(!gpu->srUpscaledAvailableForFrame(0), "and no frame id satisfies it");

    std::unique_ptr<Preprocessor> degraded =
        kfusion::sensor::makePreprocessor(PreprocessBackend::CUDA, false, nullptr, &actual);
    CHECK(actual == PreprocessBackend::CPU, "a GPU request without a GPU degrades to CPU");
    CHECK(!degraded->srUpscaledAvailable(), "and still starts unavailable");
    std::printf("  preprocessor: CPU publishes 3x=%zu, HIP-labelled available=%d\n",
                expectedBytes(3), static_cast<int>(gpu->srUpscaledAvailable()));
}

// ---------------------------------------------------------------------------
// H  determinism across repeats and OpenMP thread counts
// ---------------------------------------------------------------------------
struct Run {
    uint64_t digest   = 0;
    size_t   size     = 0;
    bool     available = false;
    uint64_t frame_id = kNoFrame;
};

Run runUpscaled(int scale, uint64_t id) {
    RawFrame raw = makeFrame(id);
    SignalConditioner sc;
    sc.setSrScale(scale);
    sc.process(raw, nullptr, kDepthMinM, kDepthMaxM);
    const std::vector<uint8_t>& buf = sc.getSrRgbUpscaled();
    return Run{digest(buf), buf.size(), sc.srUpscaledAvailable(), sc.srUpscaledFrameId()};
}

void sectionDeterminism() {
    const Run base = runUpscaled(2, 61);
    CHECK(base.available, "fixture: first run published");
    bool stable = true;
    for (int run = 0; run < 2; ++run) {
        const Run r = runUpscaled(2, 61);
        if (r.digest != base.digest || r.size != base.size || !r.available ||
            r.frame_id != base.frame_id) {
            stable = false;
        }
    }
    CHECK(stable, "repeated fresh-conditioner runs publish byte-identical buffers");
    std::printf("  repeat digest %016llx size %zu\n", static_cast<unsigned long long>(base.digest),
                base.size);

    for (int n : {1, 2, 4, 1}) {
        pinThreads(n);
        const Run r = runUpscaled(2, 61);
        restoreThreads();
        CHECK(r.digest == base.digest, "OpenMP thread count reproduces the published bytes");
        CHECK(r.size == base.size && r.available && r.frame_id == base.frame_id,
              "and the whole contract, at every thread count");
        std::printf("  OMP threads %d -> digest %016llx\n", n,
                    static_cast<unsigned long long>(r.digest));
    }
}

}  // namespace

int main() {
    sectionFreshObject();
    sectionProduction(2);
    sectionProduction(3);
    sectionProduction(4);
    sectionInvalidGeometry();
    sectionScalePolicy();
    sectionResetClears();
    sectionSharpnessFinite();
    sectionPreprocessorContract();
    sectionDeterminism();

    if (g_failures == 0) {
        std::printf("sr_upscaled_contract: PASS (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("sr_upscaled_contract: FAIL (%d failed checks)\n", g_failures);
    return 1;
}
