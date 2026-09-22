// fusion_ui_validation_contract (big-fix todo 29): CPU-only, Qt-free contract
// for the fusion hyperparameter validation, preset staging and metrics style
// band model (include/gui/FusionUiModel.h + src/gui/FusionUiModel.cpp).
//
// The test drives the REAL translation unit (linked through azu_test_core, gate
// 11 in tests/CMakeLists.txt makes a local replica fail) through its public
// namespace. Every expectation is derived from the documented rule, never
// snapshotted from product output. Cases locked here:
//   * min_depth < max_depth strictly; equal and inverted ranges are rejected
//   * max_depth over the device ceiling (4.0 m) is rejected, == 4.0 m passes
//   * truncation/voxel ratio below 2x and above 8x rejected, inclusive bounds pass
//   * NaN / +/-Inf / non-positive geometry rejected (no divide-by-zero on the ratio)
//   * FusionHyperparams::defaults() passes
//   * each preset stages ONE complete atomic value (non-preset fields carried
//     from the base, preset fields overwritten) and validates against defaults
//   * the overlap band maps icp_valid_model == -1 to the textual "--" UNKNOWN
//     band (never a premature warming colour)
//   * the band cache transitions only on a band change and never re-polishes when
//     the band is unchanged
// No device, display, GPU, OpenGL context, sensor, Qt Widgets, thread timing,
// wall clock or filesystem access.

#include "gui/FusionUiModel.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#ifndef AZU_PIPELINE_TEST_SEAM
#error "fusion_ui_validation_contract must be compiled with AZU_PIPELINE_TEST_SEAM (test-target-only definition)"
#endif

namespace {

// Every call goes through this alias so the identifier is namespace-qualified
// (`fgui::name(`): the anti-synthetic gate matches a receiver-qualified CALL
// SHAPE, and this is the exact name the widgets resolve through kfusion::gui.
namespace fgui = kfusion::gui;
using FusionHyperparams = kfusion::app::FusionHyperparams;

constexpr float kTol = 1.0e-6f;

int g_failures = 0;
int g_checks   = 0;

bool nearF(float a, float b) {
    return a == b || std::fabs(a - b) <= kTol;
}

#define CHECK(cond, what)                                                        \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            std::printf("FAIL: %s  [%s:%d]\n", std::string(what).c_str(),        \
                        __FILE__, __LINE__);                                     \
            ++g_failures;                                                        \
        }                                                                        \
    } while (false)

#define CHECK_NEAR(a, b, what)                                                   \
    do {                                                                         \
        ++g_checks;                                                              \
        const float _va = static_cast<float>(a);                                 \
        const float _vb = static_cast<float>(b);                                 \
        if (!nearF(_va, _vb)) {                                                  \
            std::printf("FAIL: %s (got %g, want %g)  [%s:%d]\n",                 \
                        std::string(what).c_str(), _va, _vb, __FILE__, __LINE__);\
            ++g_failures;                                                        \
        }                                                                        \
    } while (false)

bool hasMessage(const fgui::FusionValidationResult& r, const std::string& needle) {
    for (const std::string& m : r.messages) {
        if (m.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

FusionHyperparams defaults() { return FusionHyperparams::defaults(); }

// ---- Section 1: defaults are the one valid seed ----
void sectionDefaultsPass() {
    const fgui::FusionValidationResult r = fgui::validateFusionHyperparams(defaults());
    CHECK(r.ok, "FusionHyperparams::defaults() must validate");
    CHECK(r.messages.empty(), "defaults must produce no validation messages");

    // The seed the panel uses IS defaults(), not a parallel table.
    const FusionHyperparams seeded = fgui::uiDefaultHyperparams();
    CHECK_NEAR(seeded.min_depth, defaults().min_depth, "seed min_depth == defaults");
    CHECK_NEAR(seeded.max_depth, defaults().max_depth, "seed max_depth == defaults");
    CHECK_NEAR(seeded.tsdf.voxel_size, defaults().tsdf.voxel_size, "seed voxel == defaults");
    CHECK(seeded.tsdf.resolution == defaults().tsdf.resolution, "seed resolution == defaults");
    // The device ceiling must be the documented 4.0 m, not a looser magic number.
    CHECK_NEAR(fgui::kDeviceMaxDepthMeters, 4.0f, "device depth ceiling is 4.0 m");
}

// ---- Section 2: depth band ordering ----
void sectionDepthOrdering() {
    {
        FusionHyperparams h = defaults();
        h.min_depth = 2.0f;
        h.max_depth = 2.0f; // equal, not strictly less
        const auto r = fgui::validateFusionHyperparams(h);
        CHECK(!r.ok, "equal min==max is rejected");
        CHECK(hasMessage(r, "less than"), "equal range reports the ordering rule");
    }
    {
        FusionHyperparams h = defaults();
        h.min_depth = 3.0f;
        h.max_depth = 1.0f; // inverted
        CHECK(!fgui::validateFusionHyperparams(h).ok, "inverted max<min is rejected");
    }
    {
        FusionHyperparams h = defaults();
        h.min_depth = 0.0f; // not strictly positive
        const auto r = fgui::validateFusionHyperparams(h);
        CHECK(!r.ok, "zero min_depth is rejected");
        CHECK(hasMessage(r, "greater than zero"), "zero min_depth reports the positivity rule");
    }
    {
        FusionHyperparams h = defaults();
        h.min_depth = 0.30f;
        h.max_depth = 2.50f;
        CHECK(fgui::validateFusionHyperparams(h).ok, "valid strictly-increasing band passes");
    }
}

// ---- Section 3: device depth ceiling ----
void sectionDeviceDepthCeiling() {
    {
        FusionHyperparams h = defaults();
        h.max_depth = 4.0f; // inclusive boundary: usable
        CHECK(fgui::validateFusionHyperparams(h).ok, "max_depth == 4.0 m passes");
    }
    {
        FusionHyperparams h = defaults();
        h.max_depth = 4.01f; // just over the ceiling
        const auto r = fgui::validateFusionHyperparams(h);
        CHECK(!r.ok, "max_depth over the device ceiling is rejected");
        CHECK(hasMessage(r, "4.0"), "the range message quotes the 4.0 m ceiling");
    }
    {
        FusionHyperparams h = defaults();
        h.max_depth = 12.0f; // the old unbounded Room/spin ceiling
        CHECK(!fgui::validateFusionHyperparams(h).ok, "12 m depth max is rejected");
    }
}

// ---- Section 4: truncation / voxel ratio ----
void sectionTruncationVoxelRatio() {
    {
        FusionHyperparams h = defaults(); // voxel 0.010, trunc 0.030 -> 3x
        CHECK_NEAR(h.tsdf.truncation / h.tsdf.voxel_size, 3.0f, "default ratio is 3x");
        CHECK(fgui::validateFusionHyperparams(h).ok, "default ratio passes");
    }
    {
        FusionHyperparams h = defaults();
        h.tsdf.truncation = 0.005f; // ratio 0.5 < 2
        const auto r = fgui::validateFusionHyperparams(h);
        CHECK(!r.ok, "truncation below 2x voxel is rejected");
        CHECK(hasMessage(r, "too thin"), "low ratio reports the thin-band rule");
    }
    {
        FusionHyperparams h = defaults();
        h.tsdf.truncation = 0.200f; // ratio 20 > 8
        const auto r = fgui::validateFusionHyperparams(h);
        CHECK(!r.ok, "truncation above 8x voxel is rejected");
        CHECK(hasMessage(r, "too thick"), "high ratio reports the thick-band rule");
    }
    {
        FusionHyperparams h = defaults();
        h.tsdf.voxel_size = 0.010f;
        h.tsdf.truncation = 0.020f; // ratio exactly 2 (inclusive)
        CHECK(fgui::validateFusionHyperparams(h).ok, "ratio == 2 passes");
    }
    {
        FusionHyperparams h = defaults();
        h.tsdf.voxel_size = 0.010f;
        h.tsdf.truncation = 0.080f; // ratio exactly 8 (inclusive)
        CHECK(fgui::validateFusionHyperparams(h).ok, "ratio == 8 passes");
    }
    {
        FusionHyperparams h = defaults();
        h.tsdf.voxel_size = 0.0f; // must not divide by zero while checking ratio
        const auto r = fgui::validateFusionHyperparams(h);
        CHECK(!r.ok, "zero voxel size is rejected");
        CHECK(hasMessage(r, "Voxel size must be greater than zero"),
              "zero voxel reports the geometry rule rather than a ratio crash");
    }
}

// ---- Section 5: finiteness and the remaining positive scalars ----
void sectionFinitenessAndPositivity() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    {
        FusionHyperparams h = defaults();
        h.min_depth = nan;
        CHECK(!fgui::validateFusionHyperparams(h).ok, "NaN min_depth is rejected");
    }
    {
        FusionHyperparams h = defaults();
        h.max_depth = inf;
        const auto r = fgui::validateFusionHyperparams(h);
        CHECK(!r.ok, "Inf max_depth is rejected");
        CHECK(hasMessage(r, "finite"), "a non-finite depth reports finiteness");
    }
    {
        FusionHyperparams h = defaults();
        h.tsdf.origin = Eigen::Vector3f(nan, 0.0f, 0.0f);
        CHECK(!fgui::validateFusionHyperparams(h).ok, "NaN origin is rejected");
    }
    {
        FusionHyperparams h = defaults();
        h.tsdf.max_weight = 0.0f;
        CHECK(!fgui::validateFusionHyperparams(h).ok, "zero max_weight is rejected");
    }
    {
        FusionHyperparams h = defaults();
        h.tsdf.resolution = 0;
        CHECK(!fgui::validateFusionHyperparams(h).ok, "zero resolution is rejected");
    }
    {
        FusionHyperparams h = defaults();
        h.icp.dist_threshold = 0.0f;
        CHECK(!fgui::validateFusionHyperparams(h).ok, "zero ICP distance is rejected");
    }
    {
        FusionHyperparams h = defaults();
        h.icp.angle_threshold = -1.0f;
        CHECK(!fgui::validateFusionHyperparams(h).ok, "negative ICP angle is rejected");
    }
}

// ---- Section 6: preset staging is atomic and device-bounded ----
FusionHyperparams sentinelBase() {
    FusionHyperparams b = defaults();
    b.min_depth = 0.42f;
    b.max_depth = 3.33f;
    b.sr_scale  = 3;
    b.tsdf.voxel_size = 0.007f;
    b.tsdf.resolution = 128;
    b.tsdf.max_weight = 200.0f;
    b.tsdf.origin     = Eigen::Vector3f(0.25f, -0.5f, 0.75f);
    b.icp.dist_threshold  = 0.33f;
    b.icp.angle_threshold = 15.0f;
    b.icp.max_iterations[0] = 1;
    b.icp.max_iterations[1] = 2;
    b.icp.max_iterations[2] = 3;
    return b;
}

void sectionPresetStaging() {
    // Custom is a pure passthrough of the base value.
    {
        const FusionHyperparams base = sentinelBase();
        const FusionHyperparams out = fgui::applyFusionPreset(base, fgui::FusionPreset::kCustom);
        CHECK_NEAR(out.min_depth, base.min_depth, "custom keeps min_depth");
        CHECK_NEAR(out.max_depth, base.max_depth, "custom keeps max_depth");
        CHECK_NEAR(out.tsdf.voxel_size, base.tsdf.voxel_size, "custom keeps voxel");
        CHECK(out.tsdf.origin.isApprox(base.tsdf.origin), "custom keeps origin");
    }

    const fgui::FusionPreset presets[] = {
        fgui::FusionPreset::kHelmet,
        fgui::FusionPreset::kChair,
        fgui::FusionPreset::kRoom,
        fgui::FusionPreset::kHuman,
    };

    // Every preset, staged onto the canonical defaults, is a valid complete state.
    for (fgui::FusionPreset p : presets) {
        const FusionHyperparams staged = fgui::applyFusionPreset(defaults(), p);
        const auto r = fgui::validateFusionHyperparams(staged);
        CHECK(r.ok, "every preset staged on defaults validates against the model");
    }

    // Atomicity: applying a preset yields base + patch, never a fresh default. A
    // field no preset touches (sr_scale, origin) must survive from the base.
    for (fgui::FusionPreset p : presets) {
        const FusionHyperparams base = sentinelBase();
        const FusionHyperparams out = fgui::applyFusionPreset(base, p);
        CHECK(out.sr_scale == base.sr_scale, "preset carries sr_scale from the base");
        CHECK(out.tsdf.origin.isApprox(base.tsdf.origin), "preset carries origin from the base");
    }

    // Documented per-preset field values (the exact set the widget used to mutate).
    {
        const auto h = fgui::applyFusionPreset(sentinelBase(), fgui::FusionPreset::kHelmet);
        CHECK_NEAR(h.tsdf.voxel_size, 0.003f, "helmet voxel");
        CHECK(h.tsdf.resolution == 512, "helmet resolution");
        CHECK_NEAR(h.tsdf.truncation, 0.010f, "helmet truncation");
        CHECK_NEAR(h.max_depth, 1.5f, "helmet max depth");
        CHECK_NEAR(h.icp.dist_threshold, 0.05f, "helmet icp dist");
        CHECK(h.icp.max_iterations[2] == 20 && h.icp.max_iterations[1] == 15 &&
                  h.icp.max_iterations[0] == 10,
              "helmet iteration ladder");
        CHECK_NEAR(h.min_depth, 0.42f, "helmet leaves min depth from the base");
        CHECK_NEAR(h.tsdf.max_weight, 200.0f, "helmet leaves max weight from the base");
    }
    {
        const auto h = fgui::applyFusionPreset(sentinelBase(), fgui::FusionPreset::kChair);
        CHECK_NEAR(h.tsdf.voxel_size, 0.008f, "chair voxel");
        CHECK_NEAR(h.max_depth, 3.0f, "chair max depth");
    }
    {
        const auto h = fgui::applyFusionPreset(sentinelBase(), fgui::FusionPreset::kRoom);
        CHECK_NEAR(h.tsdf.voxel_size, 0.030f, "room voxel");
        CHECK_NEAR(h.max_depth, fgui::kDeviceMaxDepthMeters,
                   "room max depth is clamped to the device ceiling, not 8 m");
        CHECK_NEAR(h.icp.dist_threshold, 0.20f, "room icp dist");
    }
    {
        const auto h = fgui::applyFusionPreset(sentinelBase(), fgui::FusionPreset::kHuman);
        CHECK_NEAR(h.min_depth, 0.5f, "human min depth");
        CHECK_NEAR(h.max_depth, 2.5f, "human max depth");
        CHECK_NEAR(h.tsdf.max_weight, 64.0f, "human max weight");
    }
}

// ---- Section 7: overlap band mapping, icp_valid_model == -1 first ----
void sectionOverlapBandMapping() {
    {
        const auto d = fgui::computeOverlapDisplay(-1.0f, -1);
        CHECK(d.text == "--", "not-measured overlap renders as --");
        CHECK(d.band == fgui::MetricsBand::kUnknown,
              "not-measured is UNKNOWN, never a premature warming colour");
    }
    {
        // valid_model == -1 dominates even when a stale positive overlap leaked in.
        const auto d = fgui::computeOverlapDisplay(50.0f, -1);
        CHECK(d.text == "--", "valid_model == -1 forces -- regardless of overlap");
        CHECK(d.band == fgui::MetricsBand::kUnknown, "valid_model == -1 is UNKNOWN band");
    }
    {
        const auto d = fgui::computeOverlapDisplay(-1.0f, 10000);
        CHECK(d.text == "--" && d.band == fgui::MetricsBand::kUnknown,
              "negative overlap is UNKNOWN");
    }
    {
        const auto d = fgui::computeOverlapDisplay(2.0f, 100);
        CHECK(d.band == fgui::MetricsBand::kWarming, "young measured volume is WARMING");
        CHECK(d.text.find("building") != std::string::npos,
              "warming readout says building, not a bare percentage");
    }
    {
        const auto d = fgui::computeOverlapDisplay(50.0f, 10000);
        CHECK(d.band == fgui::MetricsBand::kGood, "overlap > 40 is GOOD");
        CHECK(d.text.find('%') != std::string::npos, "healthy readout carries a percentage");
    }
    {
        CHECK(fgui::computeOverlapDisplay(40.0f, 10000).band == fgui::MetricsBand::kWarn,
              "overlap exactly 40 is WARN (threshold is exclusive)");
        CHECK(fgui::computeOverlapDisplay(20.0f, 10000).band == fgui::MetricsBand::kWarn,
              "overlap 20 is WARN");
        CHECK(fgui::computeOverlapDisplay(5.0f, 10000).band == fgui::MetricsBand::kBad,
              "overlap <= 10 is BAD");
    }
    {
        // Warmup boundary: 4999 warmup, 5000 measured.
        CHECK(fgui::computeOverlapDisplay(50.0f, 4999).band == fgui::MetricsBand::kWarming,
              "valid_model just under 5000 is still warming");
        CHECK(fgui::computeOverlapDisplay(50.0f, 5000).band == fgui::MetricsBand::kGood,
              "valid_model at 5000 leaves the warming band");
    }
}

// ---- Section 8: band cache transitions, not per-tick re-polish ----
void sectionBandCache() {
    fgui::BandCache<fgui::MetricsBand> overlap;
    // First apply always primes (one initial polish), then only transitions.
    CHECK(overlap.apply(fgui::MetricsBand::kUnknown), "first apply primes the band");
    CHECK(!overlap.apply(fgui::MetricsBand::kUnknown), "same band -> no re-polish");
    CHECK(!overlap.apply(fgui::MetricsBand::kUnknown), "still same -> still no re-polish");
    CHECK(overlap.apply(fgui::MetricsBand::kWarming), "Unknown -> Warming transitions");
    CHECK(!overlap.apply(fgui::MetricsBand::kWarming), "Warming held -> no re-polish");
    CHECK(overlap.apply(fgui::MetricsBand::kGood), "Warming -> Good transitions");
    CHECK(overlap.value() == fgui::MetricsBand::kGood, "cache tracks the latest band");
    // Returning to a previously-seen band is still a transition (it differs from
    // the immediately previous band), so the style is re-applied truthfully.
    CHECK(overlap.apply(fgui::MetricsBand::kWarming), "Good -> Warming transitions again");

    fgui::BandCache<bool> status;
    CHECK(status.apply(false), "first status apply primes");
    CHECK(!status.apply(false), "unchanged status -> no re-polish");
    CHECK(status.apply(true), "OK/LOST flip transitions");
    CHECK(!status.apply(true), "held status -> no re-polish");
    CHECK(status.value() == true, "status cache tracks latest");
}

} // namespace

int main() {
    sectionDefaultsPass();
    sectionDepthOrdering();
    sectionDeviceDepthCeiling();
    sectionTruncationVoxelRatio();
    sectionFinitenessAndPositivity();
    sectionPresetStaging();
    sectionOverlapBandMapping();
    sectionBandCache();

    std::printf("fusion_ui_validation_contract: %d checks, %d failures\n", g_checks, g_failures);
    if (g_failures != 0) {
        std::printf("fusion_ui_validation_contract: FAIL\n");
        return 1;
    }
    std::printf("fusion_ui_validation_contract: PASS\n");
    return 0;
}
