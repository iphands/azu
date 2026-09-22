#pragma once

// Pure, Qt-free UI state + validation model for the fusion hyperparameters
// panel and the metrics style bands (big-fix Todo 29).
//
// This header carries NO dependency on Qt Widgets, OpenGL, a display, a Kinect
// device or any GPU backend: it is built on app::FusionHyperparams (Eigen plus
// the backend-independent TSDF/ICP param structs) and the standard library
// only, so it compiles and is unit-tested inside the Qt-free azu_test_core
// lane. The Qt widgets (ControlPanel / MetricsPanel) translate their spin-box
// and label state to/from this layer and RENDER the returned messages/bands;
// they are NOT the authority on what is valid or which band applies.
//
// Single source of truth for the tunables and their initial values stays
// FusionHyperparams::defaults(); this layer never re-lists a default, it only
// validates and stages values that flow through that one struct.

#include "app/FusionHyperparams.h"

#include <string>
#include <vector>

namespace kfusion {
namespace gui {

// Device-usable depth ceiling. The Kinect v1 sensor returns meaningful depth
// only out to ~4 m; beyond that the raw codes saturate at the invalid/2047
// sentinel, so a max_depth above this reconstructs nothing and is rejected
// rather than silently accepted. Raise ONLY when a device with a genuinely
// longer usable range is actually supported.
inline constexpr float kDeviceMaxDepthMeters = 4.0f;

// A TSDF surface band is only meaningful when truncation is a small multiple of
// the voxel size. Below the low bound the band is thinner than a voxel (holes,
// quantization); above the high bound it integrates too far off-surface and
// smears detail. This is the canonical KinectFusion / Open3D 2..8x guidance.
inline constexpr float kTruncationVoxelRatioMin = 2.0f;
inline constexpr float kTruncationVoxelRatioMax = 8.0f;

// A volume younger than this many valid model correspondences legitimately
// overlaps little, so its overlap readout is shown as neutral "warming" rather
// than a coloured healthy/lost band.
inline constexpr int kOverlapWarmupModelPoints = 5000;

struct FusionValidationResult {
    bool ok = true;
    // One truthful, user-visible line per violated rule. Collected so the UI
    // can show every problem at once instead of the first one only.
    std::vector<std::string> messages;

    void addError(std::string msg) {
        ok = false;
        messages.push_back(std::move(msg));
    }
};

/**
 * Validate every cross-field relationship the widgets cannot enforce on their
 * own (a spin box only bounds its single field). Rules, all reported:
 *   * finiteness of every float tunable (NaN/Inf rejected)
 *   * 0 < min_depth < max_depth <= kDeviceMaxDepthMeters (strict: equal and
 *     inverted ranges are both rejected)
 *   * resolution >= 1, voxel_size > 0, max_weight > 0
 *   * truncation/voxel ratio within
 *     [kTruncationVoxelRatioMin, kTruncationVoxelRatioMax]
 *   * positive, finite ICP distance/angle thresholds
 * FusionHyperparams::defaults() must pass. This is the headless authority the
 * MainWindow apply handler consults; the widgets only render the outcome.
 */
FusionValidationResult validateFusionHyperparams(const app::FusionHyperparams& h);

// FusionHyperparams::defaults() is the ONE seed source for the panel. The
// widget seeds itself through the same setHyperparams() path it uses for
// presets, so no parallel default table can drift from the controller's.
inline app::FusionHyperparams uiDefaultHyperparams() {
    return app::FusionHyperparams::defaults();
}

enum class FusionPreset {
    kCustom = 0,
    kHelmet = 1,
    kChair  = 2,
    kRoom   = 3,
    kHuman  = 4,
};

/**
 * Pure preset staging: returns the COMPLETE final state for `preset` layered
 * over `base`, produced as one value. ControlPanel stages this whole value and
 * applies it once, so applying a preset while capture is running can never
 * leave the UI or the controller half-written. kCustom returns `base`
 * unchanged. The field set matches what the widget used to mutate inline, and
 * the depth values are clamped to the device ceiling like any other input.
 */
app::FusionHyperparams applyFusionPreset(const app::FusionHyperparams& base, FusionPreset preset);

// ---- MetricsPanel style-band state machine (Qt-free) ----

enum class MetricsBand {
    kUnknown, // not measured yet -> "--"; deliberately NOT a coloured band
    kWarming, // young-but-measured volume ("building...") -> neutral grey
    kGood,    // overlap > 40%
    kWarn,    // 10% < overlap <= 40%
    kBad,     // overlap <= 10%
};

struct OverlapDisplay {
    std::string text; // label text: "--", "building... 1.2k", "63 %"
    MetricsBand band; // style band the label should carry
};

/**
 * Pure overlap rendering. A negative icp_valid_model (or a negative overlap,
 * the documented "ICP has not run yet" contract) is the UNKNOWN state: text
 * "--" and NO warming styling, so a panel that has never measured never shows
 * a coloured band. Warming is reserved for a young-but-measured volume whose
 * valid_model is in [0, kOverlapWarmupModelPoints).
 */
OverlapDisplay computeOverlapDisplay(float overlap_pct, int valid_model);

/**
 * A band cache compares the last applied band to the new one and reports a
 * transition only when it actually changes, so the panel re-polishes a label on
 * a band CHANGE and not on every 5 Hz metrics tick. The first apply() always
 * reports a transition so the initial style is established exactly once. The
 * label text is owned by the caller and updated every tick; only the style
 * band is cached here.
 */
template <typename BandT>
class BandCache {
public:
    // Returns true iff `next` differs from the last applied band, i.e. the
    // caller must set the style property and (un)polish the widget once.
    bool apply(BandT next) {
        if (!primed_) {
            primed_  = true;
            current_ = next;
            return true;
        }
        if (next == current_) {
            return false;
        }
        current_ = next;
        return true;
    }

    BandT value() const { return current_; }

private:
    BandT current_{}; // value-initialised (enum -> first enumerator, bool -> false)
    bool  primed_ = false;
};

} // namespace gui
} // namespace kfusion
