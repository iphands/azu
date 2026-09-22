// FusionUiModel.cpp — real (non-header) definitions for the Qt-free fusion UI
// validation / preset-staging / metrics-band model (big-fix Todo 29).
//
// Compiled into azu_test_core (an explicit source element, mirroring the
// Camera.cpp precedent) so the headless CPU test drives THIS translation unit,
// and into the KinectFusionQt target via the src/gui glob so the widgets share
// the exact same logic instead of a copy. No Qt, no OpenGL, no device, no GPU:
// include only FusionHyperparams (Eigen-based) and the standard library.

#include "gui/FusionUiModel.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace kfusion {
namespace gui {

namespace {

bool isFinite(float v) { return std::isfinite(v); }

// Trim a bound to one decimal for the user-facing message ("4.0", "2.0") so the
// authoritative constant, not a hand-typed literal, is what the message quotes.
std::string fmtBound(float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", static_cast<double>(v));
    return std::string(buf);
}

// Format helpers kept here so the band text is produced by the same code the
// widgets render, never re-derived at the call site.
std::string formatBuilding(int valid_model) {
    // Matches the previous QStringLiteral("building\u2026 %1k").arg(n/1000,0,'f',1)
    // byte-for-byte (the U+2023-free ellipsis is the UTF-8 sequence below).
    char buf[64];
    std::snprintf(buf, sizeof(buf), "building\u2026 %.1fk",
                  static_cast<double>(valid_model) / 1000.0);
    return std::string(buf);
}

std::string formatPercent(float pct) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.0f %%", static_cast<double>(pct));
    return std::string(buf);
}

} // namespace

FusionValidationResult validateFusionHyperparams(const app::FusionHyperparams& h) {
    FusionValidationResult r;

    const float min_d  = h.min_depth;
    const float max_d  = h.max_depth;
    const float voxel  = h.tsdf.voxel_size;
    const float trunc  = h.tsdf.truncation;
    const float weight = h.tsdf.max_weight;
    const float dist   = h.icp.dist_threshold;
    const float angle  = h.icp.angle_threshold;

    // 1) Finiteness. Any NaN/Inf poisons a kernel or an eigen solve downstream,
    //    so reject it with one clear line before cross-field arithmetic.
    if (!isFinite(min_d) || !isFinite(max_d)) {
        r.addError("Depth range must be finite numbers (no NaN or infinity).");
    }
    if (!isFinite(voxel) || !isFinite(trunc)) {
        r.addError("Voxel size and truncation must be finite numbers.");
    }
    if (!isFinite(weight)) {
        r.addError("Max weight must be a finite number.");
    }
    if (!isFinite(h.tsdf.origin.x()) || !isFinite(h.tsdf.origin.y()) ||
        !isFinite(h.tsdf.origin.z())) {
        r.addError("Volume origin must be finite numbers.");
    }
    if (!isFinite(dist) || !isFinite(angle)) {
        r.addError("ICP distance and angle thresholds must be finite numbers.");
    }

    // 2) Positive, ordered, device-bounded depth band. Reported independently of
    //    finiteness so an infinite max_depth still surfaces the "too far" truth.
    if (isFinite(min_d) && min_d <= 0.0f) {
        r.addError("Depth minimum must be greater than zero.");
    }
    if (isFinite(min_d) && isFinite(max_d) && !(min_d < max_d)) {
        r.addError("Depth minimum must be strictly less than depth maximum.");
    }
    if (isFinite(max_d) && max_d > kDeviceMaxDepthMeters) {
        r.addError("Depth maximum exceeds the sensor's usable range (max " +
                   fmtBound(kDeviceMaxDepthMeters) + " m).");
    }

    // 3) Volume geometry positivity (guarded so a non-finite/zero voxel does
    //    not turn the ratio check into a divide-by-zero).
    if (h.tsdf.resolution < 1) {
        r.addError("Volume resolution must be at least 1.");
    }
    if (isFinite(voxel) && voxel <= 0.0f) {
        r.addError("Voxel size must be greater than zero.");
    }
    if (isFinite(weight) && weight <= 0.0f) {
        r.addError("Max weight must be greater than zero.");
    }

    // 4) Truncation/voxel ratio only when both are finite and voxel > 0.
    if (isFinite(voxel) && isFinite(trunc) && voxel > 0.0f) {
        const float ratio = trunc / voxel;
        if (ratio < kTruncationVoxelRatioMin) {
            r.addError("Truncation is too thin for the voxel size (use at least " +
                       fmtBound(kTruncationVoxelRatioMin) + "x voxel size).");
        } else if (ratio > kTruncationVoxelRatioMax) {
            r.addError("Truncation is too thick for the voxel size (keep it within " +
                       fmtBound(kTruncationVoxelRatioMax) + "x voxel size).");
        }
    }

    // 5) ICP thresholds positive.
    if (isFinite(dist) && dist <= 0.0f) {
        r.addError("ICP distance threshold must be greater than zero.");
    }
    if (isFinite(angle) && angle <= 0.0f) {
        r.addError("ICP angle threshold must be greater than zero.");
    }

    return r;
}

app::FusionHyperparams applyFusionPreset(const app::FusionHyperparams& base, FusionPreset preset) {
    if (preset == FusionPreset::kCustom) {
        return base;
    }

    // Start from the caller's current state and layer the preset's documented
    // fields onto it. This returns ONE complete value: the caller writes the
    // panel and the controller once, so a preset can never be applied field by
    // field mid-capture.
    app::FusionHyperparams h = base;

    switch (preset) {
    case FusionPreset::kHelmet:
        h.tsdf.voxel_size = 0.003f;
        h.tsdf.resolution = 512;
        h.tsdf.truncation = 0.010f;
        h.max_depth       = 1.5f;
        h.icp.dist_threshold  = 0.05f;
        h.icp.angle_threshold = 45.0f;
        h.icp.max_iterations[2] = 20;
        h.icp.max_iterations[1] = 15;
        h.icp.max_iterations[0] = 10;
        break;
    case FusionPreset::kChair:
        h.tsdf.voxel_size = 0.008f;
        h.tsdf.resolution = 256;
        h.tsdf.truncation = 0.025f;
        h.max_depth       = 3.0f;
        h.icp.angle_threshold = 45.0f;
        h.icp.max_iterations[2] = 20;
        h.icp.max_iterations[1] = 15;
        h.icp.max_iterations[0] = 10;
        break;
    case FusionPreset::kRoom:
        h.tsdf.voxel_size = 0.030f;
        h.tsdf.resolution = 256;
        h.tsdf.truncation = 0.100f;
        h.icp.dist_threshold  = 0.20f;
        h.icp.angle_threshold = 60.0f;
        // The old Room preset asked for 8 m; the device cannot measure past the
        // usable ceiling, so the preset now targets exactly that ceiling.
        h.max_depth       = kDeviceMaxDepthMeters;
        h.icp.max_iterations[2] = 30;
        h.icp.max_iterations[1] = 20;
        h.icp.max_iterations[0] = 10;
        break;
    case FusionPreset::kHuman:
        h.tsdf.voxel_size = 0.005f;
        h.tsdf.resolution = 512;
        h.tsdf.max_weight = 64.0f;
        h.tsdf.truncation = 0.015f;
        h.min_depth = 0.5f;
        h.max_depth = 2.5f;
        h.icp.angle_threshold = 45.0f;
        h.icp.max_iterations[2] = 20;
        h.icp.max_iterations[1] = 15;
        h.icp.max_iterations[0] = 10;
        break;
    case FusionPreset::kCustom:
        break; // handled above
    }

    return h;
}

OverlapDisplay computeOverlapDisplay(float overlap_pct, int valid_model) {
    // "Not measured yet" first: a negative correspondence count or a negative
    // overlap is the documented "ICP has not run" contract. Unknown is NOT a
    // warming band, so a never-measured panel shows a neutral "--".
    if (valid_model < 0 || overlap_pct < 0.0f) {
        return OverlapDisplay{"--", MetricsBand::kUnknown};
    }
    if (valid_model < kOverlapWarmupModelPoints) {
        return OverlapDisplay{formatBuilding(valid_model), MetricsBand::kWarming};
    }
    const MetricsBand band = overlap_pct > 40.0f ? MetricsBand::kGood
                             : overlap_pct > 10.0f ? MetricsBand::kWarn
                                                   : MetricsBand::kBad;
    return OverlapDisplay{formatPercent(overlap_pct), band};
}

} // namespace gui
} // namespace kfusion
