#pragma once

#include <cmath>

namespace kfusion {
namespace export_io {

/**
 * Checked sRGB -> linear decode for the GLB `COLOR_0` boundary (big-fix Todo 19).
 *
 * glTF 2.0 defines `COLOR_0` vertex colors as linear-light values, while this
 * project's whole upstream color chain is sRGB-encoded: the CPU volume stores
 * float sRGB `[0,1]`, `MeshData::colors` is uint8 sRGB, PLY stays uint8 sRGB
 * (`docs/CANONICAL_SEMANTICS.md`, "Color pipeline and export"). The exporter is
 * therefore the one place that must decode, and the old raw `uint8 / 255.0f`
 * produced a float-typed accessor carrying sRGB-encoded numbers - linear-typed
 * in name only.
 *
 * Rejection, not sanitization: NaN, +/-Inf and a finite value outside `[0, 1]`
 * all return false and leave the output untouched. The exporter refuses the
 * whole export instead of clamping a component it cannot represent, so a bug
 * upstream can never reach a .glb as a plausible-looking color.
 *
 * This is deliberately STRICTER than the CPU extraction boundary
 * `utils::srgbFloatToUint8`, which is a clamp policy: it saturates any FINITE
 * out-of-range value to the endpoint byte and rejects only non-finite input. The
 * difference is intentional. Extraction consumes an interpolated float that may
 * legitimately overshoot `[0, 1]`, so it saturates; this boundary consumes a
 * value already widened from a validated uint8 sRGB byte, which is in range by
 * construction, so anything else is a caller bug and is refused rather than
 * silently repaired.
 *
 * The arithmetic is done in double (the piecewise constant and the 2.4 exponent
 * are double-exact constants) and narrowed to float afterwards.
 *
 * Standard sRGB EOTF:
 *     c <= 0.04045  ->  c / 12.92
 *     c >  0.04045  ->  ((c + 0.055) / 1.055) ^ 2.4
 */
inline bool srgbComponentToLinear(float srgb, float& linear) {
    if (!std::isfinite(srgb)) {
        return false;
    }
    if (srgb < 0.0f || srgb > 1.0f) {
        return false;
    }
    const double c = static_cast<double>(srgb);
    const double decoded =
        (c <= 0.04045) ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
    // pow() of an in-domain base cannot be non-finite; the guard keeps the
    // promise that this function never writes a NaN or an Inf.
    if (!std::isfinite(decoded) || decoded < 0.0 || decoded > 1.0) {
        return false;
    }
    linear = static_cast<float>(decoded);
    return true;
}

/**
 * Triple form of the same checked decode. Fails as a unit: if any component is
 * rejected, no output component is written, so a caller can never emit a color
 * that mixes decoded and garbage channels.
 */
inline bool srgbColorToLinear(float r, float g, float b,
                              float& linear_r, float& linear_g, float& linear_b) {
    float lr = 0.0f, lg = 0.0f, lb = 0.0f;
    if (!srgbComponentToLinear(r, lr) ||
        !srgbComponentToLinear(g, lg) ||
        !srgbComponentToLinear(b, lb)) {
        return false;
    }
    linear_r = lr;
    linear_g = lg;
    linear_b = lb;
    return true;
}

} // namespace export_io
} // namespace kfusion
