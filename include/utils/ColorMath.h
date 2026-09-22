#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace kfusion {
namespace utils {

/**
 * The single CPU color-quantization policy (big-fix Todo 19).
 *
 * A CPU voxel carries its color as float sRGB in `[0, 1]`
 * (`docs/CANONICAL_SEMANTICS.md`, "Color pipeline and export"). Every boundary
 * that turns that float back into the canonical uint8 sRGB byte - the raycast
 * color output, the global point cloud, and the Marching Cubes interpolated
 * edge color - goes through this one function, so there is exactly one answer
 * to "what byte represents this float?" and no second clamp policy can grow in
 * a second file.
 *
 * Policy: one clamp, round-to-nearest, non-finite rejected.
 *   - NaN and +/-Inf return false. They are NOT saturated: a non-finite color
 *     means an upstream bug, and turning it into a plausible black or white
 *     would hide it. It is also the only way to stay defined - `static_cast` of
 *     a non-finite float to an integer is undefined behavior, and `std::min`/
 *     `std::max` alone cannot express the distinction because both discard a
 *     NaN comparison.
 *   - EVERY FINITE value is representable: a value outside `[0, 1]` is clamped
 *     into `[0, 1]` first, then rounded. An out-of-range float is normally a
 *     mild overshoot of a legitimate interpolation - the same overshoot a
 *     clamped normal or a clamped barycentric coordinate absorbs - so the
 *     boundary saturates to the nearest endpoint byte instead of refusing.
 *   - `std::lround(clamped * 255.0f)` rounds to nearest, so `0.501960f`
 *     (128/255) is 128 and `0.999f` is 255 - not the truncation that
 *     systematically darkened every fused color.
 *
 * Clamping happens BEFORE the multiply, so no scaling can overflow: the widest
 * finite input (`FLT_MAX`, `lowest()`) saturates to 255 or 0 rather than
 * producing an out-of-range product that the guard below would have to reject.
 *
 * `*out` is left untouched on every false return, so a caller cannot emit a
 * half-written triple by accident. A caller therefore only has to handle the
 * non-finite case: withdraw the pixel, skip the voxel, or drop the edge.
 *
 * Not to be confused with `export::srgbComponentToLinear`, which is deliberately
 * STRICTER - it refuses a finite out-of-range component too - because its input
 * is an already-validated uint8 sRGB byte, so an out-of-range float there can
 * only be a caller bug.
 */
inline bool srgbFloatToUint8(float value, uint8_t& out) {
    // The cast below is only provably in range while 255 is exactly a float and
    // a long is wider than a uint8_t. True on every real target; the asserts turn
    // a hypothetical exotic platform into a compile-time failure.
    static_assert(static_cast<float>(std::numeric_limits<uint8_t>::max()) == 255.0f,
                  "255 must be exactly representable as float");
    static_assert(sizeof(long) > sizeof(uint8_t),
                  "lround result must be wider than uint8_t");

    if (!std::isfinite(value)) {
        return false;
    }
    // Saturation of a finite value, as two comparisons: NaN never reaches here
    // (excluded above) and -0.0 stays a valid zero.
    const float clamped = (value < 0.0f) ? 0.0f : (value > 1.0f ? 1.0f : value);

    const long scaled = std::lround(clamped * 255.0f);
    // Unreachable for a clamped finite value (the product spans exactly [0, 255]);
    // kept as a fail-closed guard so a future change to the clamp above cannot
    // silently produce a truncating cast instead of a rejection.
    if (scaled < 0 || scaled > static_cast<long>(std::numeric_limits<uint8_t>::max())) {
        return false;
    }
    out = static_cast<uint8_t>(scaled);
    return true;
}

/**
 * The inverse normalization: a device uint8 sRGB sample becomes the canonical
 * float sRGB the CPU volume stores. Division by 255.0f is the documented
 * widening (it is exactly the glTF-independent sRGB-encoded unit value); no
 * gamma is applied here, because the volume domain IS sRGB-encoded.
 *
 * Integration uses it for every incoming candidate, and CPU test fixtures use
 * it so a fixture never writes a raw byte value into a float color field.
 */
inline float srgbUint8ToFloat(uint8_t value) {
    return static_cast<float>(value) / 255.0f;
}

} // namespace utils
} // namespace kfusion
