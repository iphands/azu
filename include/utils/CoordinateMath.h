#pragma once

#include <cmath>
#include <limits>

namespace kfusion {
namespace utils {

/**
 * Deterministic, exception-free float -> integer coordinate conversion.
 *
 * Writes std::floor(value) to *out and returns true when `value` is finite and
 * its floor is representable by `int`. Returns false (leaving *out untouched)
 * for non-finite input or a floor outside the `int` range.
 *
 * This is the single rounding primitive the CPU coordinate paths share: the ICP
 * model-pixel projection (`src/tracking/ICPTracker.cpp`) and the TSDF world ->
 * voxel map (`src/tsdf/TSDFVolume.cpp` `worldToVoxel`). Routing both through it
 * means neither can resurrect `static_cast<int>` truncation (which disagrees
 * with `std::floor` for negative operands) or the undefined cast of a NaN/Inf.
 *
 * The range test runs in `double`: the float -> double widening is exact, the
 * floor is exact, and INT_MIN / INT_MAX are exactly representable in a double,
 * so the comparison is airtight and the final `static_cast<int>` can neither
 * truncate nor overflow. No exception is thrown on any input.
 */
inline bool floorToInt(float value, int* out) {
    // The double range check is only airtight while both int extremes survive a
    // double round trip. True for every real target (32-bit int, 53-bit double
    // significand); the asserts turn a hypothetical exotic platform into a
    // compile-time failure instead of silent mis-rounding.
    static_assert(static_cast<int>(static_cast<double>(std::numeric_limits<int>::min())) ==
                      std::numeric_limits<int>::min(),
                  "int::min() must be exactly representable in double");
    static_assert(static_cast<int>(static_cast<double>(std::numeric_limits<int>::max())) ==
                      std::numeric_limits<int>::max(),
                  "int::max() must be exactly representable in double");

    if (out == nullptr || !std::isfinite(value)) {
        return false;
    }
    const double floored = std::floor(static_cast<double>(value));
    if (floored < static_cast<double>(std::numeric_limits<int>::min()) ||
        floored > static_cast<double>(std::numeric_limits<int>::max())) {
        return false;
    }
    *out = static_cast<int>(floored);
    return true;
}

} // namespace utils
} // namespace kfusion
