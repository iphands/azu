#pragma once

namespace kfusion {
namespace sensor {

/**
 * @brief Canonical CPU border extension mode: half-sample reflection.
 *
 * ONE owner for the border formula used by every CPU signal-conditioning and
 * super-resolution pass (`src/sensor/SuperResolution.cpp`,
 * `src/sensor/SignalConditioner_omp.cpp`). Before big-fix Todo 21 the identical
 * three-line formula was duplicated as a file-local `reflectCoord` in both
 * translation units, so a border change had to be made twice to stay coherent.
 *
 * Exact formula (mirror without repeating the edge sample, i.e. `reflect-1`):
 *   coord < 0          -> -coord - 1
 *   coord >= extent    -> 2 * extent - coord - 1
 *   otherwise          -> coord
 *
 * Domain contract: `extent > 0` and `-extent <= coord <= 2 * extent - 1`, which
 * is exactly the window where the mapped result stays inside `[0, extent - 1]`.
 * Every caller satisfies it with margin: each pass offsets one valid in-range
 * index by its own stencil radius `r < extent`, so `coord` lands in
 * `[-r, extent - 1 + r]`, well inside the window. Outside the window the formula
 * still evaluates but no longer names a sample, so callers must not widen a
 * stencil past `radius < extent`.
 *
 * Why this is the canonical mode, and what it is NOT:
 * - For a radius-1 stencil the only out-of-range coordinates are `-1` and
 *   `extent`, and reflection maps them to `0` and `extent - 1` — exactly what a
 *   clamp would give. A radius-1 pass (the 5-tap RCAS cross, the 3x3 median)
 *   therefore CANNOT distinguish reflection from clamping. Only a stencil that
 *   reaches `-2` or `extent + 1` can, which in the CPU pipeline means the 4x4
 *   Catmull-Rom resample in `applyEASU_CPU` (it hits `-2` at the first
 *   destination border and `extent + 1` at the last, at every scale 2/3/4).
 *   `tests/cas_border_contract.cpp` proves both halves of that statement.
 * - CUDA/HIP keep their own border handling; the clamping `rcasGetPixel*` is a
 *   known divergence, deferred as `sensor:S-05` / `cross-backend:A20` in
 *   `docs/CUDA_HIP_DEFERRED_CHANGES.md`. Nothing in this header is compiled for,
 *   or references, a GPU backend.
 */
inline int cpuReflectCoord(int coord, int extent) {
    if (coord < 0) return -coord - 1;
    if (coord >= extent) return 2 * extent - coord - 1;
    return coord;
}

}  // namespace sensor
}  // namespace kfusion
