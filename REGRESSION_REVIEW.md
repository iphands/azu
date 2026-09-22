# Regression & Stability Review: Resolved

**Backend status of this document (see `README.md#backend-status`)**
The CPU (OpenMP) backend is the only one this repository builds and tests. The
CUDA (NVIDIA) and HIP/ROCm (AMD) paths are **deferred**: neither was compiled,
linked, executed or parity-checked here, so nothing below labeled "GPU" may be
read as a verified fix. CUDA additionally has no NVIDIA hardware available and is
not actively maintained. Every GPU item is dispositioned per finding in
[`docs/CUDA_HIP_DEFERRED_CHANGES.md`](docs/CUDA_HIP_DEFERRED_CHANGES.md);
the former wording here ("CUDA deprecated — use HIP or CPU instead") implied that
HIP was a working alternative, which was never demonstrated.

Labels used in the Status lines below, with the evidence each one means:

- **verified** — a CPU CTest contract in `tests/` proves the CPU behavior.
- **CPU-fixed** — the CPU path was corrected; the wording of the original entry is
  kept, and the GPU half of it is deferred rather than claimed.
- **deferred** — CUDA/HIP only; not compiled, not runtime-tested.

This document tracks the resolution of previously identified functional regressions and logical inconsistencies introduced during the transition to the unified `Camera` system and the high-performance pipeline refactor.

## Verification status under the CPU-only QA lane

| Item | CPU status | GPU status |
|---|---|---|
| 7. GLB color export | CPU-fixed and verified: float color accumulation plus one shared quantization policy, locked by `tests/color_convergence_contract.cpp` and `tests/glb_linear_color_contract.cpp`. | deferred — the `syncToGPU` / `syncFromGPU` / `raycastKernel` / `compactPointsKernel` / `MarchingCubes` color renormalization was never compiled (`tsdf:T10`, `tsdf:T15`, `cross-backend:B3`). |
| 8. Super-resolution sharpness | CPU-fixed: the CPU conditioner calls `sr::applyCAS(..., 0.5f)` (`src/sensor/SignalConditioner_omp.cpp`), the sharpness clamp now rejects non-finite input, and `tests/sr_upscaled_contract.cpp` plus `tests/cas_border_contract.cpp` lock the CPU CAS policy. The `0.85f` default in `include/sensor/SuperResolution.h` is only an API default, not what the CPU pipeline uses. | deferred — the CUDA/HIP sharpness reduction is uncompiled (`cross-backend:B5`, `sensor:S-05`). |
| 1. UI slider desync | CPU-fixed — unified `Camera` state; `tests/camera_basis_contract.cpp` covers the shared basis/orbit math, the slider binding itself lives in Qt widgets. | n/a |
| 2. WASD movement UI sync | CPU-fixed — `OpenGLWidget::updatePhysics()` signal emission; GUI-thread code with no CPU contract. | n/a |
| 3. 'Z' axis lock | CPU-fixed — axis-lock branch in `OpenGLWidget::mouseMoveEvent`; GUI-thread code with no CPU contract. | n/a |
| 4. Frame-rate independent physics | CPU-fixed — `deltaTime` in m/s; the first-frame tick clamp is locked by `tests/camera_basis_contract.cpp`, the `QElapsedTimer` loop is not. | n/a |
| 5. Transition instability (NaN) | CPU-fixed — direct trig projection in `Camera::updateFreeFromOrbit` / `updateOrbitFromFree`; covered indirectly by the basis/orbit contracts, not by a dedicated NaN fixture. | n/a |
| 6. GPU path "broken signal" & races | Partly CPU: the `TripleBuffer` model and relocalization thresholds also serve the CPU lane (`tests/pipeline_state_contract.cpp` locks CPU publish cadence). | deferred — "triple buffering eliminated VRAM race conditions" and the CUDA CAS edge-case fix describe device code that was never built (`pipeline:PC-02`, `cross-backend:B4`, `tsdf:T9`). |

## 7. Resolved: GLB Color Export Issues (CPU and GPU Paths)
**Status: FIXED on CPU (verified by CPU contracts); GPU path deferred**
- **Problem:** GLB exports had color issues on both CPU and GPU paths. CPU path colors were too dark due to integer truncation during color averaging. GPU path colors were broken because VoxelGPU stored colors as floats in the 0-255 range instead of the normalized 0-1 range, causing incorrect interpolation and output.
- **Resolution:**
  - **CPU Path:** Changed color averaging in `TSDFVolume::integrateCPU` to use float arithmetic with `std::round()` instead of direct uint8_t truncation, preventing darkening over time.
  - **GPU Path:** Normalized colors to 0-1 range in `syncToGPU` for both CUDA and HIP, denormalized back to 0-255 in `syncFromGPU`, `raycastKernel`, `compactPointsKernel`, and `MarchingCubes` kernels. This ensures consistent color handling across the entire GPU pipeline.
- **Impact:** GLB exports have correct, vibrant colors on the CPU path, which is the
  path the color contracts exercise. The GPU claim in this entry is historical: those
  kernels are deferred and uncompiled, so their color handling is unverified.

## 8. Resolved: Super Resolution Quality Degradation
**Status: FIXED on CPU (CPU CAS policy under test); CUDA/HIP reduction deferred**
- **Problem:** Super resolution (AMD FSR 1.0 CAS) quality was worse than the native Kinect feed due to overly aggressive sharpening with a sharpness parameter of 0.85.
- **Resolution:** Reduced the CAS sharpness parameter from 0.85 to 0.5 in all signal conditioner paths (CPU OpenMP, CUDA, and HIP). This provides a more conservative sharpening that enhances detail without introducing artifacts or degrading quality. On the CPU lane this is what actually runs: `src/sensor/SignalConditioner_omp.cpp` calls `sr::applyCAS(..., 0.5f)` for both the in-place and the upscaled pass. The CUDA and HIP edits in the original entry are deferred (their files have never been compiled here).
- **Impact:** Super resolution provides subtle enhancement without over-sharpening on the CPU path, where `tests/cas_border_contract.cpp` and `tests/sr_upscaled_contract.cpp` lock the CPU CAS behavior. Whether the GPU passes sharpen the same way is unknown until a backend lane builds them.

## 1. Resolved: UI Sliders Desync (Split-State Problem)
**Status: FIXED**
- **Problem:** The `Camera` class previously maintained dual sets of variables (`azimuth`/`elevation` for Orbit and `yaw`/`pitch` for Free mode).
- **Resolution:** The `Camera` class has been refactored to use a unified state (`yaw_`, `pitch_`, `roll_`). Both Orbit and Free modes now operate on these shared variables. The UI sliders are bi-directionally bound to these values, ensuring consistency regardless of the active navigation mode.

## 2. Resolved: WASD Movement UI Sync
**Status: FIXED**
- **Problem:** Camera updates during WASD movement (physics loop) were not being reported back to the UI.
- **Resolution:** `OpenGLWidget::updatePhysics()` now explicitly emits the `cameraRotated` signal whenever the camera position or orientation changes, keeping the `ControlPanel` sliders in perfect sync with the 3D viewport.

## 3. Resolved: 'Z' Axis Lock Behavior
**Status: FIXED**
- **Problem:** The 'Z' key previously triggered "Zoom" instead of a true Z-axis translation.
- **Resolution:** The axis-lock logic in `OpenGLWidget::mouseMoveEvent` has been corrected. Holding 'Z' now performs a true 3D translation along the world Z-axis using `camera().move(Eigen::Vector3f(0.0f, 0.0f, 1.0f), ...)`, matching the 'X' and 'Y' behavior for consistent panning.

## 4. Resolved: Frame-Rate Independent Physics
**Status: FIXED**
- **Problem:** Camera movement speed was hardcoded to a fixed value per tick, causing variable speeds on different hardware.
- **Resolution:** Implemented `deltaTime` calculation using `QElapsedTimer` (via `frame_timer_.restart()`). The movement speed is now defined in meters per second (`2.0f m/s`), ensuring consistent traversal regardless of UI framerate or ICP processing load.

## 5. Resolved: Transition Instability (NaN Fix)
**Status: FIXED**
- **Problem:** State conversion using `std::asin` was prone to `NaN` errors at steep angles.
- **Resolution:** The conversion logic in `Camera::updateFreeFromOrbit` and `updateOrbitFromFree` has been simplified to use direct trigonometric projections (`cos`/`sin`) and forward vector calculation, eliminating the risky inverse trigonometric calls.

## 6. Resolved: GPU Path "Broken Signal" & Race Conditions
**Status: deferred for the GPU path (never compiled); the CPU-visible halves are CPU-fixed**
- **Problem:** The GPU pipeline suffered from intermittent tracking failures ("broken signals") and occasional crashes during high-throughput integration.
- **Resolution:** 
    - **Triple-Buffering:** A `TripleBuffer` model for `ModelFrame` was implemented, separating the "Integration" (writing), "Tracking" (reading), and "Ready" states. This eliminated VRAM race conditions where the tracker would read a partially-integrated model. *(The same buffer structure serves the CPU lane, whose publish cadence is locked by `tests/pipeline_state_contract.cpp`; the "VRAM" claim is about device memory and is deferred.)*
    - **Super Resolution (CAS) Fix:** The CUDA-accelerated CAS filter was refined to handle edge cases in the RGB feed, preventing "black pixel" artifacts that previously poisoned the ICP correspondence search. *(CUDA kernel: deferred, not compiled, not runtime-tested — `cross-backend:B5`. The CPU border/reflect behavior is canonical and CPU-tested instead.)*
    - **Relocalization Logic:** Added a specialized relocalization mode that triggers when tracking is lost, using relaxed distance/angle thresholds to recover the pose without requiring a full system reset. *(State-machine behavior reachable on CPU; the GPU-side thresholds behind it are deferred.)*

---

## Conclusion
The "Blender-like" navigation system is now technically sound and fully integrated with the UI. The "Split-State" regression has been eliminated, and the physics engine is now robust against framerate fluctuations.

Scope of that sentence, stated plainly: it is about the CPU/GUI lane. It says nothing
about CUDA or HIP, and no GPU item in this document has been compiled or executed by
the current QA lane. For the CPU test suite that does back the CPU claims, run
`ctest --test-dir build-cpu-big-fix -L cpu --output-on-failure`.
