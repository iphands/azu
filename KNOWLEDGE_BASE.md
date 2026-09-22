# KinectFusion Signal Conditioning Knowledge Base

**Backend status of this document (see `README.md#backend-status`)**
This reference describes both lanes of the architecture. Only the **CPU (OpenMP)**
lane is compiled and tested by this repository's QA. CUDA (NVIDIA) and HIP/ROCm
(AMD) are **deferred**: never compiled, linked, executed or parity-checked here.
Where a paragraph below describes GPU behavior it is a description of intent, not
a test result; the per-finding dispositions are in
[`docs/CUDA_HIP_DEFERRED_CHANGES.md`](docs/CUDA_HIP_DEFERRED_CHANGES.md). The CPU
rules that the tests actually lock are in
[`docs/CANONICAL_SEMANTICS.md`](docs/CANONICAL_SEMANTICS.md).

This document details the architectural fixes and mathematical optimizations integrated into the `signal-conditioning-pipeline` branch. The focus of this branch is to enhance the spatial resolution of the 3D asset generation process and eliminate visual and geometric artifacts caused by earlier signal-processing implementations.

## 1. Volumetric Asset Resolution Enhancement (TSDF)

**How it works:**
The core of the 3D reconstruction is the Truncated Signed Distance Function (TSDF) volume. It stores the distance to the nearest surface at discrete spatial points (voxels).
**The Change:**
The `TSDFParams.voxel_size` was reduced from `10mm` (`0.010f`) to `5mm` (`0.005f`). Because the volume operates in a 3D grid, halving the 1D dimension effectively packs $2^3 = 8$ times as many voxels into the same spatial volume.
**Impact:**
This results in a staggering $8\times$ increase in geometric resolution for the final generated `.ply` or `.glb` models, enabling high-fidelity captures of fine details like facial features and small textures that were previously smoothed over by the larger 10mm voxels.

## 2. Temporal Ghosting Resolution (EMA Filter)

**How it works:**
Kinect sensors produce significant temporal noise. An Exponential Moving Average (EMA) filter blends the current frame's depth with previous frames to smooth out this jitter in real-time.
**The Problem:**
When a pixel temporarily lost its depth tracking (due to occlusion or hardware dropouts), the EMA algorithm bypassed the pixel. However, it never *reset* its internal history buffer. When the pixel reappeared, it blended its new depth with completely outdated historical data.
**The Fix:**
The pipeline now actively detects invalid depth points and explicitly resets their historical state (`ema_buf_m_[i] = 0.0f`).
**Impact:**
Eliminates "temporal ghosting," where shapes from several seconds ago would temporarily flash or stick to moving objects.

## 3. Depth Geometry Smearing (Guided Filter)

**How it works:**
A Guided Filter uses a high-resolution 2D image (like the RGB camera feed) to guide the filtering of a lower resolution or noisier signal (like the Depth map). It ensures that depth edges exactly match color edges.
**The Problem:**
The implementation used a massive $19 \times 19$ kernel. Crucially, if the center pixel of the kernel was completely empty (no depth reading), the filter would still look at all its valid neighbors, average their depths, and assign that average to the empty center. This turned the filter into an aggressive, unconstrained hole-filler that extrapolated geometry past physical object bounds into thin air.
**The Fix:**
Added a fast-path bailout condition: if the target center pixel is invalid, the guided filter skips it entirely rather than smearing structural geometry across empty gaps.
**Impact:**
Object contours and silhouettes remain sharp, accurately representing physical bounds without smearing into the background.

## 4. Chromatic Structural Artifacts (CLAHE)

**How it works:**
Contrast Limited Adaptive Histogram Equalization (CLAHE) stretches the color contrast of an image localized within small tiles (e.g., $8 \times 8$ blocks).
**The Problem:**
The original implementation ran on disjoint $8\times8$ blocks but failed to perform bilinear interpolation along the seams where adjacent blocks met. This created an artificial "checkerboard" of stark contrast differences in the RGB feed. Because the RGB feed guides the depth filtering, these visual grid mistakes directly warped the depth mesh, causing physical blocky artifacts in the 3D model.
**The Fix:**
The blocky CLAHE implementation was stripped out. Noise reduction is handled perfectly by the existing Sub-pixel Median Blur, while Contrast Adaptive Sharpening (CAS / Super Resolution fallback) gracefully maintains sharpness without generating disjoint grid lines.
**Impact:**
Completely smooth tracking surface generation without checkerboard geometry corruption.

## 5. Thread-Safety in the Process Pipeline

**How it works:**
The system uses multiple producer-consumer worker threads (Tracking, Integration, Meshing) managed by the `PipelineController`.
**The Problem:**
Various logging throttles inside the concurrent `trackingLoop` relied on `static int` variables. In C++, static locals in worker functions can trigger Undefined Behavior (UB) or state contamination if threads are ever killed, respawned, or concurrently duplicated (as happens during a pipeline Reset/Restart).
**The Fix:**
All stateful logic (`lost_log_counter_`, `success_log_counter_`) was decoupled from static scope and promoted to private member variables localized to the exact `PipelineController` instance. The third counter this entry used to list, `ui_skip_counter_`, was dead and has since been deleted along with the unused metrics callback (big-fix Todo 32, audit `pipeline:PC-26`), so it is no longer part of this claim.
**Impact:**
Thread-safe execution and reliable state-resets when stopping and restarting the Kinect scanning UI rapidly.
# KinectFusionQt Knowledge Base

A technical reference for the real-time 3D scanning application using Kinect v1 and volumetric TSDF reconstruction.

## 1. System Architecture

The application is built around a decoupled, multi-threaded pipeline designed to maximize throughput and minimize UI latency.

### 1.1 Threading Model
The system orchestrates four primary threads managed by `PipelineController`:

| Thread | Responsibility | Synchronization |
|---|---|---|
| **Capture Thread** | Interfacing with `libfreenect`. Polls hardware and populates circular pool. | **Zero-Copy circular pool** + `std::atomic` frame index. |
| **Tracking Thread** | Processes raw depth/RGB into `FrameData` and `FramePyramid`. Performs pose-aware ICP. | Polls `raw_queue_` via `std::condition_variable`; publishes to `integration_queue_`. Both queues are bounded and retain-latest. |
| **Integration Thread** | Integrates tracked frames into the `TSDFVolume`. Performs raycasting. | Shared/Unique locking on voxel grid. |
| **Meshing Thread**| Extracts smoothly-shaded indexed mesh from TSDF. | Parallelized weighted accumulation. |

### 1.2 Data Flow
The labels below say which lane is real today. **CPU** = compiled and covered by
the CPU CTest contracts; **deferred** = CUDA/HIP only, never compiled here.

1. **Sensor**: `KinectSensor` produces `RawFrame` (depth 11-bit, RGB). *(CPU)*
2. **Preprocessing**: depth-to-meters conversion (GPU on the deferred lane, CPU on the
   active one). CPU-based **AMD FidelityFX CAS (Super Resolution)** sharpens the RGB
   feed in-place prior to integration. *(CPU active; GPU deferred)*
3. **Tracking**: `ICPTracker` executes the Hessian reduction (CPU OpenMP is the tested
   path; the GPU variant is deferred). It compares a live `FramePyramid` against a
   `ModelFrame` raycast from the TSDF to find the 6-DOF camera pose. *(CPU active;
   "GPU-resident ModelFrame" deferred)*
4. **Integration**: `TSDFVolume` performs **pixel-parallel integration**; the CPU
   integration is the canonical, tested implementation and the GPU one is deferred.
5. **Meshing**: `MarchingCubes` extracts the mesh; the CPU table-driven extraction
   (parallel slices, serial ordered merge) is the canonical implementation and the
   multi-pass GPU extraction is deferred. *(CPU active; GPU deferred)*
6. **Rendering**: `PreviewRenderer` (OpenGL 3.3) visualizes either the live point cloud or the global mesh. *(CPU/GUI)*

---

## 2. Module Documentation

### 2.1 Sensor Module (`include/sensor/`, `src/sensor/`)

#### KinectSensor.h / .cpp
**Class**: `KinectSensor`
- **Purpose**: Low-level interface to Kinect v1 hardware via `libfreenect`.
- **Key Functions**:
    - `init()`: Initializes `libfreenect` context and circular buffer pool.
    - `start()`/`stop()`: Controls the 30 FPS capture thread.
    - `captureLoop()`: Continuously polls `freenect_process_events`.
- **Concurrency**: Uses a **circular pool of 8 `shared_ptr<RawFrame>`** buffers to decouple hardware interrupts from pipeline processing. This achieves zero-copy frame passage.

#### FrameData.h / .cpp
**Structs**: `FrameData`, `FramePyramid`
- **Purpose**: Storage for processed per-frame geometry.
- **Key Functions**:
    - `buildFrameData()`: Converts 11-bit depth to meters, back-projects to 3D camera space, and initializes RGB.
    - `computeNormals()`: Approximates surface normals using cross-products of neighboring vertices.
    - `buildFramePyramid()`: Generates 3-level downsampled pyramid (1x, 0.5x, 0.25x) for multi-resolution ICP.

#### Preprocessor.h / .cpp
- **The Unified Preprocessing Backend**: declares a CPU (OpenMP) path and a
  GPU-labelled path. Only the CPU path is compiled by this repository's QA; the
  GPU-labelled class is constructed but its `process()` is never reached in a
  CPU-only build.
- **Selection Logic**: the backend is chosen at *configure* time by the CMake
  `GPU_BACKEND` probe, not at runtime; in a `GPU_BACKEND=CPU` build no GPU code exists
  to select. At runtime `makePreprocessor()` resolves `Auto` to CPU when no GPU is
  present, and a `CUDA`/`HIP` request likewise degrades to CPU — a CPU-only behavior
  locked by `tests/sr_upscaled_contract.cpp`. `GPU_BACKEND=AUTO` at configure time can
  pick HIP on a host with `hipcc` installed, which is why QA always pins `CPU`.
- **Naming caveat (deferred `sensor:S-17`)**: the GPU-labelled class is named
  `CUDAPreprocessor` and reports `cuda` even when a HIP backend was requested.
- **State Management**: Orchestrates temporal state resets (EMA buffers) during pipeline relocalization or tracking recovery.

#### SuperResolution.h / .cpp / SuperResolution_cuda.cu
- **Purpose**: Real-time Super Resolution and edge enhancement of the Kinect RGB feed.
- **Algorithms**:
    - **CPU Path (active, tested)**: Pure C++ port of AMD's FidelityFX Contrast Adaptive Sharpening (FSR 1.0 CAS) math, parallelized via `OpenMP`.
    - **GPU Path (deferred)**: CUDA and HIP implementations of the CAS kernel. Not compiled, not executed, not parity-checked in this repository's QA.
- **Naming caveat (deferred `sensor:S-18`)**: the pass labelled "EASU" is a fixed separable
  4x4 Catmull-Rom bicubic resample, not edge-adaptive upsampling; see
  `docs/CANONICAL_SEMANTICS.md` for the canonical wording.
- **Performance**: Applies enhancement *in-place* ensuring geometric depth relationships map identically at 640x480.
- **Configuration**: CAS sharpness parameter set to 0.5 (conservative) to avoid over-sharpening artifacts while maintaining edge enhancement.

#### SignalConditioner.h / .cpp / SignalConditioner_cuda.cu / SignalConditioner_hip.hip
- **Purpose**: Depth denoising and structural filtering. The CPU OpenMP
  implementation (`src/sensor/SignalConditioner_omp.cpp`) is the canonical, tested
  lane; the CUDA and HIP conditioning TUs are deferred (never compiled), and the GPU
  bilateral path's missing guards are dossier rows `sensor:S-13` / `cross-backend:A21`.
- **Features**: Implements EMA temporal smoothing and Guided Filtering with the fixes detailed in the "Signal Conditioning" section above.

---

### 2.2 Tracking Module (`include/tracking/`, `src/tracking/`)

#### ICPTracker.h / .cpp
**Class**: `ICPTracker`
- **Purpose**: Implements Point-to-Plane Iterative Closest Point algorithm.
- **Key Functions**:
    - `track()`: Multiresolution entry point. Processes coarse-to-fine pyramids.
    - `trackLevel()`: Executes Point-to-Plane ICP iterations at a specific pyramid level.
    - `buildLinearSystem()`: **Pose-Aware Correspondence**. Projects live points through the previous pose and finds closest points in the raycasted model. Constructs the $6 \times 6$ Gauss-Newton Hessian. Now populates **Diagnostic Counters** (valid live/model points, projections, and filtered counts) for granular tracking failure analysis.
- **Tracking Statistics**:
    - `ICPResult` now includes `valid_live_points`, `valid_model_points`, `projected_points`, `dist_filtered`, and `angle_filtered` to distinguish between sensor blackout, occlusion, or volume-limit rejections.
- **Optimization**: All inner loops are parallelized using `OpenMP` for real-time tracking (30ms budget for CPU).

#### ICPTracker_cuda.cu / ICPTracker_hip.hip — deferred (not compiled here)
- **Purpose**: Fully GPU-resident tracking (CUDA and HIP ROCm ports). Not built by this
  repository's QA; the CPU ICP policy in `src/tracking/ICPTracker.cpp` plus
  `include/tracking/ICPShared.h` is canonical and tested.
- **Kernels**:
    - `computeHessianKernel`: Performs pose-aware point-to-plane correspondence and Jacobian accumulation in a single pass.
    - `reduceHessianKernel`: Block-based reduction for the $6 \times 6$ linear system.
    - `downsampleKernel`: GPU-based pyramid generation.
- **Optimization**: Replaces the $O(N)$ CPU reduction with $O(\log N)$ parallel reduction. Uses shared memory and atomic operations for diagnostic counter accumulation. Uses `__shfl_down_sync` in CUDA and `__shfl_down` in HIP. *(Deferred parity: audits `tracking:GPU-3`, `tracking:GPU-6`.)*

---

### 2.3 TSDF Module (`include/tsdf/`, `src/tsdf/`)

#### TSDFVolume.h / .cpp
**Class**: `TSDFVolume`
- **Purpose**: Global 3D model representation using a Voxel Grid.
- **Parameters**: 256³ resolution (default), 1cm voxel size, 3cm truncation distance.
- **Key Functions**:
    - `integrate()`: **Image-Centric Integration**. Directly ray-marches from valid depth pixels $(O(W \times H))$ into the volume, significantly faster than voxel-centric $(O(N^3))$ methods.
    - `raycast()`: Casts rays through the volume to find surface zero-crossings.
    - `interpolate()`: Trilinear interpolation of TSDF values for smooth gradients.
- **Concurrency**: Guarded by `std::shared_mutex`. Readers (Raycast/Mesh) use `shared_lock`, while Integrator uses `unique_lock`.
- **Acceleration**: Systematic use of `OpenMP` pragmas.

#### TSDFVolume_cuda.cu / TSDFVolume_hip.hip — deferred (not compiled here)
- **Purpose**: High-performance GPU-resident integration and raycasting. The CPU
  `integrateCPU` / `raycast` implementations are the canonical, tested behavior.
- **Kernels**:
    - `integrationKernel_PixelParallel`: **Image-Centric Integration**. Parallelizes over every depth pixel. Accurately updates only voxels within truncation via ray-marching. Uses `atomicAdd` for thread-safe weight accumulation.
    - `raycastKernel`: Optimized raycaster for generating GPU vertex and normal maps. Supports skipping to improve throughput.
- **Known divergence**: the backend raycast still hard-codes its near/far march bounds
  and detects only the entry crossing, both of which the CPU raycast no longer does.
  See `docs/CUDA_HIP_DEFERRED_CHANGES.md` rows `tsdf:T7`/`tsdf:T19`.

---

### 2.4 Meshing Module (`include/meshing/`, `src/meshing/`)

#### MarchingCubes.h / .cpp
- **Purpose**: Polygonization of the TSDF volume.
- **Key Functions**:
    - `extract()`: Polygonization kernel. Parallelized by voxel slices.

#### MarchingCubes_cuda.cu / MarchingCubes_hip.hip — deferred (not compiled here)
- **Purpose**: Multi-pass GPU Marching Cubes. The CPU extraction in
  `src/meshing/MarchingCubes.cpp` is the canonical implementation the mesh contracts
  lock (table-driven, exact edge-key welding, derived outward winding).
- **Passes**:
    1. **Classify**: Determines triangle counts per voxel.
    2. **Scan**: Prefix sum to compute global offsets (`thrust::exclusive_scan` for CUDA, host-side CPU loop for HIP to eliminate rocThrust dependency).
    3. **Generate**: Populates global vertex, normal, and color buffers.
- **Performance**: The "< 2ms for 256³ grid" figure is a historical measurement from the
  branch that wrote it. It has not been reproduced here — no backend binary was built or
  run — and it is not a CPU-lane number. Backend winding and table divergence are rows
  `meshing:B5` and `meshing:D1` in `docs/CUDA_HIP_DEFERRED_CHANGES.md`.

#### MeshData.h
- **Struct**: `MeshData` (Positions, Normals, Colors, Indices).
- **Class**: `SharedMesh`: A thread-safe container with versioning to pass new meshes from the extraction thread to the rendering thread without stalling.

---

### 2.5 Rendering Module (`include/rendering/`, `src/rendering/`)

#### PreviewRenderer.h
- **Purpose**: Core OpenGL 3.3 renderer.
- **Modes**: `PointCloud` (live-tracking debug) and `Mesh` (global reconstruction).
- **Optimization**: Normal matrices are pre-computed on the CPU to avoid expensive `transpose(inverse())` calls in the vertex shader.

#### Camera.h / .cpp
- **Unified Camera System**: Supports both **Orbit** (Blender-like) and **Free** (WASD flight) navigation modes.
- **Unified State**: Maintains a single set of Euler angles (`yaw_`, `pitch_`, `roll_`) and position, ensuring seamless transitions between modes.
- **Physics Engine**: Implements frame-rate independent movement using `deltaTime`, providing consistent traversal speed across varying hardware performance.
- **Interaction**: Features axis-locked panning (X, Y, Z) and dedicated UI slider binding.

#### NavigationGizmo.h / .cpp
- **Interactive 3D Widget**: A custom Qt widget that visualizes the camera orientation using a 3D axis gizmo.
- **Interaction**: Supports clicking and dragging to rotate the camera, with colors matching industry standards (X: Red, Y: Green, Z: Blue).
- **Sorting**: Implements painter's algorithm (Z-sorting) for correct occlusion of axis elements in 2D space.

---

### 2.6 Application & GUI Modules

#### PipelineController.h
- **The Orchestrator**: Contains the loop logic for all pipeline threads.
- Handles the state machine: `Idle` -> `Running` -> `TrackingLost` -> `Error`.
- Manages inter-thread communication via `std::queue` and `std::condition_variable`.
- **Diagnostic Logging**: Implements structured failure logging in the tracking loop. When `--verbose` is enabled, outputs a detailed breakdown of point-matching performance to help identify sensor vs. geometry issues.

#### MainWindow.h
- Qt6-based main window (`find_package(Qt6 6.2 COMPONENTS Widgets OpenGL OpenGLWidgets)`;
  Qt5 support is dropped and `QOpenGLWidget` now comes from `Qt6::OpenGLWidgets`).
- Uses a `QTimer` to refresh metrics (FPS, error, usage) at a fixed rate from the `PipelineController`.

---

## 3. Coordinate Systems

| Frame | Axes Definition |
|---|---|
| **Kinect Camera** | X: Right, Y: Down, Z: Forward (Hand: Right) |
| **TSDF World** | Same as Initial Camera Frame |
| **GLB Export** | X: Right, Y: Up, Z: Forward (Standard GLTF) |

> [!NOTE]
> During GLB export, the `GLBExporter` applies a `Y -> -Y` flip to convert from Kinect space (Y-down) to the industry-standard Y-up orientation.

---

## 4. Interaction & Controls

| Input | Action | Mode |
|---|---|---|
| **LMB Drag** | Rotate | Orbit / Free (Head-turn) |
| **RMB Drag** | Pan | Orbit / Free |
| **Wheel** | Zoom / Move | Orbit (Dist) / Free (Fwd/Bwd) |
| **W/A/S/D/Q/E**| Traversal | Free Mode |
| **Tab** | Toggle Mode | Orbit <-> Free |
| **F** | Focus Target | Orbit Mode (Reset to origin) |
| **X/Y/Z** | Axis Lock | Panning (Hold while dragging RMB) |

---

## 5. Resolved Architectural & Stability Improvements

Following a rigorous adversarial review, the system has been hardened for production stability:

### 5.1 Hardened Concurrency & Resource Management
- **RAII GPU Memory Wrappers (deferred lane)**: `utils::CudaUniquePtr` and
  `utils::HipUniquePtr` give `TSDFVolume`, `ICPTracker` and `ModelFrame` RAII device
  allocations, replacing manual `free`. The wrappers exist and are reviewed, but every
  call site is inside a never-compiled backend TU, so this is recorded as deferred, not
  verified; the "zero known leaks" claim is a CPU-lane statement only.
- **Automated Frame Recycling**: Refactored `KinectSensor` and `PipelineController` to use `std::shared_ptr` with custom recyclers. Frames are automatically returned to pools when their last reference drops, preventing OOM. *(CPU lane, active.)*
- **Zero-Copy Mesh Data**: `SharedMesh` and `MeshData` now use `std::shared_ptr`. Multi-megabyte mesh transfers between extraction and rendering threads are now instantaneous pointer swaps. *(CPU lane, active.)*

### 5.2 Performance & Throughput Optimizations
- **Integration Loop Hoisting**: Eliminated millions of redundant `pose.inverse()` calls in `TSDFVolume::integrateCPU` by hoisting transforms to the method level, resulting in a **10-15x speedup**. *(CPU lane; the figure is historical from the branch that made the change, not a number this QA lane re-measures.)*
- **Unified Vertex Extraction**: `MarchingCubes` now performs on-the-fly vertex unification during extraction, reducing the welded vertex count (the original note said "mesh VRAM footprint by ~6x"; on the CPU lane there is no mesh VRAM, and the GPU claim is deferred).
- **Ping-Pong Model Buffers**: Implemented efficient double-buffering for the tracking model in `PipelineController`, allowing tracking to proceed on a stable snapshot while integration updates the back-buffer. *(CPU lane, active; `tests/pipeline_state_contract.cpp` covers the CPU publish behavior.)*

### 5.3 Build System & Engineering
- **Backend Selection Is Explicit**: `CMakeLists.txt` takes `-DGPU_BACKEND=AUTO|HIP|CUDA|CPU`
  and `scripts/build.sh` wraps it. This project's QA pins `-DGPU_BACKEND=CPU`; `AUTO` is
  avoided because a `hipcc` on `PATH` makes it select HIP (deferred audit `cross-backend:C3`,
  which also records the hard-coded `gfx90c` architecture).
- **Qt6 Discovery**: `CMakeLists.txt` requires `Qt6` 6.2+ with `Widgets`, `OpenGL` and
  `OpenGLWidgets` and fails with a fatal error otherwise. The earlier `Qt5` discovery
  paths are gone — Qt5 support is dropped.
- **Ninja Build Integration**: Transitioned the primary build recommendation to **Ninja**, significantly reducing incremental compile times and improving build reliability on multi-core systems.
- **Modernized Qt Build Pipeline**: Transitioned to `CMAKE_AUTOMOC`, `CMAKE_AUTOUIC`, and `CMAKE_AUTORCC`. All project headers are now explicitly tracked in the build target, ensuring robust Meta-Object Compiler (MOC) generation.
- **CPU Test Lane**: `ctest --test-dir build-cpu-big-fix -L cpu --output-on-failure`
  runs the labelled CPU contracts; `scripts/test-cpu-big-fix.sh` is the gate that
  configures, builds and runs them. There is no equivalent CUDA/HIP lane in this plan.

### 5.4 GPU Path Hardening & AMD ROCm Support — DEFERRED, NOT VERIFIED
Everything in this subsection describes code that this repository's QA has never
compiled or run. It is kept as the historical record of the ROCm branch and as input
to a future backend lane; each item's current disposition and future acceptance test
is a row in `docs/CUDA_HIP_DEFERRED_CHANGES.md`.
- **ROCm/HIP Migration**: Parity with the legacy CUDA path was *attempted* for AMD iGPUs (e.g., Vega 7/5650u), using HIP primitives (`__shfl_down`), native hipStream management, and host-side prefix scans instead of rocThrust. **Not verified here**: the five `*_hip.hip` translation units have never been compiled on this host (audit `cross-backend:C1`), and the HIP path still carries CUDA names (audit `sensor:S-17`).
- **Triple-Buffer Model Management**: Implemented a robust triple-buffering system for `ModelFrame` (Front/Back/Ready indices) in `PipelineController`. This eliminates race conditions between the Integration thread (writing the new model) and the Tracking thread (reading the stable model). *(The structure is shared with the CPU lane, where it is exercised; the VRAM race claim itself is backend-only and deferred — `pipeline:PC-02`.)*
- **CUDA/Host Sync Points**: Optimized synchronization barriers during pipeline Reset/Restart to prevent `cudaErrorInvalidDevice` or `cudaErrorIllegalAddress` during rapid UI state transitions. *(CUDA only.)*
- **Kernel Robustness**: Fixed boundary conditions in `TSDFVolume_cuda.cu` and `ICPTracker_cuda.cu` that previously caused intermittent tracking failure or visual artifacts ("broken signals") when the sensor approached volume edges. *(CUDA only.)*

---

## 6. Current Status & Future Roadmap

- **Status**: **STABILIZED on the CPU lane**. The CPU pipeline is the tested lane
  (`ctest --test-dir build-cpu-big-fix -L cpu --output-on-failure`); the memory-leak and
  throughput statements below that concern GPU resources are unverified claims about
  deferred code.
- **Verified on CPU (by the CPU CTest contracts)**:
    - [x] Eliminated redundant matrix inversions in `TSDFVolume`.
    - [x] Implemented zero-copy frame and mesh passing.
    - [x] Unified vertex extraction in `MarchingCubes` (CPU weld = exact edge-key identity).
    - [x] CPU depth domain, CAS border mode, upscaled-RGB availability, color policy, mesh winding/frontier rules.
- **Deferred, NOT verified (no CUDA/HIP compile in this plan)**:
    - [ ] Full ROCm/HIP backend for AMD support (`gfx90c` target is hard-coded, not queried) and the CUDA path, each behind its own future compile gate.
    - [ ] RAII guarantees for GPU resources via `CudaUniquePtr` / `HipUniquePtr` (code present, uncompiled).
    - [ ] Backend parity for every CPU rule listed above.
- **Future Roadmap**:
    - Implement a `VoxelHash` backend for larger scale environments.
    - Add real-time loop closure detection (Pose Graph optimization).

