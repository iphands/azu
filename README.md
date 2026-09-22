# Azu
<p>
<p align="center">
<img width="640" height="360" alt="manul" src="https://github.com/user-attachments/assets/e6867f31-747b-4873-9816-77e5e9fbf995" />
</p>
<p align="center">
<em>The project is named after Az, a manul who lives in Kobe Animal Kingdom. Manuls are small wild cats with long and dense light grey fur, and rounded ears set low on the sides of the head. </em>
</p>
</p>
Real-time 3D scanning application using Kinect v1, volumetric TSDF reconstruction,
and Unity-ready GLB export. Runs on Fedora with Qt6 and the **CPU (OpenMP) backend**,
which is the only backend this repository currently builds and tests. The CUDA and
HIP/ROCm GPU paths below are **deferred**: their code is present, but it is not
compiled and not runtime-tested here. See [Backend status](#backend-status).

---

## Backend status

| Backend | Status in this fork | Compiled by the current QA lane | Runtime-tested |
|---|---|---|---|
| **CPU (OpenMP)** | Active. The lane every claim below is verified against. | yes | yes — `ctest -L cpu` |
| **CUDA (NVIDIA)** | **Deferred.** Source kept, not maintained, no NVIDIA hardware here. | no | no |
| **HIP / ROCm (AMD)** | **Deferred.** Source kept, never compiled on this host. | no | no |
| `auto` | Resolves to CPU in a CPU-only build; a GPU request degrades to CPU when no GPU is present. | n/a | n/a |

Facts this section is built on, not aspirations:

- **Qt6 is required.** `CMakeLists.txt` does `find_package(Qt6 6.2 QUIET COMPONENTS
  Widgets OpenGL OpenGLWidgets)` and fails with a fatal error otherwise (Qt5 support
  is dropped; `Qt6::OpenGLWidgets` is a separate Qt6 module).
- **The only QA command in this plan is a CPU command:**

  ```bash
  ctest --test-dir build-cpu-big-fix -L cpu --output-on-failure
  ```

  The full gate that configures, builds and runs it is
  `bash scripts/test-cpu-big-fix.sh` (`-DGPU_BACKEND=CPU -DBUILD_TESTING=ON`).
- Every CUDA/HIP finding, its CPU decision, and its future acceptance test are in
  [`docs/CUDA_HIP_DEFERRED_CHANGES.md`](docs/CUDA_HIP_DEFERRED_CHANGES.md). Read
  that file before trusting or changing any GPU claim in this README.
- `build-cuda/` and `build-cpu/` are **pre-existing directories**. They are not
  evidence of anything this plan produced, and this plan never configured, built,
  cleaned or reconfigured them. No `build-cuda-big-fix/` or `build-hip-big-fix/`
  directory was created, built, or tested under this plan — those gates are
  explicitly out of scope, so no test result of theirs exists to report.

### Command-line backend selection

`--backend <name>` accepts `auto`, `cpu`, `cuda`, `gpu` (alias of `cuda`) and
`hip`, case-insensitively. An unrecognized value — or `--backend` with no value
at all — is now a hard error naming the offending value, printed before any Qt
window system is touched, and the process exits with status `2`:

```console
$ ./build-cpu-big-fix/KinectFusionQt --backend bogus
[KinectFusionQt] error: unrecognized --backend value 'bogus' (accepted: auto, cpu, cuda, gpu, hip); note cuda/gpu and hip are accepted but DEFERRED backends in this build and resolve to CPU at runtime
```

Silently falling back to `auto` was the bug (audit `pipeline:MN-01`): a typo such
as `--backend cuda` on a CPU-only build now says so instead of quietly becoming
`auto`. `cuda`/`gpu`/`hip` remain *accepted* arguments — the runtime then degrades
to CPU, and that degradation is itself CPU behavior locked by a CPU test
(`tests/sr_upscaled_contract.cpp`: "a GPU request without a GPU degrades to CPU"),
not backend proof. The same test records `sensor:S-17`: the GPU-labelled
preprocessor reports `cuda` even for a HIP request, which is why the banner above
this application can print a backend name that a HIP build would later rename.

### How the audit claims are labeled

Three labels only, used consistently across these docs:

- **verified** — a CPU CTest contract in `tests/` proves it on the CPU lane.
- **CPU-fixed** — the CPU path was corrected and locked by a CPU test; backend
  parity stays deferred, so the label never means "the GPU does this too".
- **deferred** — CUDA/HIP only: recorded in `docs/CUDA_HIP_DEFERRED_CHANGES.md`,
  not compiled, not runtime-tested.

Applied to the findings this documentation pass was scoped to:

| Audit ID | Label | What it is |
|---|---|---|
| `pipeline:MN-01` | CPU-fixed | Unknown `--backend` value silently became `auto`; now a named error with exit status `2` (`src/main.cpp`). |
| `sensor:S-14` | CPU-fixed | Stale guided-filter radius comment in `src/sensor/SignalConditioner_omp.cpp`; it advertised a radius the code no longer uses. |
| `sensor:S-17` | deferred | The HIP path's public names (`processCuda`, `cuda_stream`, `CUDAPreprocessor`) still read as CUDA. |
| `sensor:S-18` | deferred | The pass labelled EASU is a fixed separable Catmull-Rom bicubic resample, not edge-adaptive; see `docs/CANONICAL_SEMANTICS.md`. |
| `cross-backend:C1` | deferred | No real HIP compile gate exists; the five `*_hip.hip` files have never been compiled here. |
| `cross-backend:C3` | deferred | `GPU_BACKEND=AUTO` prefers HIP and hard-codes the `gfx90c` target; no device query. |
| `cross-backend:C7` | deferred | Backend comment/label hygiene across the CUDA/HIP translation units. |

Nothing in this table is claimed as backend-verified, and no row implies a CUDA or
HIP build ran.

---

## Features

- Live Kinect v1 capture via **libfreenect** (no OpenNI)
- **CPU reconstruction pipeline (active)**: OpenMP-parallel TSDF integration, ICP tracking and meshing — the paths the CPU test suite covers.
- *(deferred, not compiled here)* **High-Performance GPU Pipeline**: a fully GPU-resident architecture targeting AMD (HIP/ROCm).
- **Image-Centric TSDF**: $O(W \times H)$ pixel-parallel integration for real-time fidelity.
- *(deferred, not compiled here)* **Real-Time Super Resolution**: HIP-accelerated CAS; the **CPU OpenMP** AMD FidelityFX CAS filter is the active path.
- *(deferred, not compiled here)* **GPU-Resident ICP**: multi-resolution tracking with block-reduced Hessian construction; the CPU lane does the same math with OpenMP.
- *(deferred, not compiled here)* **Multi-Pass Marching Cubes**: parallel mesh extraction via HIP/Host-Scan; the CPU lane is the canonical, tested implementation.
- **Navigation Gizmo**: Blender-style interactive 3D axis gizmo for orientation control.
- **OpenGL 3.3** real-time preview (point cloud + mesh modes)
- **PLY** (binary) and **GLB** (Unity-ready) export via tinygltf
- Qt6 GUI with live metrics panel

---

## Getting Started (Fedora)

This guide covers the full setup for Fedora-based systems.

### 1. Prerequisites

Install core development tools and library dependencies via `dnf`:

```bash
sudo dnf install -y \
    cmake \
    git \
    ninja-build \
    gcc-c++ \
    qt6-qtbase-devel \
    libfreenect-devel \
    eigen3-devel \
    mesa-libGL-devel \
    mesa-libGLU-devel \
    libXrandr-devel \
    libXi-devel \
    libgomp \
    pkgconf-pkg-config
```

**Qt6, not Qt5.** `CMakeLists.txt` requires `Qt6` version `6.2` or newer with the
`Widgets`, `OpenGL` and `OpenGLWidgets` components and stops with a fatal error if
they are missing; Qt5 support is dropped. On Fedora those three modules all come
from `qtbase`, so `qt6-qtbase-devel` is the package that matters
(`qt6-qtbase-gui` arrives with it). Verify before building:

```bash
rpm -q qt6-qtbase-devel && ls -d /usr/lib64/cmake/Qt6Widgets /usr/lib64/cmake/Qt6OpenGLWidgets
```

#### Optional: HIP Acceleration (AMD GPUs) — deferred, not built by this project's QA
**AMD GPUs (ROCm/HIP):**
```bash
sudo dnf install rocm-hip rocm-opencl
```

Installing ROCm does **not** make the HIP path supported here. The HIP translation
units have never been compiled on this host, so there is no HIP result to quote —
`docs/CUDA_HIP_DEFERRED_CHANGES.md` records that deferral (audit `cross-backend:C1`)
along with the compile gate a future run must pass first.

**⚠️ CUDA - DEFERRED (was "DEPRECATED"):**
The CUDA backend is untested and may not compile; no NVIDIA hardware is available
here. The code remains in the codebase and is not actively maintained. The CPU
backend is the one that is built and tested. Do not read "use HIP instead" as
advice: HIP is deferred too, for the same reason (it has never been compiled).

### 2. Cloning and Setup

```bash
git clone https://github.com/Netherquark/azu.git
cd azu

# Fetch bundled dependencies (tinygltf, stb)
bash scripts/fetch_deps.sh
```

### 3. Hardware Setup (Kinect v1)

Configure udev rules to allow non-root access to the Kinect hardware:

```bash
# Copy and reload rules
sudo cp udev/99-kinect.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

# Add your user to the plugdev group
sudo groupadd -f plugdev
sudo usermod -aG plugdev $USER
# NOTE: You must log out and back in for group changes to take effect!
```

---

### 4. Building the Project

The build system supports auto-detecting your GPU, or explicitly forcing a specific
backend. It uses **CMake 3.18+** and is optimized for speed using **Ninja**. What is
*buildable* and what is *tested* are different things here: only the CPU lane has a
test suite, and only the CPU lane is built by this project's QA.

**The CPU-only QA lane (the one with tests):**
```bash
# configure + build + run every labelled CPU test, with the gate's own guards
bash scripts/test-cpu-big-fix.sh

# ...or run the tests yourself against an existing CPU build dir
cmake -S . -B build-cpu-big-fix -G Ninja -DCMAKE_BUILD_TYPE=Debug \
      -DGPU_BACKEND=CPU -DBUILD_TESTING=ON
cmake --build build-cpu-big-fix
ctest --test-dir build-cpu-big-fix -L cpu --output-on-failure
```

`-DGPU_BACKEND=CPU` is what keeps `nvcc`/`hipcc` out of the build. Do not substitute
`GPU_BACKEND=AUTO` for it in QA: with `hipcc` on `PATH`, `AUTO` can select HIP
(audit `cross-backend:C3`), which this lane must never do.

**Using the build helper (untested convenience wrapper):**
```bash
# Auto-detects HIP (AMD), then CPU fallback — the CPU-only equivalent is --cpu
./scripts/build.sh
./scripts/build.sh --cpu     # -> build-cpu/

# Backend flows below are DOCUMENTED ONLY. They are not exercised by this
# project's QA and there is no result for them to report.
./scripts/build.sh --hip     # -> build-hip/   (HIP: deferred, never compiled here)
./scripts/build.sh --cuda    # -> build-cuda/  (CUDA: deferred, never compiled here)
```

**Manual CMake:**
```bash
mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DGPU_BACKEND=CPU
ninja
```

**Note:** If you experience "undefined reference" errors after a `git pull`, it is highly recommended to perform a clean build (`./scripts/build.sh --clean`) to refresh the automated Qt metadata.

**Note:** The CMake configuration will output a diagnostic summary at the end of the `cmake` step, showing which GPU backend was selected.

---

## Usage

1. **Connect Kinect v1** via USB before launching the application.
2. Launch the scanner from your target build directory (e.g. `build-cpu/`): `QT_QPA_PLATFORM=xcb ./build-cpu/KinectFusionQt --verbose`
3. Click **▶ Start Capture** — the pipeline will begin live tracking and volume integration.
4. **Scan**: Move the Kinect slowly and steadily around your target object.
5. **View**: Toggle between **Point Cloud** and **Mesh** modes to inspect quality in real-time.
6. **Navigate**: Use **Blender-like** controls for inspection:
   - **LMB**: Orbit around target.
   - **RMB + X/Y/Z**: Axis-locked panning.
   - **Interactive Gizmo**: Click and drag the XYZ gizmo in the bottom-right to rotate.
   - **Tab**: Switch to **Free Flight** (WASD + Q/E to fly).
   - **F**: Focus back on the origin.
7. **Export**: Once satisfied, click **Export PLY** or **Export GLB**.

### Optimization Tips

- **Range**: Maintain a distance of **0.3m to 2.5m** for optimal depth precision (tunable via **Depth min/max** sliders).
- **Lighting**: Ensure consistent, non-flickering lighting for robust RGB-based ICP tracking.
- **Volume**: The default reconstruction cube is 2.56m. You can adjust the `origin` and `voxel_size` in `include/tsdf/TSDFVolume.h` for smaller objects (e.g., set `voxel_size` to 0.005 for 5mm precision).
- **Tracking Lost**: If tracking is lost (indicated in the status panel), click **Reset** to clear the volume and start a new scan.
- **Diagnostics**: Start with `--verbose` to see a detailed breakdown of tracking failures (correspondences, projections, filtering).
- **Backend choice**: `--backend cpu` is the supported lane. `--backend auto` also resolves to CPU in a CPU-only build; `cuda`/`gpu`/`hip` are accepted arguments whose runtime effect here is the documented CPU fallback. Anything else stops the launch with an error (see [Command-line backend selection](#command-line-backend-selection)).

---

## Project Structure

```text
KinectFusionQt/
├── include/           # Header files (.h)
├── src/               # Implementation files (.cpp, .cu)
├── scripts/           # Dependency & utility scripts
├── third_party/       # External libraries (populated by fetch_deps.sh)
├── udev/              # Linux hardware rules
└── CMakeLists.txt     # Main build configuration
```

---

## Architecture

The diagram below is the **historical GPU-resident design**. It is kept as the
reference for what the CUDA/HIP backends are meant to do, and every stage marked
`(deferred)` is exactly that: uncompiled and unrun under this plan, with its audit
row in `docs/CUDA_HIP_DEFERRED_CHANGES.md`. The lane that is actually built and
tested executes the same stages on CPU/OpenMP — preprocessing, pose-to-pose ICP,
TSDF integration, raycast, meshing, export — with the CPU contracts in `tests/`.

```
Kinect HW
   │
   ▼
KinectSensor (libfreenect, capture thread)
   │  RawFrame (depth 11-bit + RGB 640×480)
   ▼
Preprocessor (Unified Backend: CPU active / CUDA deferred)
   │  AMD FSR CAS Super Resolution + Denoising
   ▼
Pipeline (GPU Resident — deferred; CPU/OpenMP is what runs here)
   │
   ├──► Tracking ──────────► GPU ICP (deferred) (Hessian reduction, pose-to-pose)
   │         │                     │ updated pose
   │         └──────────────────────┤
   │                                ▼
   └──► Integration ───────► Pixel-Parallel TSDF (CUDA / HIP = deferred)
                 │                   │
                 │               Raycast (GPU = deferred) → ModelFrame (VRAM)
                 │
                 ▼
         Meshing Thread ──► GPU Marching Cubes (deferred) ──► SharedMesh
                                                             │
                                                      ┌──────┴──────┐
                                                      ▼             ▼
                                               PLYExporter    GLBExporter
                                                                    │
                                                              Unity-ready .glb
```

---

## Coordinate Systems

| Space        | Axes                     | Notes                        |
|--------------|--------------------------|------------------------------|
| Kinect depth | X right, Y down, Z fwd   | Right-handed                 |
| TSDF world   | Same as Kinect at origin | Pose tracked via ICP         |
| GLB export   | X right, Y up, Z fwd     | Right-handed (GLTF standard) |
| Unity import | X right, Y up, Z fwd     | Left-handed (Z flipped auto) |

The GLBExporter applies `Y → -Y` to convert from Kinect Y-down to GLTF Y-up.
Unity's built-in GLTF importer then handles the right-to-left-handed flip automatically.
Scale is **1 unit = 1 metre** throughout.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No Kinect devices found" | Check udev rules; run `lsusb` to confirm device visible |
| Permission denied on USB | Add user to `plugdev`; re-login |
| Tracking immediately lost | Ensure scene has enough texture/geometry; reduce motion speed; **Check `--verbose` logs for `inliers` and `model_pts`** |
| Low FPS | Ensure Release build. The CPU (OpenMP) backend is the supported one — it is the fallback, not a workaround. |
| GLB doesn't import to Unity | Ensure Unity 2019.4+ which includes built-in GLTF support, or use GLTFast package |
| Linker / Undefined Reference errors | Run `./scripts/build.sh --clean` to refresh Qt meta-object data |
| `error: unrecognized --backend value '...'` | The value is not one of `auto, cpu, cuda, gpu, hip`. The process exits with status `2` rather than silently falling back to `auto`. |
| HIP build fails | HIP is **deferred**: its translation units have never been compiled on this host, so a HIP configure failing is the expected starting state, not a regression to chase from here. Start from `docs/CUDA_HIP_DEFERRED_CHANGES.md` (`cross-backend:C1`, `cross-backend:C3`) and build with `-DGPU_BACKEND=CPU` meanwhile. |
| GLB colors appear dark or incorrect | This should be fixed with proper color normalization. If issues persist, check that RGB integration is enabled and volume has sufficient weight |
| CUDA build requested | CUDA is **deferred**: untested, no NVIDIA hardware here, and possibly non-compiling. The CPU backend is the built and tested path. |

---

## Documentation

| Document | What it is authoritative for | Backend stance |
|---|---|---|
| [`docs/CANONICAL_SEMANTICS.md`](docs/CANONICAL_SEMANTICS.md) | The CPU rules each subsystem must keep (depth domain, winding, color, borders, pipeline lifecycle). | Canonical on CPU only; every divergence it names is a dossier row. |
| [`docs/CUDA_HIP_DEFERRED_CHANGES.md`](docs/CUDA_HIP_DEFERRED_CHANGES.md) | Every CUDA/HIP finding: audit ID, backend file/function, the CPU decision, risk, and the future acceptance test. | `deferred` / `not compiled` / `not runtime-tested` for all rows. |
| [`REGRESSION_REVIEW.md`](REGRESSION_REVIEW.md) | History of resolved regressions. | CPU items verified; GPU items deferred. |
| [`KNOWLEDGE_BASE.md`](KNOWLEDGE_BASE.md) | Module-by-module architecture reference. | Describes both lanes; only the CPU lane is tested. |
| [`HYPERPARAMETER_GUIDE.md`](HYPERPARAMETER_GUIDE.md) | Tunable parameters and their effects. | Backend-neutral. |
| [`rocm.md`](rocm.md) | Porting rules for HIP code. | Guidance for the deferred HIP lane. |

---

## License

GPLv3 — see LICENSE file.
