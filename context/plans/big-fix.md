# big-fix - Work Plan

## Plan revision

Date: 2026-09-22

Execution mode: **CPU-only**

This revision replaces the previous CPU/CUDA/HIP lockstep execution model. The CPU lane is the only build, test, and acceptance lane in this run. CUDA/HIP parity findings remain in scope as **documented deferred work**, but this run must not invoke `nvcc` or `hipcc`, configure a CUDA/HIP backend, compile a CUDA/HIP target, or run GPU tests.

## TL;DR (For humans)

This plan executes the audit remediation on the CPU path only. It keeps the CTest harness, headless `PipelineController` seam, geometry/tracking/export/pipeline/GUI cleanup, and CPU documentation work, while removing CUDA/HIP compilation from the critical path.

Every CUDA/HIP finding must be preserved in a strong, reviewable deferral dossier:

```text
docs/CUDA_HIP_DEFERRED_CHANGES.md
```

That dossier records what must change later for `nvcc` and `hipcc` backends, why it was deferred, which CPU behavior is now canonical, and what future compile/runtime gates must pass. The dossier is the audit memory; it is not an optional summary.

## Scope

### Objective

Complete the audit remediation for the CPU/CMake/Ninja/CTest/Qt/OpenMP path in `/home/iphands/prog/slop/kin/vendor/azu`, while preserving CUDA/HIP findings as explicit deferred work.

### In scope

- CPU-only CMake configure, build, and CTest execution.
- CTest-based headless tests for deterministic CPU code.
- CPU/OpenMP geometry, TSDF, tracking, sensor, export, pipeline, GUI, and utility fixes.
- Qt and OpenGL application work that compiles without CUDA or HIP.
- CPU-only build scripts and CPU-only QA evidence.
- A backend deferral dossier covering every CUDA/HIP audit finding.
- Documentation of backend parity risk and future compile/runtime acceptance.

### Out of scope

- No `nvcc` invocation.
- No `hipcc` invocation.
- No CMake configure with `GPU_BACKEND=CUDA` or `GPU_BACKEND=HIP`.
- No build or compile gate for CUDA or HIP targets.
- No CUDA or HIP runtime tests.
- No edits to `*.cu`, `*.hip`, or other CUDA/HIP-only translation units in this run.
- No claims that CUDA/HIP behavior was compiled, linked, runtime-tested, or parity-verified.
- No physical Kinect device requirement.
- No GUI display requirement for required tests.

### Approved decisions

- D1: CPU-only execution is the authoritative scope for this plan revision.
- D2: Missing `nvcc` and `hipcc` are expected environment states, not blockers.
- D3: CUDA/HIP parity findings are preserved through `docs/CUDA_HIP_DEFERRED_CHANGES.md`; they are not deleted or silently skipped.
- D4: CUDA/HIP files are not modified in this run. This prevents a partially updated backend from appearing canonical without compilation.
- D5: Future backend work requires a separate run that starts from the deferral dossier and explicitly permits CUDA/HIP compilation.

## Hard rules

- Required acceptance uses only:
  - `cmake`
  - `ninja`
  - `ctest`
  - `Qt6`
  - `eigen3`
- The required CPU lane is defined by those tools. `libfreenect`, `nvcc`, and `hipcc` are optional for this run.
- If `libfreenect` is missing, CPU tests must use the planned no-device path and must not open a device.
- If `nvcc` or `hipcc` is missing, record `deferred` and continue. Do not mark the CPU lane blocked because of missing backend compilers.
- Do not add a script or test that runs `nvcc`, `hipcc`, `build.sh --hip`, `build.sh --cuda`, `GPU_BACKEND=CUDA`, or `GPU_BACKEND=HIP`.
- Do not create `build-cuda-big-fix` or `build-hip-big-fix` in this run.
- Do not delete, clean, configure, or mutate existing `build-cuda/`.
- Do not edit CUDA/HIP-only translation units. If a CPU-only change would otherwise require touching them, stop at the CPU boundary and write the exact required backend change into the deferral dossier.
- Do not weaken CPU-only tests to hide missing backend coverage.
- Do not use a CPU-only test as proof that CUDA or HIP will compile.

## Canonical semantics

These are CPU-path canonical rules for this run. Backend parity is deferred until a separate backend plan executes the dossier.

| Topic | CPU canonical rule |
|---|---|
| Marching-cubes source of truth | CPU consumes one shared table header. CUDA/HIP duplicate tables are documented as deferred migration items. |
| Marching-cubes winding | The CPU signed-volume/outward-normal fixture is authoritative for this run. Backend winding must later derive from the same fixture. |
| Empty TSDF voxel | `tsdf = 1.0f`, `weight = 0.0f`, neutral gray color. |
| Mesh vertex validity | A cube may mesh if a crossing edge is supported by weighted voxels; unobserved voxels are sampled as `+1.0f`. |
| Mesh welding | CPU welding uses exact edge identity where practical; quantized float welding is not canonical. |
| Floor vs truncate | CPU coordinate and pixel projection uses floor semantics matching documented GPU behavior, without compiling GPU code. |
| Invalid raw depth | Raw `0` and raw `>= 2047` are invalid. Raw `2047` is not a fillable hole. |
| Depth-to-meters | Output must be finite and within configured bounds; invalid or out-of-range output becomes zero/invalid rather than a clamped wall. |
| ICP damping | CPU uses adaptive Tikhonov: default `0.1`, level 0 `0.01`, ill-conditioned Hessian escalates to `1.0`. |
| ICP guards | CPU rejects non-finite values and enforces translation cap `0.2 m` and rotation cap `0.5 rad`. |
| ICP residual | CPU reports the weighted Huber objective consistently. |
| Color storage | Voxel color is float sRGB `[0,1]`; MeshData emits uint8 sRGB; GLB `COLOR_0` is linear `[0,1]` float; PLY remains sRGB uint8. |
| Integration near-depth gate | CPU integration uses configured `min_depth`; no hard-coded `0.1f` may override it. |
| Numeric tolerance | CPU deterministic repeats use byte-identical comparison where specified. Backend numeric parity tolerances are documented for future runs. |
| Backend selection | This run configures only `GPU_BACKEND=CPU`. It does not claim backend selection parity. |

## Backend deferral dossier

The deferral dossier is `docs/CUDA_HIP_DEFERRED_CHANGES.md`. It must be created early and updated whenever a CPU todo discovers a backend parity requirement.

Required dossier structure:

```markdown
# CUDA/HIP deferred changes from big-fix CPU-only execution

Status: deferred; not compiled in this run

## Environment at deferral time
- nvcc: missing|version
- hipcc: missing|version
- CUDA runtime: present|unknown
- HIP runtime: present|unknown
- existing build-cuda/: present|absent

## Future backend gate
- DO NOT RUN IN THIS PLAN: cmake -S . -B build-cuda-big-fix -G Ninja -DGPU_BACKEND=CUDA -DBUILD_TESTING=OFF
- DO NOT RUN IN THIS PLAN: cmake --build build-cuda-big-fix
- DO NOT RUN IN THIS PLAN: cmake -S . -B build-hip-big-fix -G Ninja -DGPU_BACKEND=HIP -DBUILD_TESTING=OFF
- DO NOT RUN IN THIS PLAN: cmake --build build-hip-big-fix

## Deferred change table
| Audit ID | Backend | File/function | Current CPU decision | Required CUDA/HIP change | Risk if not done | Future acceptance |
```

The table must contain one row for every CUDA/HID finding, including GPU-only preprocessing, GPU lifecycle, GPU color storage, GPU static buffers, HIP raycast fall-through, backend duplicate tables, and backend winding/color/depth parity.

Each row must name the exact backend file and function or table symbol. If the exact line is unknown, the row must name the exact symbol and the audit ID.

## Verification strategy

### Environment gate

The existing phase-0 probe remains valid. It records toolchain state and emits:

```text
required_cpu_lane: pass|blocked
```

For this revision:

- `required_cpu_lane: pass` allows the plan to execute.
- Missing `libfreenect`, `nvcc`, or `hipcc` is recorded, not blocked.
- A future backend run may add a separate backend gate, but that gate is not part of this plan.

### CPU-only build

Required command shape:

```bash
cmake -S . -B build-cpu-big-fix -G Ninja -DGPU_BACKEND=CPU -DBUILD_TESTING=ON
cmake --build build-cpu-big-fix
ctest --test-dir build-cpu-big-fix -L cpu --output-on-failure
```

`build-cpu` may be used as the todo-2 fresh harness directory. The final gate uses fresh `build-cpu-big-fix`.

### Test labels

- `cpu`: required, deterministic, no display, no device, no CUDA, no HIP.
- `qt`: optional Qt/offscreen or widget tests, not required for this plan unless explicitly proven available.
- `cuda`: reserved for future backend runs only.
- `hip`: reserved for future backend runs only.

Required tests use `-L cpu` only.

### Backend evidence policy

Backend evidence in this run is documentary:

- `deferred`
- `not compiled`
- `not runtime-tested`
- `future gate required`

It is never reported as compile-pass or runtime-pass.

## Execution strategy

- Work waves are CPU-only.
- Each shared algorithm change must leave a deferral row for any backend counterpart.
- Tests must be written before behavior fixes when deterministic.
- Existing behavior must be pinned with characterization tests before product edits when behavior changes.
- Keep commits atomic and per todo.
- Do not combine backend deferral documentation with unrelated CPU cleanup.

## Todos

- [x] 1. Add a phase-0 environment probe and evidence artifact.
  - Current status: completed by commit `841a00e`.
  - Current CPU lane evidence: `required_cpu_lane: pass`.
  - Backend status: `nvcc` and `hipcc` are deferred optional tools, not blockers.

- [x] 2. Add a CPU-only CTest harness and headless `PipelineController` test seam without moving tests under `src/`.
  - References: `CMakeLists.txt`, `tests/`, `src/app/PipelineController.cpp`, `include/app/PipelineController.h`, `src/sensor/KinectSensor.cpp`, `include/sensor/KinectSensor.h`, `include/app/FusionHyperparams.h`, `include/meshing/MeshData.h`, `include/tsdf/TSDFVolume.h`.
  - Actions:
    - Create `tests/` outside `src/`.
    - Guard test targets with `GPU_BACKEND STREQUAL "CPU"` and `BUILD_TESTING`.
    - Add conditional `enable_testing()` and `include(CTest)` without requiring CUDA/HIP.
    - Add `azu_test_core` for Qt-free CPU numeric/export contracts.
    - Add `azu_test_pipeline` for the real `src/app/PipelineController.cpp` and its non-GPU dependencies.
    - Ensure test targets do not receive `CUDA_ENABLED` or `HIP_ENABLED` definitions.
    - Use `HAVE_FREENECT=0` or the existing CPU-safe no-device equivalent when libfreenect is absent.
    - Add `AZU_PIPELINE_TEST_SEAM` compile definition only to test targets.
    - Add `startWithoutSensorForTests()` and `injectRawFrameForTests(RawFrame)` under the seam.
    - Centralize `QMetaObject::invokeMethod(qApp, ...)` callback delivery.
    - Under the seam with null `qApp`, deliver through a test hook and record delivery count.
    - Add `cpu_smoke_harness` and `pipeline_test_seam_smoke`, both labeled `cpu`.
    - Add a mechanical configure-time source/link gate proving the seam uses the real controller translation unit.
  - Acceptance: fresh `build-cpu` configures, builds, and passes CTest with both named CPU tests.
  - Evidence: `.omo/evidence/big-fix/harness-cmake.txt`, `.omo/evidence/big-fix/harness-ctest-cpu.txt`, `.omo/evidence/big-fix/harness-baseline-missing-tests.txt`, `.omo/evidence/big-fix/pipeline-test-seam.txt`, `.omo/evidence/big-fix/pipeline-null-qapp-callback.txt`, `.omo/evidence/big-fix/pipeline-test-source-gate.txt`.
  - Commit: `test(big-fix): add CPU-only CTest harness and pipeline seam`

- [x] 3. Add a CPU-only QA runner and backend deferral scaffold.
  - References: `CMakeLists.txt`, `scripts/`, `.omo/evidence/big-fix/`, `docs/`.
  - Actions:
    - Add `scripts/test-cpu-big-fix.sh` or an equivalent script that creates/uses fresh `build-cpu-big-fix`.
    - The script configures and builds only `GPU_BACKEND=CPU`.
    - The script runs only `ctest --test-dir build-cpu-big-fix -L cpu --output-on-failure`.
    - The script must never invoke `nvcc`, `hipcc`, CUDA, or HIP.
    - Create `docs/CUDA_HIP_DEFERRED_CHANGES.md` with the required dossier structure and current probe values.
    - The script emits a backend deferral matrix with `not compiled` and `not runtime-tested` values.
  - Acceptance: CPU script runs required CPU tests; backend matrix contains deferred, not compiled, and not runtime-tested for CUDA and HIP.
  - Evidence: `.omo/evidence/big-fix/cpu-gate.txt`, `.omo/evidence/big-fix/backend-deferral-scaffold.txt`.
  - Commit: `build(big-fix): add CPU-only QA gate and backend deferral scaffold`

- [x] 4. Consolidate the CPU Marching Cubes table into one shared source of truth without editing backend translation units.
  - References: `src/meshing/MarchingCubes.cpp`, `src/meshing/MarchingCubes_cuda.cu`, `src/meshing/MarchingCubes_hip.hip`, meshing D1, cross-backend A8.
  - Actions:
    - Create `include/meshing/MarchingCubesTables.h` or equivalent shared header.
    - Define `edge_table` and `tri_table` once for CPU use.
    - Make the CPU Marching Cubes translation unit consume the shared header.
    - Leave `src/meshing/MarchingCubes_cuda.cu` and `src/meshing/MarchingCubes_hip.hip` untouched.
    - Record each remaining CUDA/HIP table duplicate in the deferral dossier with exact file and symbol.
  - Acceptance: CPU build passes; grep shows CPU uses the shared header; grep also documents backend duplicate files as deferred.
  - Evidence: `.omo/evidence/big-fix/table-consolidation-grep.txt`, `.omo/evidence/big-fix/backend-table-deferral.txt`.
  - Commit: `refactor(meshing): centralize CPU marching cubes tables`

- [x] 5. Record canonical CPU semantics and initialize the backend deferral dossier.
  - References: audit draft `.omo/drafts/big-fix.md`, `README.md`, `REGRESSION_REVIEW.md`, `include/app/FusionHyperparams.h`, `include/tsdf/VoxelGPU.h`.
  - Actions:
    - Create or update `docs/CANONICAL_SEMANTICS.md`.
    - Document CPU canonical decisions for geometry, depth, color, export, ICP, TSDF, and winding.
    - Expand `docs/CUDA_HIP_DEFERRED_CHANGES.md` with all discovered backend findings from the audit and this CPU revision.
    - State explicitly that GLB `COLOR_0` is linear float.
    - State explicitly that raw `0` and raw `>= 2047` are invalid.
    - State explicitly that the outward-normal fixture is the CPU authoritative winding rule.
  - Acceptance: docs contain required canonical strings and backend deferral rows.
  - Evidence: `.omo/evidence/big-fix/canonical-semantics-grep.txt`, `.omo/evidence/big-fix/backend-deferral-grep.txt`.
  - Commit: `docs(big-fix): record CPU canonical semantics and backend deferrals`

- [x] 6. Add analytic Marching Cubes table and CPU sphere geometry tests before fixing the table.
  - References: CPU Marching Cubes source, shared table header, meshing A1/A2, cross-backend A7.
  - Actions:
    - Validate all 256 edge-table rows against a crossing-edge construction oracle.
    - Add a small analytic sphere SDF meshing test.
    - Assert closed surface edge counts, no degenerate triangles, outward normals, and expected triangle-count band.
    - Do not use parity alone as the oracle.
  - Acceptance: tests compile and are registered; at todo 6 they are expected to fail against the corrupt CPU/shared table.
  - Fail-before-fix evidence is required and must show CTest executed the intended red assertions.
  - Evidence: `.omo/evidence/big-fix/marching-cubes-test-fail-before-fix.txt`.
  - Commit: `test(meshing): add marching cubes table and sphere contracts`

- [x] 7. Fix Marching Cubes edge-table corruption for CPU/shared table indices 213/214/215.
  - References: shared Marching Cubes table header, CPU Marching Cubes source, meshing A1/A2.
  - Actions:
    - Set the shared table entries to `edge_table[213] = 0x83f`, `edge_table[214] = 0xb35`, and `edge_table[215] = 0xa3c`.
    - Remove misleading CPU fix comments.
    - Leave CUDA/HIP table corruption untouched and record exact future corrections in the deferral dossier.
  - Acceptance: todo 6 CPU tests pass for the shared table; grep shows no old CPU values in the shared header.
  - Evidence: `.omo/evidence/big-fix/edge-table-fixed-grep.txt`, `.omo/evidence/big-fix/edge-table-test-pass.txt`.
  - Commit: `fix(meshing): repair CPU marching cubes edge table`

- [x] 8. Fix CPU TSDF reset sentinel and stale host state.
  - References: `src/tsdf/TSDFVolume.cpp`, `include/tsdf/TSDFVolume.h`, TSDF T2/T5/T11/T13, meshing C10.
  - Actions:
    - Introduce `EMPTY_TSDF = 1.0f` and `EMPTY_WEIGHT = 0.0f`.
    - Reset host volume to `Voxel{1.0f, 0.0f, neutral color}`.
    - Make CPU parameter changes that alter voxel size/origin clear the host volume.
    - Reject or re-derive stale truncation/max-weight/origin state.
    - Record GPU reset, sync, and device mirror requirements in the deferral dossier.
  - Acceptance: CPU reset test sees empty voxels with `tsdf == 1.0f` and no stale CPU mirror.
  - Evidence: `.omo/evidence/big-fix/tsdf-reset-test.txt`, `.omo/evidence/big-fix/tsdf-reset-grep.txt`.
  - Commit: `fix(tsdf): use CPU empty TSDF sentinel and clear stale state`

- [x] 9. Fix pipeline stop lost-wakeup deadlock.
  - References: `src/app/PipelineController.cpp`, `include/app/PipelineController.h`, worker predicates, pipeline PC-01.
  - Actions:
    - In `stop()`, acquire each queue mutex before notifying its condition variable.
    - Ensure predicate state cannot change between worker predicate check and wait registration.
    - Add a bounded start/stop stress test using the todo 2 seam.
    - Add a source/link gate proving the test calls the real controller and links `azu_test_pipeline`.
  - Acceptance: `ctest --test-dir build-cpu -L cpu -R "pipeline_stop" --output-on-failure` passes three consecutive runs or one 50-cycle loop under timeout.
  - Evidence: `.omo/evidence/big-fix/pipeline-stop-loop.txt`, `.omo/evidence/big-fix/pipeline-stop-grep.txt`, `.omo/evidence/big-fix/pipeline-stop-source-gate.txt`.
  - Commit: `fix(pipeline): synchronize CPU stop notifications with worker queues`

- [x] 10. Add `MeshData::validate()` and call it from PLY/GLB exports.
  - References: `include/meshing/MeshData.h`, `src/export/PLYExporter.cpp`, `src/export/GLBExporter.cpp`, export GLB-01/02/07, PLY-05, MESH-01/02.
  - Actions:
    - Implement `MeshData::validate(std::string* reason)`.
    - Check non-empty positions, index count multiple of 3, in-range indices, finite positions/normals, and all-or-none normals/colors.
    - Call validation from PLY and GLB entry points.
    - Return false and log a reason when validation fails.
  - Acceptance: CPU CTest export validation tests pass and reject empty indices, non-multiple-of-3 indices, out-of-range index, NaN position, and partial colors.
  - Evidence: `.omo/evidence/big-fix/mesh-validation-tests.txt`.
  - Commit: `fix(export): validate CPU mesh data before writing files`

- [x] 11. Fix CPU coordinate rounding and boundary semantics.
  - References: CPU tracking and TSDF coordinate conversion code, tracking CPU-1, CPU-2, TSDF T6, cross-backend A18/A19.
  - Actions:
    - Replace CPU `static_cast<int>(value + 0.5f)` with floor-based rounding.
    - Replace CPU `worldToVoxel` truncation with floor semantics.
    - Reject non-finite values before casting to integer coordinates.
    - Record CUDA/HIP parity requirements in the deferral dossier.
  - Acceptance: CTest coordinate tests pass for negative, fractional, boundary, NaN, and Inf cases.
  - Evidence: `.omo/evidence/big-fix/coordinate-rounding-tests.txt`, `.omo/evidence/big-fix/coordinate-rounding-grep.txt`.
  - Commit: `fix(tracking,tsdf): align CPU coordinate floor semantics`

- [x] 12. Make CPU ICP Huber weighting and finite-value rejection consistent.
  - References: CPU ICP tracker and frame data code, tracking ALL-1/ALL-3/ALL-4/CPU-9.
  - Actions:
    - Apply Huber weight consistently to curvature and gradient accumulation.
    - Reject non-finite residuals and Jacobian entries.
    - Move shared constants into one header.
    - Add a depth-jump guard to CPU pyramid downsampling.
    - Record GPU Huber and finite-guard parity requirements in the deferral dossier.
  - Acceptance: deterministic CPU ICP test on synthetic inliers/outliers passes and reports the documented weighted objective.
  - Evidence: `.omo/evidence/big-fix/icp-weighting-tests.txt`.
  - Commit: `fix(tracking): apply CPU Huber weighting and finite guards`

- [x] 13. Align CPU ICP damping, validity gates, rotation guards, residual reporting, and counters.
  - References: CPU ICP tracker and shared header, tracking CPU-4/CPU-5/CPU-10/CPU-11/HDR-1, cross-backend A6/A9/A10/A16/A17/A30.
  - Actions:
    - Implement canonical CPU damping policy.
    - Add CPU rotation cap `0.5 rad`.
    - Use squared-norm normal validity checks.
    - Document and use weighted residual consistently.
    - Treat acceptable final-step/inlier quality as success where the audit identified false TrackingLost.
    - Initialize `ICPResult::pose` to identity and validate/clamp `angle_threshold`.
    - Use determinant-corrected Procrustes or SVD orthonormalization.
    - Record GPU counter and kernel parity requirements in the deferral dossier.
  - Acceptance: CPU unit fixture passes within documented tolerances and all normals are unit or rejected.
  - Evidence: `.omo/evidence/big-fix/icp-parity-cpu.txt`, `.omo/evidence/big-fix/icp-kernel-deferral.txt`.
  - Commit: `fix(tracking): unify CPU ICP numeric behavior`

- [x] 14. Fix CPU TSDF integration OpenMP race deterministically.
  - References: `src/tsdf/TSDFVolume.cpp`, TSDF T3.
  - Actions:
    - Replace concurrent read-modify-write with a deterministic two-phase merge.
    - Preserve image-space update semantics and color/weight update formulas.
    - Add a repeated-run determinism test.
  - Acceptance: CPU integration fixture produces byte-identical volume/mesh bytes across three repeated runs for the same input.
  - Evidence: `.omo/evidence/big-fix/tsdf-cpu-race-hashes.txt`, `.omo/evidence/big-fix/tsdf-cpu-race-tsan.txt` or TSan skip.
  - Commit: `fix(tsdf): serialize CPU integration updates deterministically`

- [x] 15. Fix CPU TSDF raycast interpolation, NaN guards, color ghosting, configured depth bounds, and sampling guards.
  - References: CPU TSDF source, TSDF T7/T8/T9/T16/T19/T26/T27, cross-backend A11/A12/A27.
  - Actions:
    - Use actual previous step size in CPU zero-crossing interpolation.
    - Reject NaN/Inf `t_hit` and `hit_world`.
    - Pass configured min/max depth into CPU raycast bounds.
    - Replace hard-coded CPU integration near-plane floor with configured `min_depth`.
    - Implement continuous ray marching with trilinear SDF interpolation and steps no larger than `0.5 * voxel_size`.
    - Detect both positive-to-negative and negative-to-positive TSDF crossings.
    - Record GPU raycast and integration parity in the deferral dossier.
  - Acceptance: CPU raycast fixture hits a known synthetic surface within `1e-5 m`; `min_depth` fixture changes gate behavior; thin-sub-voxel fixture fills after the fix.
  - Evidence: `.omo/evidence/big-fix/tsdf-raycast-tests.txt`, `.omo/evidence/big-fix/tsdf-integration-min-depth-tests.txt`, `.omo/evidence/big-fix/tsdf-subvoxel-thin-feature-tests.txt`.
  - Commit: `fix(tsdf): correct CPU raycast, configured depth gates, and sampling guards`

- [x] 16. Fix CPU Marching Cubes unobserved voxel handling, border normals, and NaN normal guards.
  - References: CPU Marching Cubes source, meshing B1/B2/C2/C3/B3/B4/C7/D8, cross-backend A14.
  - Actions:
    - Sample SDF with weight guard: unobserved means `+1.0f`.
    - Require crossing-edge support rather than dropping cubes due to unrelated unobserved corners.
    - Guard normal normalization and interpolation.
    - Skip triangles with non-finite interpolated positions.
    - Use one-sided border differences where possible.
    - Cache or share corner normals.
  - Acceptance: CPU sphere/cube fixture produces closed surface, finite normals/vertices, and no observed/unobserved boundary cracks.
  - Evidence: `.omo/evidence/big-fix/meshing-unobserved-tests.txt`.
  - Commit: `fix(meshing): CPU mesh observed frontier without holes or NaN normals`

- [x] 17. Replace CPU quantized welding with exact edge-key welding and enforce mesh invariants.
  - References: CPU Marching Cubes source, `include/meshing/MeshData.h`, meshing B2/B6/C1/C4/C9/C10/D5/D7/D9/D10, cross-backend A29.
  - Actions:
    - Weld by exact edge identity `(voxel, edge_index)` or equivalent.
    - Keep a `truncated` flag when triangle cap is reached.
    - Push or clear colors in lockstep with positions.
    - Expose `has_colors` explicitly.
    - Share interpolation parameter across vertex, normal, and color interpolation.
    - Guard tiny resolutions and progress division by zero.
    - Document progress callback thread-safety.
  - Acceptance: CPU analytic mesh tests assert no duplicate welded vertices for shared edges and no false merges; color buffer is empty or exactly `positions.size() * 3`.
  - Evidence: `.omo/evidence/big-fix/mesh-welding-tests.txt`, `.omo/evidence/big-fix/mesh-color-invariant-tests.txt`, `.omo/evidence/big-fix/mesh-truncation-progress-tests.txt`.
  - Commit: `fix(meshing): CPU weld by edge identity and protect mesh invariants`

- [x] 18. Derive CPU winding convention and defer backend winding normalization.
  - References: CPU Marching Cubes source, meshing B5, cross-backend A1.
  - Actions:
    - Add a signed-volume/outward-normal CPU test derived from the reference sphere construction.
    - Treat the derived outward-normal rule as authoritative.
    - If the derived rule requires reverse table order, change CPU only.
    - If the derived rule requires forward table order, change CPU only.
    - Document the derived convention in canonical semantics.
    - Record CUDA/HIP winding parity requirements in the deferral dossier.
  - Acceptance: CPU winding test passes without a hardcoded expected table order; deferral dossier lists backend winding work.
  - Evidence: `.omo/evidence/big-fix/winding-tests.txt`, `.omo/evidence/big-fix/winding-derived-sign.txt`.
  - Commit: `fix(meshing): derive CPU triangle winding`

- [x] 19. Fix CPU color accumulation, mesh quantization, and GLB linear color export.
  - References: CPU TSDF color code, MeshData, `src/export/GLBExporter.cpp`, TSDF T10/T15, GLB-04, cross-backend A13/A26/B3.
  - Actions:
    - Accumulate CPU voxel color as float sRGB `[0,1]`.
    - Apply one CPU clamp policy at read/extraction.
    - Emit MeshData color as uint8 sRGB.
    - Apply sRGB-to-linear decode in GLB export.
    - Reject NaN/Inf color components before export.
    - Record `VoxelGPU` storage and GPU clamp parity in the deferral dossier.
  - Acceptance: CPU color convergence test passes; GLB golden test asserts decoded linear values within `1e-5`.
  - Evidence: `.omo/evidence/big-fix/color-convergence-tests.txt`, `.omo/evidence/big-fix/glb-linear-color.txt`.
  - Commit: `fix(color): canonicalize CPU sRGB storage and linear GLB export`

- [x] 20. Fix CPU raw depth validity, depth-to-meters rejection, and hole-fill sentinel behavior.
  - References: CPU sensor/preprocessing code, sensor S-01/S-02/S-03/S-04/S-10/S-12/S-20, cross-backend A5.
  - Actions:
    - Use invalid predicate `raw == 0 || raw >= 2047`.
    - Reject valid raw values whose meter conversion is outside configured bounds.
    - Zero invalid output instead of clamping to a wall.
    - Fill only raw `0` holes.
    - Split CPU depth EMA source/destination buffers.
    - Add explicit OpenMP sharing clauses.
    - Record GPU depth-domain and hole-fill parity in the deferral dossier.
  - Acceptance: depth-domain CPU tests pass for raw `0`, raw `2047`, near-pole, negative, below-minimum, and above-maximum conversions; EMA repeat is deterministic.
  - Evidence: `.omo/evidence/big-fix/depth-domain-tests.txt`, `.omo/evidence/big-fix/depth-ema-determinism.txt`.
  - Commit: `fix(sensor): validate CPU depth and reject out-of-range meters`

- [x] 21. Unify CPU CAS border behavior and super-resolution guidance luma.
  - References: CPU signal-conditioner and super-resolution sources, sensor S-05, cross-backend A20.
  - Actions:
    - Use one CPU reflection border-mode helper or constant.
    - Add a 1-pixel border test where reflection and clamp diverge.
    - Add CPU guidance-luma tests.
    - Record CUDA/HIP border-mode parity in the deferral dossier.
  - Acceptance: CPU border tests pass and CPU guidance luma is deterministic for synthetic edge patterns.
  - Evidence: `.omo/evidence/big-fix/cas-border-tests.txt`.
  - Commit: `fix(sensor): unify CPU CAS reflection border mode`

- [x] 22. Make super-resolution upscaled RGB behavior explicitly CPU-only and defer GPU upscaled expectations.
  - References: CPU super-resolution and pipeline consumer code, sensor S-06, cross-backend A15/A24/B5/C8, pipeline C8.
  - Actions:
    - Add `sr_upscaled_available()` or equivalent availability contract.
    - CPU returns `true` only after producing a fresh upscaled buffer for the current frame.
    - GPU/HIP availability must be reported as unavailable by the CPU-only path, not by compiling GPU code.
    - Keep the commented GPU product consumer disabled and documented as future scope.
    - Record GPU EASU/CAS upscaled defects and stale-buffer hazards in the deferral dossier.
  - Acceptance: no CPU product path can read a stale or zero upscaled buffer as valid.
  - Evidence: `.omo/evidence/big-fix/sr-upscaled-path.txt`, `.omo/evidence/big-fix/sr-sharpness-nan-tests.txt`.
  - Commit: `fix(sensor): make super-resolution upscale CPU-only and fail closed`

- [x] 23. Fix Kinect RGB/depth pairing staleness and queue latest-frame semantics without requiring a device.
  - References: `src/sensor/KinectSensor.cpp`, `src/sensor/FrameData.cpp`, sensor S-07/S-08/S-09/S-15.
  - Actions:
    - Require `abs(timestamp_depth - timestamp_rgb)` below a documented threshold, recommended 50 ms.
    - Make `getLatestFrame()` return the newest queued frame and drop older queued frames.
    - Copy the pending frame pointer and invoke callbacks outside `sync_mutex_`.
    - Add deterministic timestamp fixture tests.
    - Record depth-to-color registration status as `SKIP: no depth registration source` when libfreenect/registration is unavailable.
  - Acceptance: stale RGB pair is rejected; latest frame is returned; callback path is outside sensor sync lock.
  - Evidence: `.omo/evidence/big-fix/kinect-pairing-tests.txt`, `.omo/evidence/big-fix/kinect-depth-color-alignment.txt`.
  - Commit: `fix(sensor): reject stale RGB/depth pairs and return newest frame`

- [x] 24. Fix hyperparameter propagation and callback races in `PipelineController`.
  - References: `src/app/PipelineController.cpp`, `src/gui/ControlPanel.cpp`, pipeline PC-03/PC-08/PC-09/PC-11/PC-12/PC-13, cross-backend A28, GUI GL-35.
  - Actions:
    - Re-read depth min/max and relevant hyperparams per iteration under `hyper_mutex_`.
    - Guard tracker params and preprocessor setters with the existing mutex pattern.
    - Serialize TSDF resize against active meshing/tracking/integration state.
    - Serialize callbacks or make them set-before-start-only with asserts.
    - Apply thread-count changes only at a frame boundary or while stopped.
  - Acceptance: CPU seam fixture starts the real controller, injects frames, changes hyperparams, and observes updated values.
  - Evidence: `.omo/evidence/big-fix/pipeline-hyperparams-tests.txt`.
  - Commit: `fix(pipeline): propagate CPU hyperparameters and guard callbacks`

- [x] 25. Fix CPU reset/start state, mesh request race, metrics, and dropped-frame observability.
  - References: `src/app/PipelineController.cpp`, pipeline PC-04/PC-05/PC-06/PC-07/PC-15/PC-19/PC-21/PC-22/PC-24.
  - Actions:
    - Reset `last_pose_` on start or require reset in the start transition.
    - Replace mesh request flags with a versioned request counter.
    - Use drop-oldest or retain-latest queue behavior.
    - Lock metrics writes and add dropped-frame counters.
    - Make mesh cadence time-based or on-demand without stalling capture.
    - Keep the most recent startup frame.
  - Acceptance: CPU seam fixture proves correct initial pose, no stale first prediction, versioned mesh snapshots, and dropped-count updates under synthetic backpressure.
  - Evidence: `.omo/evidence/big-fix/pipeline-state-tests.txt`, `.omo/evidence/big-fix/pipeline-mesh-cadence-tests.txt`.
  - Commit: `fix(pipeline): harden CPU reset, mesh requests, and metrics`

- [x] 26. Move GUI export/reset work off the GUI thread and expose progress truthfully.
  - References: `src/gui/MainWindow.cpp`, `src/app/PipelineController.cpp`, pipeline PC-10/PC-18/PC-25, GUI GL-16/QT-01.
  - Actions:
    - Run PLY/GLB extraction and file writing on a worker thread or Qt concurrent worker.
    - Disable export buttons while running.
    - Deliver success/failure through a queued callback.
    - Make reset non-blocking for the UI.
    - Extract a shared `exportMesh(path, writer_fn)` helper.
    - Make queued callbacks safe against controller lifetime.
  - Acceptance: optional Qt/offscreen proof or compile-only plus manual QA; shared export helper compiles and is used by both export paths.
  - Evidence: `.omo/evidence/big-fix/export-thread-manual.txt` or `.omo/evidence/big-fix/export-thread-skip.txt`.
  - Commit: `fix(gui): run export and reset without blocking the GUI thread`

- [x] 27. Fix camera orbit/pan/free-flight math, timer clamp, and gizmo cursor state.
  - References: `src/rendering/Camera.cpp`, `src/gui/OpenGLWidget.cpp`, `include/gui/OpenGLWidget.h`, `src/gui/NavigationGizmo.cpp`, `src/gui/ControlPanel.cpp`, GUI GL-01/GL-05/GL-17/GL-19/GL-20.
  - Actions:
    - Add one `orbitEye()` helper.
    - Use consistent negative Y orbit offset.
    - Include roll in pan basis through a shared camera basis helper.
    - Change camera rotation feedback to float degrees end-to-end.
    - Clamp first-frame `dt`.
    - Restore open-hand cursor on gizmo release.
  - Acceptance: headless camera matrix tests pass; compile confirms float feedback signal.
  - Evidence: `.omo/evidence/big-fix/camera-tests.txt`.
  - Commit: `fix(rendering): correct CPU camera orbit/pan and timer state`

- [x] 28. Fix renderer shader initialization, resource cleanup, point cloud upload guards, and light space.
  - References: `src/rendering/PreviewRenderer.cpp`, `src/rendering/ShaderProgram.cpp`, GUI GL-10/GL-13/GL-14/GL-22/GL-24/GL-25/GL-32/GL-33/GL-34.
  - Actions:
    - Check shader `load()` return values.
    - Delete old shader program before overwriting handle.
    - Early-return uniform setters when `program_id == 0`.
    - Validate point cloud sizes before loops.
    - Clear ghost mesh state on empty upload.
    - Transform world-space light to view space each frame.
    - Zero GL handles after delete.
    - Use scope guards or snapshot/restore for GL state.
    - Avoid recomputing inverse-transpose normal matrix every frame.
  - Acceptance: CPU build passes; static/grep checks prove guards and state restoration.
  - Evidence: `.omo/evidence/big-fix/renderer-static-tests.txt` or offscreen skip.
  - Commit: `fix(rendering): harden CPU shader and upload lifecycle`

- [x] 29. Add cross-field UI validation, seed defaults, and fix GUI hygiene/perf/focus issues.
  - References: `src/gui/ControlPanel.cpp`, `include/gui/ControlPanel.h`, `src/gui/MainWindow.cpp`, `src/gui/MetricsPanel.cpp`, `include/app/FusionHyperparams.h`, GUI GL-15/GL-18/GL-21/GL-26/GL-27/GL-28/GL-30/GL-31/GL-36.
  - Actions:
    - Validate `min_depth < max_depth`.
    - Validate truncation/voxel ratio.
    - Cap depth max to device usable range, recommended 4.0 m, unless future device support is added.
    - Seed UI defaults from `FusionHyperparams::defaults()`.
    - Stage preset application while capture is running.
    - Cache metrics-panel style bands and re-polish only on transitions.
    - Set stylesheets on the main window instead of `qApp`.
    - Handle key events correctly.
    - Treat `icp_valid_model == -1` as `--` before warming bands.
    - Reuse one `hyperparamsSnapshot()` per apply handler.
  - Acceptance: headless UI model validation tests pass where Qt allows; otherwise compile-only plus manual QA.
  - Evidence: `.omo/evidence/big-fix/ui-validation-tests.txt`, `.omo/evidence/big-fix/ui-style-cache-tests.txt`.
  - Commit: `fix(gui): validate fusion hyperparameters and harden CPU UI hygiene`

- [x] 30. Fix PLY writer schema, flush/close correctness, partial files, and performance.
  - References: `src/export/PLYExporter.cpp`, export PLY-01/02/03/04/06/07/08.
  - Actions:
    - Explicitly close and check failure.
    - Remove partial files on failure.
    - Make binary and ASCII schemas consistent or delete ASCII if unused.
    - Warn about discarded trailing indices.
    - Warn about color size mismatches.
    - Write contiguous records.
    - Handle little-endian explicitly or error on big-endian.
  - Acceptance: CPU PLY parse tests pass, including failure paths.
  - Evidence: `.omo/evidence/big-fix/ply-export-tests.txt`.
  - Commit: `fix(export): validate and robustify CPU PLY writer`

- [x] 31. Fix GLB writer glTF validity, linear color, normals, and buffer move.
  - References: `src/export/GLBExporter.cpp`, `include/export/GLBExporter.h`, export GLB-01/03/04/05/06/08/09/10, MESH-02.
  - Actions:
    - Validate mesh before writing.
    - Reject zero-index and non-multiple-of-3 index counts.
    - Use unit normals or fallback normal.
    - Emit `COLOR_0` as linear float.
    - Fix coordinate comment to right-handed Y-up glTF.
    - Move buffer data instead of copying.
    - Report real writer failure reason.
  - Acceptance: GLB JSON chunk parse and tinygltf round-trip CPU tests pass.
  - Evidence: `.omo/evidence/big-fix/glb-export-tests.txt`.
  - Commit: `fix(export): make CPU GLB output valid and color-linear`

- [x] 32. Remove CPU dead code and defer named backend dead code hygiene.
  - References: CPU meshing, tracking, sensor, GUI, pipeline, utils, backend hygiene files; meshing D3-D6/D11/D12, tracking CPU-6/CPU-7/CPU-8, pipeline PC-16/PC-20/PC-26/JS/RB, GUI GL-23/GL-29, sensor S-17/S-18, cross-backend A8/C4/C5/C6/A34/A35.
  - Actions:
    - Remove duplicate/unused CPU vector hashers, dead mesh paths, unused interpolation/hash helpers, dead `projectModel`, unused intrinsic helpers, unused callbacks, dead renderer helpers, unused QSlider includes, dead reset wrappers, and always-true conditions.
    - Delete or fix unused CPU JobSystem and RingBuffer code.
    - Do not edit CUDA/HIP-only dead code; record named dead backend symbols in the deferral dossier.
  - Acceptance: CPU build passes and CPU dead-symbol grep gate returns expected absent symbols.
  - Evidence: `.omo/evidence/big-fix/dead-code-grep.txt`, `.omo/evidence/big-fix/backend-dead-code-deferral.txt`.
  - Commit: `refactor(core): remove CPU dead audit-flagged code`

- [x] 33. Fix CPU utils/logging/timer/job-system and global stats hygiene.
  - References: `src/utils/Logger.cpp`, `src/utils/Timer.cpp`, `include/app/JobSystem.h`, `src/tsdf/TSDFVolume.cpp`, `src/sensor/FrameData.cpp`, `src/sensor/SignalConditioner_omp.cpp`, `src/app/PipelineController.cpp`, pipeline LG/TM/JS/PC-17, sensor S-16/S-19, TSDF T23.
  - Actions:
    - Make logger level atomic or otherwise race-free.
    - Use `localtime_r`.
    - Avoid hot-path string allocation in timer where feasible.
    - If keeping JobSystem, replace `std::result_of` with `std::invoke_result_t`.
    - Document JobSystem submit-after-shutdown and drain-on-stop behavior.
    - Move per-instance stats into owning objects or explicitly document process-global stats.
    - Fix average stats to divide by contributing samples.
    - Replace raw pipeline `std::cout`/`std::cerr` logging with the project logger.
  - Acceptance: CPU utils tests or compile-only pass; stats fixtures divide by contributing counts.
  - Evidence: `.omo/evidence/big-fix/utils-tests.txt`, `.omo/evidence/big-fix/stats-fixtures.txt`.
  - Commit: `fix(utils): harden CPU logger and timer hygiene`

- [x] 34. Finalize the CUDA/HIP deferral dossier as a future backend implementation input.
  - References: all completed todos, `.omo/drafts/big-fix.md`, `docs/CUDA_HIP_DEFERRED_CHANGES.md`, existing `build-cuda/` status.
  - Actions:
    - Verify every CUDA/HIP audit ID has a dossier row.
    - Verify every row names backend file/function or table symbol.
    - Verify every row states `not compiled in this run`.
    - Verify every row states a future acceptance command or test.
    - Add exact rows for GPU preprocessing fallback, GPU lifecycle, GPU static buffers, HIP CPU raycast fall-through, backend duplicate tables, GPU winding, GPU color storage, GPU depth validity, GPU hole fill, GPU CAS border, GPU SR upscaled RGB, and GPU singletons.
    - Mark unresolved risk explicitly where CPU-only behavior cannot prove backend correctness.
  - Acceptance: dossier is complete enough that a later backend run can start without re-auditing the audit.
  - Evidence: `.omo/evidence/big-fix/backend-deferral-matrix.txt`, `.omo/evidence/big-fix/backend-deferral-grep.txt`.
  - Commit: `docs(big-fix): finalize CUDA and HIP backend deferral dossier`

- [x] 35. Reconcile documentation with CPU-only behavior and deferred backend status.
  - References: `README.md`, `REGRESSION_REVIEW.md`, `HYPERPARAMETER_GUIDE.md`, `KNOWLEDGE_BASE.md` if present, `CMakeLists.txt`, `src/main.cpp`, CPU signal-conditioner and super-resolution sources, cross-backend C1/C3/C7, sensor S-14/S-17/S-18, pipeline MN-01.
  - Actions:
    - Document current fork reality: Qt6, CPU-only QA path, CUDA/HIP deferred.
    - Correct README claims about active/deprecated backends without claiming backend compilation.
    - Mark audit claims as verified, CPU-fixed, or deferred.
    - Document CPU-only CTest commands.
    - Make `main.cpp` warn or exit on unrecognized `--backend` values.
    - Ensure docs never claim `build-cuda-big-fix` or `build-hip-big-fix` passed under this plan.
  - Acceptance: docs contain CPU CTest command and backend status section; unknown backend CLI fixture emits warning/error.
  - Evidence: `.omo/evidence/big-fix/docs-grep.txt`, `.omo/evidence/big-fix/backend-cli-warning.txt`.
  - Commit: `docs(big-fix): reconcile CPU-only and deferred backend documentation`

- [ ] 36. Run the final CPU-only gate and produce the delivery matrix.
  - References: all prior todos, `.omo/evidence/big-fix/`, `docs/CUDA_HIP_DEFERRED_CHANGES.md`.
  - Actions:
    - Configure fresh `build-cpu-big-fix` with `GPU_BACKEND=CPU` and `BUILD_TESTING=ON`.
    - Build `build-cpu-big-fix`.
    - Run `ctest --test-dir build-cpu-big-fix -L cpu --output-on-failure`.
    - Write `.omo/evidence/big-fix/final-matrix.txt`.
    - Matrix columns must include: required CPU tests, CUDA status, HIP status.
    - CUDA/HIP status values must be `deferred`, `not compiled`, and `not runtime-tested` for this plan.
    - Mirror this plan file to `context/plans/big-fix.md`.
  - Acceptance: required CPU tests pass; final matrix does not claim CUDA/HIP compile or runtime pass; plan copy exists.
  - Evidence: `.omo/evidence/big-fix/final-matrix.txt`, `context/plans/big-fix.md`.
  - Commit: `chore(big-fix): record CPU final QA matrix and plan copy`

## Traceability

This matrix is the compliance join key for F1. Every lane ID from `.omo/drafts/big-fix.md` maps to a CPU todo, a verified-safe no-fix decision, or a row in `docs/CUDA_HIP_DEFERRED_CHANGES.md`.

Canonical lane prefixes are: `tracking:`, `tsdf:`, `meshing:`, `sensor:`, `pipeline:`, `gui:`, `export:`, and `cross-backend:`.

| Lane IDs | Disposition |
|---|---|
| `tracking:CPU-1`, `cross-backend:A18`, `cross-backend:A19` | Todo 11 CPU floor; backend parity deferred dossier |
| `tracking:CPU-2`, `tracking:CPU-3`, `tracking:ALL-1`, `tracking:ALL-3`, `tracking:ALL-4`, `tracking:CPU-9` | Todo 12 CPU ICP weighting; backend parity deferred dossier |
| `tracking:CPU-4`, `tracking:CPU-5`, `tracking:ALL-2`, `tracking:CPU-10`, `tracking:CPU-11`, `tracking:GPU-3`, `tracking:GPU-4`, `tracking:GPU-6`, `tracking:GPU-7`, `tracking:GPU-8`, `tracking:HDR-1`, `cross-backend:A6`, `cross-backend:A9`, `cross-backend:A10`, `cross-backend:A16`, `cross-backend:A17`, `cross-backend:A30` | Todo 13 CPU ICP policy; GPU IDs deferred dossier |
| `tracking:GPU-1`, `tracking:GPU-2` | Deferred dossier |
| `tracking:HDR-2` | Todo 24 |
| `tracking:CPU-6`, `tracking:CPU-7`, `tracking:CPU-8`, `tracking:GPU-5` | Todo 32 CPU cleanup; `tracking:GPU-5` deferred dossier |
| `tsdf:T1`, `tsdf:T12`, `tsdf:T14`, `tsdf:T17`, `tsdf:T20`, `tsdf:T21`, `tsdf:T22`, `cross-backend:A33` | Deferred dossier; CPU lifecycle guards only where reachable without backend compile |
| `tsdf:T2`, `tsdf:T5`, `tsdf:T11`, `tsdf:T13` | Todo 8 |
| `tsdf:T3` | Todo 14 |
| `tsdf:T6` | Todo 11 |
| `tsdf:T7`, `tsdf:T8`, `tsdf:T9`, `tsdf:T16`, `tsdf:T19`, `tsdf:T26`, `tsdf:T27`, `cross-backend:A11`, `cross-backend:A12`, `cross-backend:A27` | Todo 15 CPU raycast; GPU parity deferred dossier |
| `tsdf:T10`, `tsdf:T15`, `cross-backend:B3`, `meshing:C8`, `cross-backend:A13`, `cross-backend:A26` | Todo 19 CPU color; GPU storage/clamp deferred dossier |
| `tsdf:T18`, `tsdf:T24`, `tsdf:T25` | Todo 32 CPU cleanup |
| `tsdf:T23` | Todo 33 |
| `meshing:A1`, `meshing:A2`, `meshing:D2`, `cross-backend:A7` | Todos 6-7 CPU table; backend table deferred dossier |
| `meshing:D1` | Todo 4 CPU shared table; backend duplicates deferred dossier |
| `meshing:B1`, `meshing:B3`, `meshing:B4`, `meshing:C7`, `meshing:D8`, `cross-backend:A14` | Todo 16 |
| `meshing:B2`, `meshing:B6`, `meshing:C1`, `meshing:C2`, `meshing:C3`, `meshing:C4`, `meshing:C6`, `meshing:C9`, `meshing:D5`, `meshing:D7`, `meshing:D9`, `meshing:D10`, `cross-backend:A29` | Todo 17 |
| `meshing:B5`, `cross-backend:A1` | Todo 18 CPU winding; backend winding deferred dossier |
| `meshing:C5` | Todo 28 |
| `meshing:C10` | Todos 8 and 17 |
| `meshing:D3`, `meshing:D4`, `meshing:D6`, `meshing:D11`, `meshing:D12` | Todo 32 |
| `sensor:S-01`, `sensor:S-02`, `sensor:S-03`, `sensor:S-10`, `sensor:S-12`, `sensor:S-20` | Todo 20 CPU depth validity; GPU parity deferred dossier |
| `sensor:S-04`, `cross-backend:A5` | Todo 20 CPU hole fill; GPU parity deferred dossier |
| `sensor:S-05`, `cross-backend:A20` | Todo 21 CPU CAS; GPU parity deferred dossier |
| `sensor:S-06`, `cross-backend:A15`, `cross-backend:A24`, `cross-backend:B5`, `cross-backend:C8` | Todo 22 CPU SR contract; GPU SR deferred dossier |
| `sensor:S-07`, `sensor:S-08`, `sensor:S-09`, `sensor:S-15` | Todo 23 |
| `sensor:S-11`, `cross-backend:B4` | Deferred dossier |
| `sensor:S-13`, `cross-backend:A21`, `cross-backend:A22`, `cross-backend:A23` | Deferred dossier |
| `sensor:S-16`, `sensor:S-19` | Todo 33 |
| `sensor:S-14`, `sensor:S-17`, `sensor:S-18` | Todos 32 and 35 |
| `pipeline:PC-01` | Todo 9 |
| `pipeline:PC-02` | Deferred dossier HIP raycast fall-through |
| `pipeline:PC-03` | Todo 24 |
| `pipeline:PC-04`, `pipeline:PC-05`, `pipeline:PC-06`, `pipeline:PC-07`, `pipeline:PC-15`, `pipeline:PC-19`, `pipeline:PC-21`, `pipeline:PC-22`, `pipeline:PC-24` | Todo 25 |
| `pipeline:PC-08`, `pipeline:PC-09`, `pipeline:PC-11`, `pipeline:PC-12`, `pipeline:PC-13`, `gui:GL-35` | Todo 24 |
| `pipeline:PC-10`, `pipeline:PC-18`, `pipeline:PC-25`, `pipeline:QT-01`, `gui:GL-16` | Todo 26 |
| `pipeline:PC-14` | Verified-safe no-fix: `SharedMesh::update` publishes whole shared_ptrs under mutex |
| `pipeline:PC-16`, `pipeline:PC-20`, `pipeline:PC-26`, `pipeline:JS-01`, `pipeline:RB-01`, `pipeline:RB-02`, `pipeline:RB-03` | Todo 32 |
| `pipeline:PC-17`, `pipeline:LG-01`, `pipeline:LG-02`, `pipeline:TM-01`, `pipeline:JS-02`, `pipeline:JS-03`, `pipeline:JS-04` | Todo 33 |
| `pipeline:PC-23` | Verified-safe no-fix: triple-buffer index invariant was verified by pipeline audit |
| `pipeline:MN-01` | Todo 35 |
| `gui:GL-01`, `gui:GL-05`, `gui:GL-17`, `gui:GL-19`, `gui:GL-20` | Todo 27 |
| `gui:GL-10`, `gui:GL-13`, `gui:GL-14`, `gui:GL-22`, `gui:GL-24`, `gui:GL-25`, `gui:GL-32`, `gui:GL-33`, `gui:GL-34` | Todo 28 |
| `gui:GL-15`, `gui:GL-18`, `gui:GL-21`, `gui:GL-26`, `gui:GL-27`, `gui:GL-28`, `gui:GL-30`, `gui:GL-31`, `gui:GL-36` | Todo 29 |
| `gui:GL-23`, `gui:GL-29` | Todo 32 |
| `export:GLB-01`, `export:GLB-02`, `export:GLB-07`, `export:PLY-05`, `export:MESH-01`, `export:MESH-02` | Todo 10 |
| `export:GLB-03`, `export:GLB-04`, `export:GLB-05`, `export:GLB-06`, `export:GLB-08`, `export:GLB-09`, `export:GLB-10` | Todo 31 |
| `export:PLY-01`, `export:PLY-02`, `export:PLY-03`, `export:PLY-04`, `export:PLY-06`, `export:PLY-07`, `export:PLY-08` | Todo 30 |
| `cross-backend:A8`, `cross-backend:C4`, `cross-backend:C5`, `cross-backend:C6`, `cross-backend:B2` | Todo 32 CPU cleanup; backend hygiene deferred dossier |
| `cross-backend:C1`, `cross-backend:C3`, `cross-backend:C7` | Todos 36 and 35 CPU CPU/docs; GPU gates deferred dossier |
| `cross-backend:C2` | Verified-safe no-fix: CPU variants are required and not dead |
| `cross-backend:A28` | Todo 24 |
| `cross-backend:A34`, `cross-backend:A35` | Todo 32 CPU cleanup; backend deferred dossier |

## Final verification wave

- [ ] F1. Plan compliance audit.
  - Verify every lane ID from `.omo/drafts/big-fix.md` resolves to a lane-prefixed traceability token.
  - Verify each token maps to a CPU todo, verified-safe no-fix, or deferred dossier row.
  - Verify no deferred row claims backend compilation.
  - Verify the plan revision states CPU-only execution and backend deferral.
  - Evidence: `.omo/evidence/big-fix/f1-plan-compliance.txt`, `.omo/evidence/big-fix/f1-traceability-matrix.txt`, `.omo/evidence/big-fix/f1-deferral-matrix.txt`.

- [ ] F2. CPU-only code quality and backend deferral review.
  - Verify no required CPU test invokes `nvcc` or `hipcc`.
  - Verify no script in the required CPU lane configures or builds CUDA/HIP.
  - Verify `.cu` and `.hip` files are unchanged relative to the Todo 2 start point.
  - Verify the deferral dossier contains one row per deferred backend finding.
  - Verify CPU canonical semantics match code.
  - Evidence: `.omo/evidence/big-fix/f2-code-quality.txt`.

- [ ] F3. Real CPU QA execution.
  - Run fresh `build-cpu-big-fix` configure, build, and `ctest --test-dir build-cpu-big-fix -L cpu --output-on-failure`.
  - Record GUI/offscreen QA only if available; GUI is not the sole acceptance vehicle.
  - Record CUDA/HIP as deferred, not compiled, and not runtime-tested.
  - Evidence: `.omo/evidence/big-fix/f3-cpu-qa.txt`.

- [ ] F4. Scope fidelity and artifact delivery.
  - Verify existing `build-cuda/` was not deleted or mutated.
  - Verify no `build-cuda-big-fix` or `build-hip-big-fix` was created by this plan.
  - Verify `context/plans/big-fix.md` contains the CPU-only plan copy.
  - Verify `docs/CUDA_HIP_DEFERRED_CHANGES.md` exists and is referenced by final evidence.
  - Evidence: `.omo/evidence/big-fix/f4-scope.txt`.

## Commit strategy

- Commit order: todo 1, todo 2, todo 3, then todos 4+ in wave order.
- Each todo gets one atomic commit unless it explicitly names a tightly coupled contract group.
- Do not combine CPU behavior fixes with unrelated backend deferral edits.
- Do not amend, rebase, squash, or push unless the user explicitly requests it during execution.

## Success criteria

- Required CPU CTest suite passes in fresh `build-cpu-big-fix`.
- No CUDA/HIP compiler is invoked by required CPU acceptance.
- No CUDA/HIP target is built by required CPU acceptance.
- `docs/CUDA_HIP_DEFERRED_CHANGES.md` contains a complete backend deferral matrix.
- The plan copy is mirrored to `context/plans/big-fix.md`.
- Final evidence explicitly says CUDA/HIP were deferred, not compiled, and not runtime-tested.
