# Canonical CPU semantics for the big-fix run

Status: authoritative for this run; CUDA/HIP parity is deferred and is recorded
in `docs/CUDA_HIP_DEFERRED_CHANGES.md`. Nothing in this file was verified by
compiling or running a backend; every statement below is a CPU-path statement
backed by the CPU sources named in each section.

## How to read this file

Each topic separates four different things, and a statement is meaningless
unless its layer is named:

- **Canonical** — the rule this run declares correct. Backends must later
  converge here.
- **Current CPU behavior** — what the CPU code does today on HEAD `7151f5f`.
  It is not automatically canonical.
- **Known CPU defect** — current CPU behavior that contradicts the canonical
  rule and is owned by a later big-fix todo. Recorded here so the divergence is
  not rediscovered.
- **Deferred backend** — a CUDA/HIP divergence from the canonical rule. Only
  the ID and the deferral live in `docs/CUDA_HIP_DEFERRED_CHANGES.md`; nothing
  backend was compiled or runtime-tested.

## Depth domain

Canonical invalid predicate: raw `0` and raw `>= 2047` are invalid. Raw `2047`
is a sensor "no valid depth returned" sentinel, not a fillable hole, and it must
never be interpolated over.

Current CPU behavior in `include/sensor/KinectSensor.h` `rawDepthToMeters()`:

```cpp
if (raw == 0 || raw >= 2047) return 0.0f;
return 1.0f / (raw * -0.0030711016f + 3.3309495161f);
```

The invalid predicate is therefore already canonical. The curve is not.

The reciprocal form has a physical pole at
`raw = 3.3309495161 / 0.0030711016 ≈ 1084.6`. Consequences, all reachable
inside the raw 11-bit band:

| Raw | Meters | State |
|---|---|---|
| `0` | `0.0` (rejected) | invalid, canonical |
| `1` | ~0.331 | inside the default band |
| `954` | ~2.499 | last raw inside the default band |
| `955`–`1003` | 2.50–4.00 | above `max_depth`, out of band |
| `1004`–`1084` | 4.00 → ~533 | runaway approach to the pole |
| `1085` | ~ -217 | **negative** |
| `1085`–`2046` | negative, rising toward ~ -0.0 | **negative but finite and non-zero** |
| `2047` | `0.0` (rejected) | invalid, canonical |

Canonical depth-to-meters rule: output must be finite and inside the configured
band, and anything else becomes invalid (zero) rather than a clamped wall or a
negative value. The pole means a raw value that is not the sentinel can still
produce a geometrically impossible negative distance. The invalid predicate
alone is not a sufficient validity gate; a downstream band check is mandatory.

Canonical band source: `include/app/FusionHyperparams.h`
`min_depth = 0.30f`, `max_depth = 2.50f`, and
`FusionHyperparams::syncIcpDepthFromRange()` propagates that same pair to ICP.
`include/tsdf/TSDFVolume.h` `TSDFParams` does **not** carry `min_depth` /
`max_depth` fields at all: the struct holds only `resolution`, `voxel_size`,
`truncation`, `max_weight` and `origin`. The configured band therefore reaches
ICP but not the volume, and no TSDF entry point can consult it.

Known CPU defect (owned by the later CPU depth-domain todo, not a backend
divergence): propagate the configured `0.30f` / `2.50f` band into `TSDFParams`
as explicit `min_depth` / `max_depth` fields and honor it in integration and
raycast. Today the TSDF stage ignores the configured band entirely, which is why
the integration and raycast rules below are canonical rules the CPU does not yet
satisfy. See `tsdf:T8` and `tsdf:T19` in the dossier.

Known CPU defect (owned by the depth-domain todo): the raw band and the
min/max band are enforced at different layers, and the CPU hole fill in
`src/sensor/SignalConditioner_omp.cpp` `fillDepthHoles()` treats only raw `0`
as a hole, so raw `2047` is left in place. That matches the canonical rule
(raw `2047` is not fillable). The divergence is with the backends, which treat
raw `2047` as a hole; see `cross-backend:A5` / `sensor:S-04` in the dossier.

## TSDF volume

Canonical empty-voxel state: `tsdf = 1.0f`, `weight = 0.0f`, neutral gray
color. `+1.0f` means "unobserved / free space, not yet truncated", and
emptiness is decided by `weight`, never by the `tsdf` literal.

Current CPU behavior:

- `include/tsdf/TSDFVolume.h` `struct Voxel` defaults to
  `tsdf = EMPTY_TSDF`, `weight = EMPTY_WEIGHT`, `r = g = b = EMPTY_COLOR`
  (128) — canonical, and a `static_assert` in the same header makes
  `Voxel{}` equal to the named constants at compile time.
- `src/tsdf/TSDFVolume.cpp` `unlocked_reset()` fills
  `Voxel{EMPTY_TSDF, EMPTY_WEIGHT, EMPTY_COLOR, EMPTY_COLOR, EMPTY_COLOR}` and
  zeroes `integrated_frames_`, so `reset()` restores the canonical empty state
  exactly. Fixed by the TSDF-lifecycle todo: the fill used to be
  `Voxel{0.0f, 0.0f, 128, 128, 128}`, which put every reset voxel on the zero
  isosurface so an unobserved voxel could be meshed as surface. The CPU
  contract is `tests/tsdf_reset_contract.cpp`.
- `include/tsdf/TSDFVolume.h` now names the CPU sentinel pair
  `EMPTY_TSDF = 1.0f` / `EMPTY_WEIGHT = 0.0f` (plus `EMPTY_COLOR = 128`), and
  `unlocked_reset()`, `getTSDF()` and the `raycast()` empty-space markers use
  them. Nine raw `1.0f` literals remain in `src/tsdf/TSDFVolume.cpp` and none of
  them is an empty-voxel marker: two are the ray projection `z = 1`, two are the
  `[-1,1]` truncation clamp, one is the per-frame weight increment `w_new`,
  three are the color fusion denominators, one is the point-cloud
  `weight > 1.0f` gate. The backends still carry no such constant, and CPU
  meshing still gates a cube on `weight <= 0.001f` rather than on
  `EMPTY_WEIGHT`: one named epsilon remains open (`tsdf:T2`, `tsdf:T5`,
  `tsdf:T11`, `tsdf:T24`).
- Emptiness thresholds are inconsistent across layers: `weight > 0.0f` in the
  CPU trilinear sampler, `weight <= 0.001f` in CPU meshing. Canonical: one
  named epsilon.
- `src/tsdf/TSDFVolume.cpp` `getTSDF()` samples an unobserved voxel as
  `EMPTY_TSDF` — canonical, and it is the rule Marching Cubes relies on.
- `src/tsdf/TSDFVolume.cpp` `setParams()` compares every field of the incoming
  `TSDFParams` against the current one and clears the host volume (and
  `integrated_frames_`) on **any** difference, reallocating only when
  `resolution` changed. Fixed by the same todo: it used to clear only when
  `resolution` changed, so a `voxel_size`, `origin`, `truncation` or
  `max_weight` change left voxels fused under the old parameters, and
  `worldToVoxel()` / `voxelToWorld()` reinterpreted them through the new
  geometry. An exactly-equal parameter set clears nothing.
- `include/tsdf/TSDFVolume.h` `voxelAt()` is documented "bounds checked" while
  `src/tsdf/TSDFVolume.cpp` implements it as a bare `voxels_[idx(x, y, z)]`
  with no bounds test. **Known CPU defect**: either bounds-check it or fix the
  comment; the header must not promise a check that does not exist.
- `include/tsdf/VoxelGPU.h` `VoxelGPU` is 32 bytes with `float padding[3]`
  (12 permanently unused bytes per voxel) and no `static_assert` on its size;
  host `Voxel` is 12 bytes with `uint8_t` color. The layout translation happens
  in exactly one place per backend. Deferred as `cross-backend:B3`.

Canonical integration gate: integration honors the configured `min_depth`. The
hard-coded `0.1f` in `src/tsdf/TSDFVolume.cpp`
(`d_meas < 0.1f` and `t_min = std::max(0.1f, ...)`) must not override the
configured value. Deferred backend instance: `tsdf:T8`.

Canonical raycast: the configured near/far bounds are used, not literals. CPU
currently hard-codes `t = 0.3f` and `t < 5.0f`
(`src/tsdf/TSDFVolume.cpp` `raycast()`), identical to both backends
(`tsdf:T19`). Note the volume's own extent is `256 × 0.010 m = 2.56 m`, so the
`5.0f` far bound is never the limiting factor. Both bounds are hard-coded
literals: the near bound is a Kinect-v1 constant, and neither one is a
`TSDFParams` field today (see the band-source note and the known CPU defect
recorded with it).

Canonical ray crossing: a TSDF exit crossing (`prev < 0 && cur >= 0`) must be
detected, not only the entry crossing. CPU, CUDA and HIP all detect only
`prev_tsdf > 0.0f && tsdf <= 0.0f`. **Known CPU defect** plus deferred backend
parity (`tsdf:T26`): a ray that starts inside material never resolves an exit,
and a single empty sample resets the pending crossing.

Canonical `worldToVoxel` / projection rounding: floor semantics. Every CPU
coordinate conversion now floors through the shared helper
`kfusion::utils::floorToInt` (`include/utils/CoordinateMath.h`): `worldToVoxel()`
and the ICP model-pixel projection (`std::floor(model + 0.5)`) both floor and
reject a non-finite or out-of-`int` input before any integer coordinate is
produced, so CPU integration, `worldToVoxel()` and projection agree for negative
coordinates. A non-finite or out-of-range `worldToVoxel()` input returns the
out-of-bounds sentinel `(INT_MIN, INT_MIN, INT_MIN)`, which `inBounds()` rejects
(so a below-origin point is no longer truncated to voxel `(0,0,0)`). **Fixed by
big-fix Todo 11**, locked by `tests/coordinate_rounding_contract.cpp`; the
backend instance remains deferred as `cross-backend:A18` / `A19`.

## Marching Cubes and winding

Canonical source of truth: one shared CPU table header. The CPU path consumes
`include/meshing/MarchingCubesTables.h`
(`kfusion::meshing::tables::{edge_table, tri_table}`) as of commit `7151f5f`.
The CUDA and HIP TUs still carry hand-copied duplicates (`meshing:D1`,
`cross-backend:A8`).

Canonical winding rule: **the outward-normal fixture is the CPU authoritative
winding rule.** The CPU signed-volume / outward-normal test derived from the
reference sphere construction (big-fix todo 6) is the single arbiter of
triangle orientation for this run. A mesh is correct when its face normals
point away from the signed-volume interior, and the CPU test is what decides
that. Backends do not get their own winding opinion: CUDA and HIP winding must
later be derived from the same fixture, and any backend that disagrees is the
one that is wrong.

Current CPU behavior: `src/meshing/MarchingCubes.cpp` emits triangles in
reverse table order (`for (int i = 2; i >= 0; --i)`, comment "Reverse winding
(2, 1, 0 instead of 0, 1, 2) to fix front-face culling"). HIP matches CPU
(`2, 1, 0`); CUDA emits forward order `0, 1, 2` with no explanatory comment,
so the CUDA file currently reads as the unpatched original
(`meshing:B5` / `cross-backend:A1`). Whether `2, 1, 0` is actually right is
exactly the question todo 6's fixture answers; until then the CPU reverse order
is canonical *by definition of the fixture*, and the fixture is the artifact
that gets validated.

Known CPU defect (carried deliberately): `edge_table[213] = 0x835`,
`edge_table[214] = 0xb3f`, `edge_table[215] = 0xa36` in all three copies
(CPU header line 57, CUDA line 62, HIP line 58). The canonical reference values
are `0x83f` / `0xb35` / `0xa3c`, and the correct mirror row already exists in
all three files at CPU header line 36, CUDA line 41, HIP line 37 — the corrupt
row is its own mirror, which is what identifies it as corruption rather than a
variant. Todo 4 was a value-preserving move, so the corruption is intentional
and untouched; todo 6 adds the failing analytic tests and todo 7 corrects the
shared header only. The `// fix: 0x835→0xb35 variant` comment at
`include/meshing/MarchingCubesTables.h:57` is itself wrong: it claims a fix and
names the wrong target (`[213]` canonical is `0x83f`, not `0xb35`). Deferred
backend propagation: `cross-backend:A7`.

Canonical mesh vertex validity: a cube may mesh only when a crossing edge is
supported by weighted voxels, unobserved voxels sample as `+1.0f`, and a
position that is not finite is never emitted.

Current CPU behavior: **no finite-value guard at all.** The only cube-level gate
is `vox.weight <= 0.001f`; emission at `src/meshing/MarchingCubes.cpp:221-224`
tests nothing. **Known CPU defect**: NaN positions and NaN normals can reach
`MeshData`. Related CPU defects in the same function:
`edge_norms[e] = (n0 + t * (n1 - n0)).normalized()` divides by zero for a
cancelled pair (no `isApprox(0)` test), and `computeNormal()` maps a degenerate
normal to a fabricated `(0, 0, 1)` instead of rejecting the vertex, so a
bad-faith normal is indistinguishable from a real one. Backends replace NaN with
the world origin and then emit it as a legitimate vertex
(`meshing:C2`), which is a different wrong, not a fix.

Canonical welding: exact edge identity where practical. The CPU weld in
`src/meshing/MarchingCubes.cpp` keys a hash map on a quantized float position
(`QuantizedHash`, divisor `voxel_size * 0.01f`), which is not exact edge
identity and silently merges distinct vertices that quantize together.
**Known CPU defect.** The HIP weld adds a distance-validated escape branch
(`merge_threshold = voxel_size * 0.5f`) that is unreachable for finite input —
two positions that collide under a `0.01·vs` quantizer differ by less than
`0.0087·vs`, always under the threshold — and if it ever did fire it would
overwrite the map entry instead of chaining (`meshing:D3` / `cross-backend:A29`).

## Color pipeline and export

Canonical storage chain:

```
volume voxel      float sRGB [0,1]
MeshData.colors   uint8 sRGB
GLB COLOR_0       linear float [0,1]
PLY colors        uint8 sRGB
```

Explicit canonical statements:

- **GLB `COLOR_0` is linear float.** `COLOR_0` in a glTF `_accessor_` is
  emitted as a `VEC4` `float` accessor whose components are
  **linear-light** values in `[0, 1]`, per the glTF 2.0 `COLOR_0` semantics
  (`vertex colors ... in linear space`). It is not uint8, and it is not sRGB.
  The uint8 sRGB → linear float conversion is mandatory at the exporter
  boundary; a raw `value / 255.0f` is sRGB-encoded and is **not** a compliant
  `COLOR_0`.
- **PLY stays sRGB uint8.** PLY has no color-space tag in this writer, so it
  keeps the raw device-space uint8 triple and must not be treated as linear.
- Voxel color is float sRGB `[0,1]`; `MeshData::colors` is `uint8` RGB.

Current CPU behavior:

- `include/meshing/MeshData.h` stores `std::vector<uint8_t> colors` (RGB, no
  alpha). Canonical-consistent as the uint8 sRGB stage.
- `src/export/GLBExporter.cpp` emits `COLOR_0` as a `VEC4` `float` accessor
  built from `uint8 / 255.0f` with alpha `1.0`. The container is therefore
  already linear-*typed* float, but the values are sRGB-encoded, so the output
  does not satisfy "GLB COLOR_0 is linear float". **Known CPU defect** owned by
  the export/color todo: apply sRGB→linear decode to the RGB components (alpha
  stays as-is) and keep the float `VEC4` accessor.
- `src/export/GLBExporter.cpp` converts Kinect Y-down to glTF Y-up by negating
  Y and flipping Z; the README documents `Y → -Y` and Unity's importer applying
  the handedness flip. Any winding-rule change must be re-checked against this
  transform, because the export flips handedness twice and can mask a winding
  bug in one of the two stages.
- `src/export/PLYExporter.cpp` writes binary PLY with
  `property uint8 red/green/blue` but the ASCII writer emits
  `property uchar ...` plus `property list uchar uint vertex_indices`. Schema
  drift between the two writers for the same logical mesh. **Known CPU defect**
  (owned by the export todo): one schema, one spelling, both writers.
- `src/meshing/MarchingCubes.cpp` interpolates edge color with an unclamped
  `static_cast<uint8_t>` of a float, so an out-of-range interpolation wraps as
  undefined behavior instead of saturating. HIP deliberately matches CPU
  ("simple cast without clamping"), CUDA clamps with
  `fminf(255.0f, fmaxf(0.0f, ...))`. Canonical: one clamped, non-UB
  float→uint8 conversion everywhere; deferred backend rows `meshing:C8` /
  `cross-backend:A26`.

## ICP / tracking

Canonical CPU ICP policy:

- Adaptive Tikhonov damping: default `0.1`, level 0 `0.01`, escalating to `1.0`
  when the Hessian is ill-conditioned, with a condition-number test.
- Non-finite rejection on the assembled system and on the update.
- Translation cap `0.2 m` and rotation cap `0.5 rad` per iteration.
- A Huber-weighted objective reported consistently: with
  `w = min(1, k / abs_err)` (`k = 0.02f`), the IRLS/MM surrogate
  `0.5*sum(w_i * e_i^2)` makes the canonical triple — curvature
  `A += w*J*Jᵀ`, gradient `b -= J*(w*e)`, and the reported per-correspondence
  objective the **true Huber loss** `psi(abs_err)` (`t^2` for `t <= k`,
  `2*k*t - k^2` beyond). The pre-Todo-12 rule "if the gradient uses `w·e`, the
  curvature must use `w²·JᵀJ`" is **false and withdrawn**: `w²*JᵀJ` pairs with
  a `w²*e*J` gradient, not the `w*e` gradient every file computes, and
  `0.5*(w*e)^2` is bounded by `2*k^2` so it cannot majorize the unbounded
  Huber loss it claims to report. The weight is `w` everywhere; see
  `include/tracking/ICPShared.h`.
- A rotation matrix re-orthonormalization that cannot return a reflection.

Current CPU behavior (`src/tracking/ICPTracker.cpp`):

- Damping is a constant `A += I * 0.1f` at every level, with no
  `SelfAdjointEigenSolver`, no condition-number test, and no escalation path.
  **Known CPU defect** — the CPU is the backend that is missing the canonical
  adaptive policy that the GPU already implements
  (`ICPTracker_cuda.cu:419-435`: level-0 `0.01f`, escalation `1.0f`,
  `cond > 1e7f || min_eig < 1e-4f`). The canonical rule is the adaptive one;
  CPU converges to a different fixed point at level 0 by a factor of 10.
- Translation is capped at `0.2f`; **rotation is never capped**, so an
  unbounded per-iteration rotation is accepted. GPU caps it at `0.5f`.
  **Known CPU defect** (`cross-backend:A10`). The only backstop is
  `PipelineController.cpp`'s post-hoc `angle > 0.52f` whole-pose rejection.
- Huber weight `k = 0.02` **fixed by big-fix Todo 12**: it now applies
  consistently to the curvature (`A += w*J*Jᵀ`) and the gradient
  (`b -= J*(w*e)`), and the reported objective is the true Huber loss
  `psi(abs_err)`. The pre-Todo-12 code accumulated an unweighted curvature
  (`A_data[k++] += J[i]*J[j]`), a `w*e` gradient, and a `(w*e)²` objective —
  three mutually inconsistent quantities. `k` and `psi` now live in one shared
  header `include/tracking/ICPShared.h` (`kHuberK`, `huberLossFromAbs`), the
  single source of truth for CPU. This is a shared CPU/GPU correctness item,
  not a divergence; the backend port is deferred as `tracking:A13`. The CPU
  triple is locked by `tests/icp_weighting_contract.cpp`.
- Non-finite rejection **fixed by big-fix Todo 12**: the residual is checked
  finite *before* the weight is computed, and all six Jacobian components are
  guarded (the pre-Todo-12 guard sampled only `J[0]` and `J[3]`). A NaN model
  depth yields a NaN residual with all six Jacobian entries finite, so the
  residual guard is mandatory and cannot be replaced by a Jacobian-only guard;
  the two guards are independent and both are required. Locked by
  `tests/icp_weighting_contract.cpp` (an `+Inf` model normal and a NaN model
  depth each leave the 11 clean correspondences' pose and objective
  bit-identical). Backend parity is deferred as `tracking:ALL-3` (both backends
  still sample only `J[0]`/`J[3]` and never test the residual).
- SVD re-orthonormalization is `svd.matrixU() * svd.matrixV().transpose()` with
  **no determinant correction**, identically on CPU and both backends. `U·Vᵀ`
  can have `det = -1`, which is a reflection and not a rotation; the correct
  form is `U·diag(1,1,det(UVᵀ))·Vᵀ`. Zero `determinant`/`.det()` hits anywhere
  in `src/tracking/` or `include/tracking/`. This is a shared defect and must
  not be filed as a backend divergence.
- `include/tracking/ICPTracker.h` `ICPResult::pose` is declared without an
  initializer, so a result that returns before the pose is written carries an
  indeterminate matrix.
- `ICPParams::min_depth` / `max_depth` are assigned from
  `include/app/FusionHyperparams.h` and are **never read** by any of the three
  backends. Canonical: either honor them or delete them; a GUI-facing field that
  silently does nothing is not a contract.
- Shared defaults, single source of truth, already canonical:
  `max_iterations{10, 5, 4}`, `dist_threshold = 0.1f`, `angle_threshold = 30.0f`
  from `include/tracking/ICPTracker.h`.
- CPU excludes the 1-pixel image border from the correspondence scan
  (`for y in [1, H-1)`, `for x in [1, W-1)`); the GPU scans the full frame. CPU
  increments `valid_live` only after the `v_ref.z > 0.001f` test; the GPU
  increments it before. CPU additionally gates on a live normal with
  `live_n.z() == 0.0f` (itself a bug: `(0,1,0)` and a `(1,0,0)` sentinel both
  pass), which the GPU does not do at all. Consequence:
  `metrics_.icp_overlap_pct`, computed as `valid_model_points /
  valid_live_points`, is not comparable across backends. Canonical: one
  definition per counter, CPU-defined, and the distance/angle gates must be
  evaluated in one named space (CPU measures in world space, GPU in
  reference-camera space).
- CPU pyramid downsampling (`src/sensor/FrameData.cpp` `downsample()`) **has
  the depth-jump guard as of big-fix Todo 12**. A 2×2 block whose valid depths
  span more than the shared `depthJumpThreshold(d_min) = max(0.03, 0.05·d_min)`
  is **not** averaged (averaging blended both surfaces into a ghost vertex at
  every coarse level); the whole nearest (smallest-depth) valid sample is kept
  instead, deterministic by first-minimum scan order. Blocks within the
  threshold average exactly as before, sentinel `0` samples are still excluded,
  and fully-invalid blocks stay zero. The threshold constant moved into the
  shared `include/tracking/ICPShared.h` and the level-0 normal kernel now reads
  the same value, so the two cannot drift. Canonical: the depth-jump guard
  applies at every level. The level-0 normal kernel is unchanged. Locked by
  `tests/icp_weighting_contract.cpp`. Deferred backend instance:
  `tracking:GPU-1` / `tracking:A34`.

## Super-resolution and conditioning

Canonical border mode: reflect. Current CPU behavior is reflect everywhere
(`SuperResolution.cpp` `reflectCoord()`, `SignalConditioner_omp.cpp`
`reflectCoord()`), and CUDA/HIP `SuperResolution_{cuda.cu,hip.hip}` also
reflect — except `applyCASKernel`'s `rcasGetPixelCUDA` / `rcasGetPixelHIP`,
which clamp (`max(0, min(x, w - 1))`). The clamp is the isolated outlier and it
doubles the edge value into the RCAS cross-stencil, over-sharpening a visible
1-pixel frame at exactly the border that feeds the UI preview. Deferred as
`sensor:S-05` / `cross-backend:A20`. The comment claiming the two kernels are
"the same algorithm" is false: identical math, different boundary extension.

Canonical sharpness clamp: NaN must not produce NaN. GPU
`std::max(0.0f, std::min(sharpness, 1.0f))` maps `NaN → 0.0` (a valid mild
sharpen, because `std::min(NaN, 1.0f)` is `NaN` and `std::max(0.0f, NaN)`
returns its first argument). CPU `std::clamp(sharpness, 0.0f, 1.0f)` returns
`val` unchanged when both comparisons are false, so `NaN → NaN → peak = NaN →
static_cast<uint8_t>(NaN * 255.0f)`, which is undefined behavior. Canonical:
an explicit finite test, not an accident of the clamp used. Deferred as
`cross-backend:B5`.

Canonical super-resolution naming: the pass must be called what it is. The
"EASU" implementation is a fixed 4×4 separable Catmull-Rom bicubic resample —
no local min/max edge-direction estimate, no anisotropic kernel, no
`tanDistSq`, no gradient vectors, no 12-tap footprint — in both the HIP kernel
(`easuKernel`, `cubic()`) and the CPU reference. The CPU comment "Bicubic
interpolation with edge-adaptive sampling" and the attribution to the FSR paper
describe an algorithm the code does not implement. Canonical: either implement
edge-adaptivity or rename the pass to bicubic; do not keep a paper citation
above code that does not match it. Deferred as `sensor:S-18`.

Canonical failure contract: an allocation or launch failure must be reported,
never absorbed. Current CPU-side reality: `SignalConditioner::process()`
returns early on a frame-size mismatch
(`SignalConditioner_omp.cpp:264`), `processCuda` returns `true`
unconditionally and throws on allocation failure instead of returning `false`
(deferred `cross-backend:A22`), and `applyEASU_GPU` is `void`, zero-filling
`dst` before the allocation so a failed allocation yields a black frame that is
consumed as valid (deferred `sensor:S-06` / `cross-backend:A24`).

Canonical temporal EMA: single-buffer output. `applyDepthEma()` writes
`depth[i]` while other threads read their neighbours' current-frame values from
the same buffer, so the reset decision is made on partially updated data and
the result differs every run. **Known CPU defect** (a data race), with the same
shape in both GPU EMA kernels. Every other depth pass correctly ping-pongs
between the in/out buffers; only the EMA is in-place.

Canonical upscaled-RGB path: either consumed or deleted.
`sr_rgb_upscaled_` is produced every frame (a full 4× bicubic upsample plus
RCAS over 3.69 MB, plus a 3.69 MB copy) and its only consumer,
`PipelineController.cpp:505`, is commented out with "causes black textures" —
which is precisely the `applyEASU_GPU` zero-fill bug above. On CUDA the buffer
is never even written, because `SuperResolution_cuda.cu` implements no EASU at
all and `applyEASU` dispatches to CPU there. Deferred as `cross-backend:C8`.

Canonical super-resolution scale: `setSrScale()` must have an effect.
`sr_scale_` is never read in either GPU conditioner, so the GUI slider is a
silent no-op on GPU. Deferred as `cross-backend:B2`.

## Pipeline publication

Canonical rule: a published model frame implies a raycast-written buffer.

Current behavior in `src/app/PipelineController.cpp`: `emitCpuPreview()` is
defined under `#ifndef HIP_ENABLED`, and the `#elif defined(HIP_ENABLED)` raycast
block ends in an empty `else { // Fall through to CPU raycast }`. Under HIP with
`use_gpu_ == false` no raycast runs at all, yet `model_buffers_.swap()` and
`model_ready_.store(true)` still publish, so tracking consumes an all-zero or
stale model frame. Deferred as `pipeline:PC-02`.

Canonical depth band reaches the consumer: `min_depth` / `max_depth` must be
read inside the integration loop, not snapshotted before it
(`PipelineController.cpp:778-783` caches `d_min`/`d_max` before the
`while (running_.load())` loop at `:785`), or a live slider change silently does
nothing until restart.

## Determinism and tolerance

Canonical CPU determinism: repeated CPU runs over identical input are
byte-identical where a test asserts it. That is why the EMA race above is a
correctness defect and not a cosmetic one: it breaks the repeat contract.

GPU is not deterministic in the same sense, and this is documented rather than
fixed: the Hessian is reduced with `atomicAdd` over ~300 blocks, so the fp32
accumulation order varies frame to frame and the pose is not bit-reproducible.
Backend numeric tolerances belong to a future backend plan; the CPU lane claims
byte-identity and nothing here grants a backend a tolerance it has not earned.

## Backend parity pointer

Every CUDA/HIP divergence named above is a row in
`docs/CUDA_HIP_DEFERRED_CHANGES.md` with an audit ID, a file/function, the CPU
decision, and a future acceptance test. Each such row is marked `deferred`,
`not compiled`, and `not runtime-tested`. No backend translation unit was
compiled, linked, or executed in this run, and no CPU test is evidence that a
backend compiles or agrees.
