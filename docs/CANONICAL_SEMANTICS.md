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
`min_depth = 0.30f`, `max_depth = 2.50f`. That pair is the single owner, and the two
consumers reach it differently:

- `raycast()` bounds its march from `include/tsdf/TSDFVolume.h` `TSDFParams`
  `min_depth` / `max_depth` (defaults `0.30f` / `5.00f`, the historical raycast
  bounds). `FusionHyperparams` is mirrored into those fields by
  `syncTsdfDepthFromRange()` — the same role `syncIcpDepthFromRange()` plays for
  ICP — and `PipelineController` calls both syncs in its constructor and in
  `setHyperparams()`, before handing `hp.tsdf` / `hp.icp` to the volume and the
  tracker. `TSDFParams` is therefore the raycast's volume configuration plus part
  of the volume's parameter identity, not a second user-facing store.
- `integrate()` / `integrateCPU()` do **not** consult `TSDFParams`: they gate on
  the `min_depth` / `max_depth` arguments the caller passes at that public entry
  point, which `PipelineController` supplies from the same owner.

Because the band is part of the parameter identity that `sameParams()` compares
exactly, changing it in `TSDFParams` clears the volume like any other field (the
Todo 8 rule). Locked by `tests/tsdf_integration_min_depth_contract.cpp`.

Half-live handoff (**known, owned by Todo 24**): after `setHyperparams()` the GUI
depth sliders take effect immediately for the CPU raycast, because the mirror
reaches `params_` through `setParams()`. Integration is not live until restart:
`integrationLoop()` snapshots `d_min` / `d_max` **once** before its worker loop
(`src/app/PipelineController.cpp:818-823`) and passes those locals to every
`integrate()` call, so a band change only reaches integration when the loop is
restarted. Do not describe the integration gate as live mid-session.

The band is a camera-plane **Z-depth** bound in meters, on both the integration
and the raycast side. A ray-cam direction `(u, v, 1)` reaches camera depth `z` at
Euclidean ray parameter `t = z * ||(u, v, 1)||`, so every march that counts `t`
must scale the band by that norm: the integration near clamp is
`max(min_depth * ray_dist_scale, t_meas - truncation)` and the raycast marches
`[min_depth, max_depth] * ||ray_cam||`. Applying the band as a raw ray parameter
instead mislocates every off-axis hit by that ratio.

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

Canonical integration gate: `integrate()` honors the band **passed to it** — both as
the per-pixel validity gate and as the near clamp of the truncation march. The CPU
hard-coded `0.1f` floor (`d_meas < 0.1f`, `t_min = std::max(0.1f, ...)`) that silently
overrode the caller's near bound is gone; both sites now use the `min_depth` argument,
with the march clamp scaled to the ray parameter as described above. `integrateCPU()`
never reads `TSDFParams::min_depth` / `max_depth`; those fields bound the raycast, and
the band reaches integration only through the public entry point's arguments.
**Fixed by big-fix Todo 15**, locked by `tests/tsdf_integration_min_depth_contract.cpp`
(which drives the band through `integrate()`'s parameters). Deferred backend instance:
`tsdf:T8`. The snapshot lifecycle that keeps the GUI slider stale for integration until
restart is Todo 24, not a Todo 15 defect.

Canonical raycast: the march is bounded by `params_.min_depth` / `params_.max_depth`
scaled to the ray parameter (`t = depth * norm(ray_cam)`), never by the literals `0.3f`
/ `5.0f` that CPU, CUDA and HIP all used to hard-code. CPU now samples the trilinear field
at a uniform `0.5 * voxel_size` step — the crossing is interpolated between two adjacent
samples, so a coarser step skips features thinner than the step — and resolves the hit
with `t_prev + (t - t_prev) * f_prev / (f_prev - f_cur)`, where the divisor is the step
actually taken (the legacy `t - vs * tsdf / (tsdf - prev)` assumed a full-voxel step
while the refinement stepped half a voxel, biasing every hit by ~half a voxel). A
non-finite sample aborts that ray to the canonical no-surface state, and vertex,
normal and color are written together only for a resolved finite hit, so a rejected ray
can never leave a stale vertex behind a fresh color. Color is read from the voxel the
resolved hit actually falls in, and only when that voxel is fused (`weight > 0`).
**Fixed by big-fix Todo 15**, locked by `tests/tsdf_raycast_contract.cpp` and
`tests/tsdf_subvoxel_thin_feature_contract.cpp`. Both backends still hard-code the
bounds and the legacy interpolation (`tsdf:T7`, `tsdf:T16`, `tsdf:T19`, `cross-backend:A11`
/ `A12` / `A27`). Note the volume's own extent is `256 × 0.010 m = 2.56 m`, so under the
default `max_depth = 2.50f` the configured far bound, not the extent, is the limit.

Canonical ray crossing: a TSDF exit crossing (`prev < 0 && cur >= 0`) must be detected,
not only the entry crossing. CPU detects both directions, so a ray that starts inside
material reports the back face (with the flipped normal) instead of nothing, and one
empty sample no longer resets a pending crossing — the old `prev_tsdf = EMPTY_TSDF`
reset could also *fabricate* a crossing for an interior start. CUDA and HIP still detect
only `prev_tsdf > 0.0f && tsdf <= 0.0f`: **known CPU defect fixed by big-fix Todo 15**,
deferred backend parity as `tsdf:T26`.

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

- Adaptive Tikhonov damping is implemented as of big-fix Todo 13 through
  `include/tracking/ICPShared.h`: level 0 uses `0.01`, coarse levels use `0.1`,
  and the solver escalates the diagonal to `1.0` when the undamped Hessian has
  `cond > 1e7` or `min_eig < 1e-4`. Locked by
  `tests/icp_numeric_policy_contract.cpp`.
- Translation and rotation are capped at `0.2 m` and `0.5 rad` per accepted
  update. An update violating either cap is rejected without mutating the pose
  or `final_step`; `PipelineController.cpp`'s post-hoc whole-pose check remains a
  pipeline safety net, not the per-iteration cap.
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
- SVD re-orthonormalization uses the shared determinant-corrected `projectToSO3`
  helper as of big-fix Todo 13: full SVD, flip the last `U` column when
  `det(U·Vᵀ) < 0`, then return `U·diag(1,1,det(U·Vᵀ))·Vᵀ`. The result is a proper
  rotation, not a reflection. Backend parity is deferred as `tracking:GPU-8`.
- `ICPResult` initializes `pose` to identity and `final_step` to `+infinity`.
  `final_step` is the Euclidean norm of the last accepted six-vector update
  (translation in metres and rotation in radians); it remains `+infinity` when no
  update is accepted. `tracking_ok = pose.allFinite() && inliers > 100 &&
  (converged || final_step <= 1e-3)`.
- `ICPParams::angle_threshold` is clamped to `[0,85]` degrees at construction and
  in `setParams()`; a non-finite value falls back to `30`.
- `ICPParams::min_depth` / `max_depth` are assigned from
  `include/app/FusionHyperparams.h` and are **never read** by any of the three
  backends. Canonical: either honor them or delete them; a GUI-facing field that
  silently does nothing is not a contract.
- Shared defaults, single source of truth, already canonical:
  `max_iterations{10, 5, 4}`, `dist_threshold = 0.1f`, `angle_threshold = 30.0f`
  from `include/tracking/ICPTracker.h`, sanitized through the `ICPTracker`
  constructor and `setParams()`.
- CPU excludes the 1-pixel image border from the correspondence scan
  (`for y in [1, H-1)`, `for x in [1, W-1)`); the GPU scans the full frame.
  Canonical CPU counter definitions: `valid_live_points` counts finite live
  vertices that pass the positive reference-camera depth test;
  `projected_points` counts those whose rounded model pixel is in bounds;
  `valid_model_points` counts projected samples after model vertex/normal
  validity gates and before distance/angle filtering; `dist_filtered` counts
  valid-model samples outside `dist_threshold`; `angle_filtered` counts
  distance-passing samples whose live normal is valid but fails the angle test.
  A live-normal failure is a silent drop and does not increment `angle_filtered`.
- Validity predicates are shared as of big-fix Todo 13: model vertices must be
  finite with `||v||² > 1e-12`; model and live normals must be finite with
  `||n||² > 0.9`. There is no upper normal bound and surviving normals are not
  renormalized.
- Consequence: `metrics_.icp_overlap_pct`, computed as `valid_model_points /
  valid_live_points`, is not yet comparable across backends because the GPU still
  places `valid_live` differently and evaluates distance/angle in
  reference-camera space while CPU uses world space. Canonical: one definition
  per counter, CPU-defined, and the distance/angle gates must be evaluated in one
  named space. Backend port is deferred as `tracking:A10`, `tracking:A11`, and
  `tracking:A17`.
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

CPU TSDF integration determinism is **fixed by big-fix Todo 14**. `integrateCPU()`
used to run `#pragma omp parallel for` over pixels and read-modify-write
`voxels_[idx(vx,vy,vz)]` from every thread, so two pixels whose ray marches
overlapped a voxel raced on `weight` / `tsdf` / color and the fused voxel changed
run to run. It is now a deterministic two-phase merge: Phase 1 derives each
candidate (target voxel, `tsdf_new`, color-eligibility, its own RGB) from a single
(pixel, step) with no shared write, and Phase 2 applies every candidate **serially in
canonical raster + march order** — increasing `y`, then increasing `x`, then increasing
march step — through the unchanged weight / TSDF / color formulas (`w_new = 1`, weight
cap `max_weight`, TSDF blend denominator `w_old + 1 + 1e-6f`, color blend over the same
denominator, color eligibility `rgb && sdf > -trunc * 0.5f`). Candidates are ordered by
a lossless `((y*width+x) << 32) | step` sequence key, so the fold is a pure function of
the input: byte-identical across repeated runs and across OpenMP thread counts 1 / 2 /
4, and identical to the single-thread fold. Locked by
`tests/tsdf_integration_race_contract.cpp` (three-run byte-identical volume + mesh and
a hand-derived double blend oracle). The depth `0.1f` floor, the `vs * 0.75f` march
step, the `sdf < -trunc` rejection, and the NaN/Inf guards are unchanged; the
configured `min_depth` is still owned by the later depth-domain todo (Todo 15). The
CUDA/HIP integration kernels keep the same concurrent read-modify-write and are
untouched — deferred, not compiled, not runtime-tested (`tsdf:T3`).

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
