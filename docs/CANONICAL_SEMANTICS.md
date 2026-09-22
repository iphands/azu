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
- **Current CPU behavior** — what the CPU code does today on the HEAD pinned
  below. It is not automatically canonical.
- **Known CPU defect** — current CPU behavior that contradicts the canonical
  rule and is owned by a later big-fix todo. Recorded here so the divergence is
  not rediscovered. A defect this plan already repaired is **not** one of these:
  it is labeled CPU-fixed, and that label never implies a backend agrees.
- **Deferred backend** — a CUDA/HIP divergence from the canonical rule. Only
  the ID and the deferral live in `docs/CUDA_HIP_DEFERRED_CHANGES.md`; nothing
  backend was compiled or runtime-tested.

**Pinned baseline: HEAD `a520cb9`.** Every "Current CPU behavior" statement
below describes the CPU sources at that commit. The pin this file carried
(`7151f5f`, the Todo 4 table-consolidation commit) predates the CPU repairs this
plan landed afterwards — the edge-table repair, the shared PLY header, the dead
code removal and the logger/timer hygiene — so statements written against it had
drifted out of truth even though the code they described had been fixed.

Line-number citations here are point-in-time, taken at the pinned HEAD, and they
drift the moment the cited file is edited. That is why each one names its symbol
on the same line. When a number and its symbol disagree, the symbol is right and
the number is stale: re-derive with `grep -n '<symbol>' <file>` rather than
trusting any number in this file.

## Depth domain

Canonical invalid predicate: raw `0` and raw `>= 2047` are invalid. Raw `2047`
is a sensor "no valid depth returned" sentinel, not a fillable hole, and it must
never be interpolated over.

Current CPU behavior: one boundary owns the whole domain,
`include/sensor/DepthValidity.h`. `include/sensor/KinectSensor.h` includes it
(`rawDepthToMeters()` used to be defined there) and
`src/sensor/SignalConditioner_omp.cpp` includes it directly:

```cpp
if (isInvalidRawDepth(raw)) return 0.0f;   // raw == 0 || raw >= 2047
return 1.0f / (raw * kRawDepthCurveA + kRawDepthCurveB);   // A=-0.0030711016f, B=3.3309495161f
```

The invalid predicate is therefore already canonical. The curve is not.

The reciprocal form has a physical pole at
`raw = 3.3309495161 / 0.0030711016 ≈ 1084.6106`. Consequences, all reachable
inside the raw 11-bit band (values below are the float32 result of the curve):

| Raw | Meters | State |
|---|---|---|
| `0` | `0.0` (rejected) | invalid, canonical |
| `1` | 0.300492 | first raw inside the default band |
| `954` | 2.493029 | last raw inside the default band |
| `955` | 2.512263 | above `max_depth`, out of band |
| `1003` | 3.989871 | out of band |
| `1084` | 533.219 | runaway approach to the pole |
| `1085` | -836.352 | **negative** |
| `1500` | -0.783882 | **negative but finite and non-zero** |
| `2046` | -0.338693 | **negative but finite and non-zero** |
| `2047` | `0.0` (rejected) | invalid, canonical |

So raw `1..954` is exactly the in-band raw set for the default `[0.30, 2.50]`
band (954 contiguous codes), and `cpuDepthMetersToRaw(rawDepthToMeters(raw))`
returns `raw` unchanged for every one of them. The pole means a raw value that is
not the sentinel can still produce a geometrically impossible negative distance.
The invalid predicate alone is not a sufficient validity gate; a band check is
mandatory, and it is applied at this boundary rather than at each consumer.

Canonical depth-to-meters rule, implemented as `cpuDepthMeters(raw, min, max)`:
accept only when `rawDepthToMeters(raw)` is finite and within the configured
band; anything else returns **exactly `+0.0f`** — never `min_depth`, never
`max_depth`, never a wall value, never a negative. The reverse direction
`cpuDepthMetersToRaw(meters, min, max)` rejects out-of-band and non-finite input
with raw `0` instead of clamping `lround(raw)` into `[1, 2046]`, so a wall code
can never be written into a depth frame. `isDepthMetersInBand()` is the shared
finite-plus-band predicate and `isUsableRawDepth()` (valid raw AND in-band
meters) is the neighbour predicate for every filtering window.

All six CPU depth passes go through that boundary, and the named call is the
site each one consults it: `buildFrameData()`
(`src/sensor/FrameData.cpp:155`, `cpuDepthMeters()`), `denoiseDepthSpatial()`
(`src/sensor/SignalConditioner_omp.cpp:443`), `applyDepthEma()` (`:495`),
`fillDepthHoles()` (`:605`), `guidedDepthFilter()` (`:643`) — all
`cpuDepthMeters()` — and `computeDepthGradient()` (`:126`,
`isUsableRawDepth()`). Two further CPU-only rules live with them and are locked
by `tests/depth_domain_contract.cpp` and
`tests/depth_ema_determinism_contract.cpp`:

- Hole fill may change **only** pixels that are raw `0`. Raw `2047` is written
  back exactly as it arrived (demoting it to `0` would silently convert "no
  valid depth returned" into "fillable hole" for every later consumer), and a
  window whose usable neighbours are all sentinels or all out-of-band leaves the
  hole unfilled rather than inventing a value.
- `applyDepthEma()` reads a source snapshot (`depth_src_`) and writes the
  destination (`depth`), so its output is a pure function of
  (input frame, per-pixel history, band) and is bit-identical across thread
  counts. Reading the destination mid-pass lets already-filtered neighbours
  contaminate the neighbour mean, which flips the jump-reset branch per pixel and
  makes the result schedule-dependent.

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

The band is live on **both** consumers (**fixed by big-fix Todo 24**):
`trackingLoop()` and `integrationLoop()` each take one `hyperparamsSnapshot()`
per processed frame — after that frame is popped and before anything consumes it
(`hyperparamsSnapshot()` in `trackingLoop()` at
`src/app/PipelineController.cpp:754`, and in `integrationLoop()` at `:1075`) —
so a `setHyperparams()` issued
while the pipeline runs reaches the ICP, the preprocessor and the next
integrated frame without a restart. The pre-Todo-24 shape, where
`integrationLoop()` cached `d_min` / `d_max` **once** before its worker loop and
passed those locals to every `integrate()` call, so a band change reached
integration only at restart, no longer exists. Locked by
`tests/pipeline_hyperparams_contract.cpp`. Deferred backend instance:
`cross-backend:A28`.

The band is a camera-plane **Z-depth** bound in meters, on both the integration
and the raycast side. A ray-cam direction `(u, v, 1)` reaches camera depth `z` at
Euclidean ray parameter `t = z * ||(u, v, 1)||`, so every march that counts `t`
must scale the band by that norm: the integration near clamp is
`max(min_depth * ray_dist_scale, t_meas - truncation)` and the raycast marches
`[min_depth, max_depth] * ||ray_cam||`. Applying the band as a raw ray parameter
instead mislocates every off-axis hit by that ratio.

The raw band and the min/max band are no longer enforced at different layers on
CPU: `cpuDepthMeters()` applies the raw predicate and the band in one call, and
`fillDepthHoles()` treats only raw `0` as a hole, so raw `2047` is left in place.
That matches the canonical rule (raw `2047` is not fillable). The divergence is
now purely with the backends, which treat raw `2047` as a hole; see
`cross-backend:A5` / `sensor:S-04` in the dossier.

## TSDF volume

Canonical empty-voxel state: `tsdf = 1.0f`, `weight = 0.0f`, neutral gray
color. `+1.0f` means "unobserved / free space, not yet truncated", and
emptiness is decided by `weight`, never by the `tsdf` literal.

Current CPU behavior:

- `include/tsdf/TSDFVolume.h` `struct Voxel` defaults to
  `tsdf = EMPTY_TSDF`, `weight = EMPTY_WEIGHT`, `r = g = b = EMPTY_COLOR`
  (`EMPTY_COLOR = 128.0f / 255.0f`, the float sRGB widening of the neutral byte
  `128`) — canonical, and a `static_assert` in the same header makes
  `Voxel{}` equal to the named constants at compile time.
- `src/tsdf/TSDFVolume.cpp` `unlocked_reset()` fills
  `Voxel{EMPTY_TSDF, EMPTY_WEIGHT, EMPTY_COLOR, EMPTY_COLOR, EMPTY_COLOR}` and
  zeroes `integrated_frames_`, so `reset()` restores the canonical empty state
  exactly. Fixed by the TSDF-lifecycle todo: the fill used to be
  `Voxel{0.0f, 0.0f, 128, 128, 128}`, which put every reset voxel on the zero
  isosurface so an unobserved voxel could be meshed as surface. The CPU
  contract is `tests/tsdf_reset_contract.cpp`.
- `include/tsdf/TSDFVolume.h` now names the CPU sentinel pair
  `EMPTY_TSDF = 1.0f` / `EMPTY_WEIGHT = 0.0f` (plus
  `EMPTY_COLOR = 128.0f / 255.0f`, which publishes the neutral byte `128`), and
  `unlocked_reset()`, `getTSDF()` and the `raycast()` empty-space markers use
  them. Seven lines of `src/tsdf/TSDFVolume.cpp` still carry the raw literal
  `1.0f` (eight textual occurrences — the diagnostic line carries two), and none
  of them is an empty-voxel marker. Site by site: the ray projection `z = 1`
  twice, `ray_cam(..., 1.0f)` at `:283` (integration) and `:425` (raycast); the
  `[-1,1]` truncation clamp once, `tsdf_new = std::min(1.0f, sdf / trunc)` at
  `:321`; the clamping diagnostic once, `tsdf_new >= 1.0f || tsdf_new <= -1.0f`
  at `:324`; the per-frame weight increment `w_new = 1.0f` at `:359`; the color
  fusion denominator once, `denom = w_old + 1.0f + 1e-6f` at `:383` — one
  literal and not three, because the three color channels divide by that single
  `denom`, and the TSDF blend at `:362` reaches the same value through the named
  `w_new` instead of a second literal;
  and the point-cloud `weight > 1.0f` gate at `:631`. The backends still carry no such constant, and CPU
  meshing applies its own named `kWeightEpsilon = 0.001f` rather than
  `EMPTY_WEIGHT`: one shared epsilon remains open (`tsdf:T2`, `tsdf:T5`,
  `tsdf:T11`, `tsdf:T24`).
- Emptiness thresholds are inconsistent across layers: `weight > 0.0f` in the
  CPU trilinear sampler, `kWeightEpsilon = 0.001f` in CPU meshing. Canonical:
  one named epsilon.
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
- `include/tsdf/TSDFVolume.h` `voxelAt()` documents a **precondition**, not a
  service it performs: `x, y, z` are already inside `[0, resolution)` and the
  caller consults `inBounds()` or its own loop bounds first (header comment
  `:113-116`). Both overloads in `src/tsdf/TSDFVolume.cpp` enforce that
  precondition with `assert(inBounds(x, y, z))` immediately before the indexed
  read — `voxelAt() const` at `:552`, `voxelAt()` at `:557` — so it is
  assert-checked in a debug build and deliberately bare in a release build,
  where the accessor sits in the meshing hot path. **CPU-fixed**:
  the header used to advertise an automatic check that the body did not
  implement; what it advertises now is exactly what the body does, and the CPU
  gate keeps the removed phrasing out of `TSDFVolume.*`
  (`scripts/test-cpu-big-fix.sh:279`, the dead-symbol guard labelled
  `tsdf:T24 false bounds-checked doc claim`). Violating the
  precondition in a release build is a caller bug, and nothing here promises it
  is caught.
- `include/tsdf/VoxelGPU.h` `VoxelGPU` is 32 bytes with `float padding[3]`
  (12 permanently unused bytes per voxel) and no `static_assert` on its size;
  host `Voxel` is 20 bytes — five `float`s (`tsdf, weight, r, g, b`) with
  `alignof == 4`, color stored as float sRGB — which is the same fact the color
  section states below as "20 B vs 32 B". The layout translation happens
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
`tsdf:T8`. The snapshot lifecycle that kept the GUI slider stale for integration until
restart was Todo 24's scope rather than a Todo 15 defect, and Todo 24 closed it on CPU
(per-frame `hyperparamsSnapshot()` inside `integrationLoop()` at
`PipelineController.cpp:1075`, locked by
`tests/pipeline_hyperparams_contract.cpp`); the backend instance stays deferred as
`cross-backend:A28`.

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
reset could also *fabricate* a crossing for an interior start. The CPU rule is
**CPU-fixed by big-fix Todo 15**: `src/tsdf/TSDFVolume.cpp:446` accepts either
direction with `(f_prev > 0.0f && f_cur <= 0.0f) || (f_prev < 0.0f && f_cur >= 0.0f)`.
CUDA and HIP still test only `prev_tsdf > 0.0f && tsdf <= 0.0f`, so backend
parity is deferred as `tsdf:T26` (not compiled, not runtime-tested here).

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
winding rule, and the rule is derived, not chosen.** In right-handed world
space (`e_x × e_y = e_z`), the divergence theorem makes the signed volume
`V_signed = (1/6) Σ a·(b×c)` of a closed surface positive exactly when every
face normal points outward; for a signed-distance sphere the outward normal is
the direction of increasing tsdf, away from the known center. So the outward
convention is a theorem about the reference sphere construction, never a vote
of table rows. `tests/marching_cubes_winding_contract.cpp` (big-fix Todo 18)
establishes the rule on a hand-built unit cube before touching any product
mesh, verifies the sphere fixture's sign structure voxel-by-voxel, and requires
`V_signed > 0` (within the derived inscribed-mesh band), a strict per-face
radial test, and a per-cube derivation that evaluates BOTH windings of
independently interpolated crossing vertices. Backends do not get their own
winding opinion: CUDA and HIP winding must later be derived from the same
fixture, and any backend that disagrees is the one that is wrong.

Derived CPU convention (Todo 18; derivation record in
`.omo/evidence/big-fix/winding-derived-sign.txt`): with the shared table's edge
incidence, the corner-sign rule "`tsdf < 0` = inside", and crossings
interpolated from the canonical lower endpoint, the outward order of every
emitted triangle is the **reverse table order `(2, 1, 0)`**; the forward order
`(0, 1, 2)` is mathematically inward under the same fixture. `tri_table` is
consulted only for *which edges* form a triangle, never for their orientation —
no expected table order is hardcoded anywhere in the contract. Current CPU
behavior therefore stands unchanged: `src/meshing/MarchingCubes.cpp` emits
`for (int i = 2; i >= 0; --i)`, now labeled the derived convention (the old
"fix front-face culling" comment was an ad hoc description of the same order)
and locked by the new contract plus the pre-existing sphere outward test, both
of which reject a flip to `0, 1, 2` (mutation witness: signed volume flips to
exactly `-0.1127967361`, 8588/8588 faces inward). HIP emits the same literal
`2, 1, 0` order; CUDA emits forward `0, 1, 2` with no explanatory comment, so
the CUDA file currently reads as the unpatched original and is inward under
the derived fixture (`meshing:B5` / `cross-backend:A1`; deferred, not compiled,
not runtime-tested on this lane — matching the order is that todo's job, not
this one's).

Canonical CPU edge table — **CPU-fixed by big-fix Todo 7** (commit `f09bfc2`
"fix(meshing): repair CPU marching cubes edge table"): the shared CPU table
`include/meshing/MarchingCubesTables.h` carries `edge_table[213] = 0x83f`,
`edge_table[214] = 0xb35`, `edge_table[215] = 0xa3c` on the `edge_table` data row
that holds indices 208-215 (`:59`), and that repair is recorded in the header's
own comment block (`:8-10`), not as an inline marker on the data row. These are
the canonical crossing-edge values, and with them the whole CPU table satisfies
the mirror identity `edge_table[i] == edge_table[255-i]` for all 256 entries.
That identity is what identified the old triplet as corruption rather than a
variant: the mirror of indices 208-215 is the row holding indices 40-47
(`MarchingCubesTables.h:38`), and it already carried the correct reversed
sequence `0xa3c / 0xb35 / 0x83f` in every copy.

The corruption is therefore confined to the two **deferred** copies, which still
carry `0x835 / 0xb3f / 0xa36`: `src/meshing/MarchingCubes_cuda.cu:62` and
`src/meshing/MarchingCubes_hip.hip:58`. Each of them breaks the mirror identity
at exactly indices 40/41/42 and 213/214/215, which is the same test the CPU
table now passes. Timeline: Todo 4 (`7151f5f`) was the value-preserving move
into the shared header; Todo 6 added the analytic table/sphere contracts that
pinned the defect and go green against exactly these three repaired CPU values;
Todo 7 repaired the shared CPU header only. Locked by
`tests/marching_cubes_table_contract.cpp` — the Todo 6 first-principles oracle
that was RED before the repair and is GREEN against exactly these three values —
and by `tests/marching_cubes_sphere_contract.cpp`, which pins
`{213, 0x83f} {214, 0xb35} {215, 0xa3c}` at `:97`. Backend propagation stays deferred
as `cross-backend:A7` — not compiled, not runtime-tested in this run.

There is no `// fix: 0x835→0xb35 variant` comment anywhere in the shared header,
and there never was one to critique: the only lines in
`include/meshing/MarchingCubesTables.h` matching `fix` are `:4` (the Todo 4
provenance note) and `:8` (the first line of the Todo 7 repair block). The
canonical target of `[213]` is `0x83f`, and that is what the CPU table now
stores.

Canonical mesh vertex validity (CPU, established by big-fix todo 16):

- An unobserved voxel, or one whose `tsdf` is not finite, samples as
  `EMPTY_TSDF` (`+1.0f`) and carries no support bit. The literal stored in an
  unobserved voxel is never read as material, so a stale negative left behind by
  an earlier integration cannot mint geometry.
- A cube is evaluated from its crossing edges. It is never dropped because a
  corner that takes part in no crossing edge is unobserved.
- A crossing edge is emitted only when **both** of its endpoints are observed
  and finite.
- A triangle is emitted only when all three of its vertices survived. A vertex
  whose interpolated position is not finite, or whose blended normal is not
  finite or collapses below `1e-6`, is refused outright and never substituted.
- A corner normal is a central difference where both neighbours support one and
  a one-sided difference on the axis that has no two-sided support, so a vertex
  at the volume border still gets a real normal. The central form keeps its
  `0.5` factor so a border axis is never weighted twice an interior axis of the
  same physical slope.
- Repeated extraction of one volume is byte-identical.

Current CPU behavior: **fixed** in `src/meshing/MarchingCubes.cpp`
(`MarchingCubes::voxelNormal` is the one normal primitive, and the cube loop
samples each corner once and caches one normal per corner). Locked by
`tests/marching_cubes_frontier_contract.cpp`. The replaced path gated a whole
cube on `vox.weight <= 0.001f` and nothing else, so one unobserved corner
deleted an otherwise fully supported cube and the mesh stopped being closed;
`computeNormal()` mapped a degenerate gradient to a fabricated `(0, 0, 1)` and
`(n0 + t * (n1 - n0)).normalized()` divided by zero for a cancelled endpoint
pair, so a NaN normal and an invented normal both reached `MeshData` and were
indistinguishable from real ones. Corner normals are now evaluated once per cube
corner instead of once per crossing-edge endpoint - up to 24 evaluations for the
same 8 corners before. CUDA and HIP are untouched by this todo; their
equivalents stay deferred and are itemised in
`docs/CUDA_HIP_DEFERRED_CHANGES.md` (`meshing:B1`, `meshing:B2`, `meshing:B3`,
`meshing:B4`, `meshing:C2`, `meshing:C3`, `meshing:C7`, `meshing:D8`,
`cross-backend:A14`).

Canonical welding: exact edge identity. The CPU weld in
`src/meshing/MarchingCubes.cpp` keys a hash map on the canonical identity of a
physical crossing edge - the LOWER endpoint voxel coordinate plus the single axis
the endpoints differ on - so every cube (and every OpenMP slice boundary) that
reaches the same edge resolves it to exactly one global vertex. This replaces the
pre-Todo-17 map keyed on a quantized float position (`QuantizedHash`, divisor
`voxel_size * 0.01f`), which was not exact edge identity: it silently merged
distinct vertices that quantized together and split FP-variant shared edges.
**Fixed by big-fix Todo 17.** The HIP weld still keys a quantized position and
adds a distance-validated escape branch (`merge_threshold = voxel_size * 0.5f`)
that is unreachable for finite input - two positions that collide under a
`0.01·vs` quantizer differ by less than `0.0087·vs`, always under the threshold -
and if it ever did fire it would overwrite the map entry instead of chaining;
CUDA/HIP welding stays deferred (`meshing:D3` / `cross-backend:A29`).

Current CPU behavior: **fixed** in `src/meshing/MarchingCubes.cpp`
(`MarchingCubes::crossingFromLower` is the one interpolation primitive). For each
canonical edge the crossing is parameterised once from the LOWER endpoint, and a
single parameter `t` drives position, blended normal and RGB color together, so
adjacent cubes agree byte-for-byte and the weld never disagrees about a shared
vertex's payload. The merge runs serially in ascending-slice, local-triangle order
and welds by `EdgeKey` (first insertion wins - byte-identical because every cube
derives the same payload for a key). A CPU triangle budget
(`MarchingCubes::setMaxTriangles`) stops at a full-triangle boundary, sets
`MeshData::truncated` and never emits a partial row. The progress callback fires
only during this serial merge, from the calling thread, so it can never overlap and
its sequence is a pure function of the resolution; a resolution below 2 returns an
empty mesh instead of dividing a `(resolution - 2)` progress span. Locked by
`tests/mesh_welding_contract.cpp`, `tests/mesh_color_invariant_contract.cpp` and
`tests/mesh_truncation_progress_contract.cpp`. The CPU reverse-winding emission
(`2, 1, 0`) is now the derived, locked winding convention (winding section
above; Todo 18), no longer an open item; the float→uint8 color quantization is
now the single shared policy described below (Todo 19).

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
- `include/tsdf/TSDFVolume.h::Voxel` is now
  `{ float tsdf, weight, r, g, b }` — five floats, the color pair triply widened
  to match the canonical domain. `include/tsdf/VoxelGPU.h` already carried float
  `r/g/b` plus `float padding[3]` under `alignas(32)`, so the two structs agree
  on the *type* of every color field but not on size or alignment (20 B vs 32 B);
  no GPU-side semantics are claimed here, and the migration stays in the deferred
  backend dossier.
- `src/export/GLBExporter.cpp` emits `COLOR_0` as a `VEC4` `float` accessor whose
  RGB components are the checked sRGB→linear decode
  (`include/export/ColorConversion.h`, `c <= 0.04045 ? c/12.92 :
  ((c+0.055)/1.055)^2.4`, computed in double) of the uint8 sRGB `MeshData`
  colors, with alpha `1.0` and no alpha conversion. A component that is
  non-finite or outside `[0, 1]` rejects the whole export **before** the file is
  opened, so no partial triple and no half-written file can escape. That checked
  rejection is stricter than the extraction clamp below and stays so on purpose:
  an exporter input is already a validated uint8 byte, so a value out of `[0, 1]`
  there can only be a caller bug. Locked by
  `tests/glb_linear_color_contract.cpp`, which exports real temporary `.glb`
  files and reads the accessor back.
- `src/export/GLBExporter.cpp` converts Kinect Y-down to glTF Y-up by negating
  Y and flipping Z; the README documents `Y → -Y` and Unity's importer applying
  the handedness flip. Any winding-rule change must be re-checked against this
  transform, because the export flips handedness twice and can mask a winding
  bug in one of the two stages.
- `src/export/PLYExporter.cpp` emits **one** header for both formats:
  `headerText()` (`:91-105`) builds every element/property line, and the binary
  writer (`writeBinary`, calling it at `:209`) and the ASCII writer
  (`writeASCII`, calling it at `:224`) both append it with only the format token
  differing (`binary_little_endian` vs `ascii`), over the same `writeRecords()`
  body. So both publish `property uchar red/green/blue` and
  `property list uchar int vertex_indices` for the same logical mesh. **CPU-fixed
  by big-fix Todo 30** (commit `7b36d5b` "fix(export): validate and robustify CPU
  PLY writer") — one schema, one spelling, both writers. The spellings
  this passage used to blame on the two writers (`property uint8 …` in binary,
  `property list uchar uint …` in ASCII) do not occur anywhere in the file. The
  one residual nuance is signedness, not drift: the shared header declares the
  index type as `int` while the binary sink writes each index with
  `appendU32LE()` — same 4-byte width, identical for both formats — and the
  per-face count byte is the constant `3` (`:154`). The only other asymmetry is
  a guard, not a schema: `writeBinary()` refuses a big-endian host before
  opening the file (`:198-202`), because it emits a little-endian payload; ASCII
  needs no such check.
 - `src/meshing/MarchingCubes.cpp` interpolates the edge color in float sRGB with
   the same parameter that produced the vertex position and turns it into the
   canonical byte through `kfusion::utils::srgbFloatToUint8`
   (`include/utils/ColorMath.h`) - one clamp plus round-to-nearest, so a finite
   overshoot of the blend saturates to the endpoint byte instead of wrapping.
   Only a NON-FINITE endpoint or blend invalidates the crossing edge outright:
   it is treated as not crossed and the triangle rows that depend on it are
   dropped, exactly like an unusable normal, rather than emitting an
   undefined-behavior byte. Positions, normals, the shared interpolation
   parameter, welding and winding are untouched by either path (locked: a
   uniform out-of-range color produces the identical vertex and triangle counts).
   The same single helper quantizes the raycast color output and the global point
   cloud, so there is exactly one answer to "what byte represents this float".
   HIP deliberately keeps the old "simple cast without clamping" and CUDA keeps
   `fminf(255.0f, fmaxf(0.0f, ...))`; both still owe the migration to this policy
   (deferred backend rows `meshing:C8` / `cross-backend:A26`).
 - `include/utils/ColorMath.h::srgbFloatToUint8` is that one CPU policy and it is
   a **clamp** policy: every FINITE value is representable, a value outside
   `[0, 1]` is clamped before it is scaled (clamping first means even `FLT_MAX`
   cannot overflow the multiply) and then `std::lround`-ed, so `-0.001`, `-1.0`
   and `lowest()` are byte `0` while `1.001`, `2.0` and `FLT_MAX` are byte `255`.
   NaN and `+/-Inf` are the only rejected inputs - they return false and leave
   the output byte untouched, because `static_cast<uint8_t>` of a non-finite
   float is undefined behavior and saturating a NaN would hide an upstream bug;
   a caller therefore withdraws the raycast pixel, skips the point-cloud voxel,
   or drops the Marching Cubes edge, and never publishes a half-written triple.
   The GLB boundary is deliberately stricter (see the `ColorConversion.h` bullet
   above): an export input is already a validated byte, so range there is a bug.
   `include/tsdf/TSDFVolume.h::EMPTY_COLOR` is `128.0f / 255.0f` so the neutral
   byte 128 stays a representable color in the float domain (a raw `120` in a
   voxel color field is an out-of-range sRGB component, not a color, and the
   boundary publishes 255 for it rather than wrapping). CPU integration widens
   every incoming candidate with `kfusion::utils::srgbUint8ToFloat` (one division
   by 255, no gamma: the volume domain is sRGB-encoded) and accumulates in float,
   so repeated fusion converges to the true mean instead of stalling in a
   per-update uint8 rounding dead zone. Locked by
   `tests/color_convergence_contract.cpp`.

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

Canonical border mode: reflect, from one helper. The CPU carried two byte-identical
copies of `reflectCoord()` — one in `SuperResolution.cpp`, one in
`SignalConditioner_omp.cpp` — and both now call
`kfusion::sensor::cpuReflectCoord()` (`include/sensor/BorderMode.h`), defined on
`-extent <= coord <= 2 * extent - 1` and mapping into `[0, extent - 1]`. The two backends
do not reflect identically. `SuperResolution_cuda.cu` reflects only through its
radius-1 RCAS helper (`reflectCoordCAS`, used by `rcasGetPixel`) and has no EASU
resample at all; `SuperResolution_hip.hip` reflects on both its EASU resample
(`reflectCoordEASU`) and its radius-1 CAS (`reflectCoordCAS`). The only clamping
sites are the SignalConditioner helpers `rcasGetPixelCUDA` / `rcasGetPixelHIP`
(`max(0, min(x, w - 1))`), reached solely from `applyCASKernel`. Every offset
passed to those two is ±1, and at radius 1 the modes are the same function —
`reflect(-1) = 0 = clamp(-1)` and `reflect(extent) = extent - 1 = clamp(extent)`,
checked over extents 1..1024 by `tests/cas_border_contract.cpp`. So the claim that
the clamp "doubles the edge value into the RCAS cross-stencil, over-sharpening a
visible 1-pixel frame" does not hold: only a stencil reaching ±2 separates the
modes, and that is the 4×4 Catmull-Rom resample, whose destination column 0 reads
source taps `{-2, -1, 0, 1}` with non-zero weight on the `-2` tap — and that
resample exists only as the CPU `applyEASU_CPU` and the HIP `easuKernel`, both of
which already reflect; CUDA has no EASU pass, so no backend clamps a ±2 stencil.
What survives of `sensor:S-05` /
`cross-backend:A20` is a latent text difference, not an observed pixel delta; it
stays open until a backend is compiled, and the comment calling those two kernels
"the same algorithm" is accurate rather than false.

Canonical guidance luma: the ICP guidance image is the luma of the *sharpened*
frame. `buildSuperResolutionGuidance()` copies the input into `sr_rgb_`, runs
`sr::applyCAS(sr_rgb_, FRAME_W, FRAME_H, 0.5f)` in place, then maps each post-CAS
pixel with Rec.601 weights and a `/255`:
`guidance_luma_[i] = (0.299*R + 0.587*G + 0.114*B) / 255`, in float32, which puts
the buffer in `[0, 1]` — a bound that is actually attained, since an exhaustive
float32 scan over all 2^24 byte triples gives exactly `0.0f` for flat black and
exactly `1.0f` for flat white. Reading the unsharpened input instead of
`sr_rgb_`, Rec.709 weights, swapped R/B weights, and a dropped `/255` are four
different wrong images; all four are pinned by
`tests/cas_border_contract.cpp`.

Canonical sharpness clamp: a non-finite sharpness must map to a defined finite
sharpen, never to NaN. CPU `applyCAS_CPU()` now owns that decision with one
explicit `std::isfinite(sharpness)` test: `NaN`, `+Inf` and `-Inf` all map to
`0.0f`, the softest valid sharpen, before the
`peak = -1 / ((1-t)*8 + t*5)` mapping, so no non-finite value can reach `peak`
and the byte store can never evaluate `static_cast<uint8_t>(NaN * 255.0f)` — the
undefined behavior the old `std::clamp(sharpness, 0.0f, 1.0f)` allowed, since
`std::clamp` returns `val` unchanged when both comparisons are false
(`NaN → NaN → peak = NaN`). On this x86 build that UB resolved to an all-zero
image, i.e. a black frame, which is exactly the signature the contract forbids.
Finite out-of-range values keep the documented clamp (`-1.0 → 0.0`, `2.0 → 1.0`).
Test-locked by `tests/sr_upscaled_contract.cpp` against the REAL CPU pass, with
the `0.5f` and `1.0f` runs pinned as the non-vacuity witness so "identical to
`0.0f`" is a decision rather than a pass that ignores its input. The GPU form
`std::max(0.0f, std::min(sharpness, 1.0f))` maps `NaN → 0.0` only as an accident
of `std::max(0.0f, NaN)` returning its first argument; it is not compiled and not
runtime-tested here, and that backend asymmetry stays deferred as
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
invalidates the upscaled-RGB availability contract at entry and only then returns
early on a frame-size mismatch (`@S.omp` `process()`), so a rejected frame can no
longer leave the previous frame's bytes readable as if they were fresh (big-fix
Todo 22, `tests/sr_upscaled_contract.cpp`); `processCuda` returns `true`
unconditionally and throws on allocation failure instead of returning `false`
(deferred `cross-backend:A22`), and `applyEASU_GPU` is `void`, zero-filling
`dst` before the allocation so a failed allocation yields a black frame that is
consumed as valid (deferred `sensor:S-06` / `cross-backend:A24`).

Canonical temporal EMA: single-buffer output. `applyDepthEma()` must not read
the buffer it is writing, or the reset decision is made on partially updated
neighbours and the result depends on schedule order. On CPU it now snapshots the
incoming frame into `depth_src_` and writes `depth`, so the pass is a pure
function of (input frame, per-pixel history, band) and bit-identical at 1, 2 and
4 threads — locked by `tests/depth_ema_determinism_contract.cpp`, whose fixture
is asymmetric between a pixel's own history and its neighbours' precisely so that
a destination read (which resets the centre and emits its raw input instead of
the blended code) cannot pass. The same in-place shape still exists in both GPU
EMA kernels; deferred as `sensor:S-19`. Every other depth pass already ping-pongs
between the in/out buffers.

Canonical upscaled-RGB availability: CPU-only and fail-closed. The buffer is
gated by `SignalConditioner::srUpscaledAvailable()` / `srUpscaledFrameId()` /
`srUpscaledAvailableForFrame()`, mirrored on the `Preprocessor` interface where
every non-CPU backend keeps the base `false`. Availability is geometry-aware, not
one boolean: it requires `2 <= getSrScale() <= 4`, a buffer of exactly
`FRAME_W * scale * FRAME_H * scale * 3` bytes, and a publish by `move` only after
the CPU stage has produced that whole frame, whose `raw.frame_id` it records. It
is invalidated at the top of `process()` and by `reset()`, so scale 1, a scale
outside `[2, 4]`, a frame rejected on size and a just-reset buffer all report
unavailable — including while the bytes still fit a valid 2× shape, which is why
the geometry test and the flag are both load-bearing. The getter alone is never a
validity signal: the constructor preallocates `FRAME_W * FRAME_H * 3` zero bytes,
so an ungated read textures TSDF with black. Scale 1 no longer writes an
original-resolution image into this buffer either; that copy is what let a
consumer sized `FRAME_W * scale × FRAME_H * scale` read out of bounds. Locked by
`tests/sr_upscaled_contract.cpp`, which drives the real `process()` path and the
public `applyEASU_CPU` / `applyCAS_CPU` passes.

Canonical upscaled-RGB consumer: either consumed or deleted — still open. The
only consumer, the commented pair at `src/app/PipelineController.cpp:782-783` in
`trackingLoop()`, stays disabled (its note now says a future consumer must check
the availability contract first), so the product still produces the buffer every
frame and still textures from raw RGB. Enabling it is future scope, not this
todo. Deferred as `cross-backend:C8`. On CUDA the buffer is never even written,
because `SuperResolution_cuda.cu` implements no EASU at all and `applyEASU`
dispatches to CPU there. The GPU upscaled-RGB hazard — the dead path, the HIP
`applyEASU_GPU` zero-fill that yields a black frame consumed as valid, and the
RCAS sharpness asymmetry — stays deferred as `cross-backend:A15` alongside
`sensor:S-06` / `cross-backend:A24` / `cross-backend:B5`; none of it was compiled,
linked or runtime-tested in this CPU-only run.

Canonical super-resolution scale: `setSrScale()` must have an effect.
`sr_scale_` is never read in either GPU conditioner, so the GUI slider is a
silent no-op on GPU. Deferred as `cross-backend:B2`.

## Sensor pairing and latest-frame semantics

Canonical timestamp unit: libfreenect delivers `uint32_t` microseconds, and the
sensor converts once — `timestamp / 1000.0` — so `RawFrame::timestamp_depth` and
`RawFrame::timestamp_rgb` are milliseconds and nothing downstream re-scales them.

Canonical temporal pairing, **fixed by big-fix Todo 23** (`sensor:S-08`): a depth
packet and an RGB packet are paired iff both timestamps are finite AND

```cpp
std::fabs(timestamp_depth - timestamp_rgb) < kMaxFrameSyncDeltaMs   // 50.0 ms
```

The comparison is strict, so an **exactly-50 ms delta is stale** and is rejected.
`kMaxFrameSyncDeltaMs` lives in `include/sensor/KinectSensor.h`; the threshold is
not a magic number at the call site. uint32 microsecond wraparound (about every
71.6 minutes of device uptime) is outside this policy: the delta is taken on the
raw converted values, so a wrapped sample reads as stale and is dropped rather
than mispaired.

Canonical stale-pair policy (`src/sensor/KinectSensor.cpp:218-254`): a rejected
pair publishes nothing, **consumes no frame id**, and copies no RGB. The
older-timestamped side is recycled to the pool and the newer side is retained for
the next sample on the stale stream, so one lagging stream degrades to a delay
instead of a permanent stall. Depth-led and RGB-led arrival orders run through
the same helper — the policy is symmetric, not two hand-written copies.

Canonical latest-frame semantics, **fixed by Todo 23** (`sensor:S-09`):
`getLatestFrame()` is a newest-wins consumer, so the ready side is a single slot
(`PoolState::ready_frame`), not a queue. `publishLatest()` installs the new frame
and lets the frame it displaced recycle **outside** `pool_state_->mutex` — the
pooled deleter (`:274-291`) locks that same mutex, so destroying a ready or
pending frame while holding it self-deadlocks. Three accepted pairs consumed with
no reader leave exactly one frame, the newest; `getLatestFrame()` moves the slot
out, so the next call returns `nullptr` rather than a queued older frame. The old
`ready_queue` handed out `front()`, i.e. the oldest frame ever queued — a
backpressured pipeline was reconstructing a frame it had already superseded.

Canonical lock discipline, **fixed by Todo 23** (`sensor:S-15`,
`src/sensor/KinectSensor.cpp:141-216`): the pairing lock covers the pending
slots, the staleness decision, the RGB copy, the frame-id assignment and the copy
of `frame_callback_`. The product callback itself is invoked **after**
`sync_mutex_` is released, and so is the throttled synchronization log. A callback
that observes `sync_mutex_` held can block the sibling capture stream.
`tests/kinect_pairing_contract.cpp` asserts `syncMutexIsFreeForTests()` from
inside every callback, which is the mechanical form of this rule.

Temporal pairing is **not** spatial registration. Meeting the 50 ms window makes
the two packets contemporaneous in time; it does not put a depth pixel and an RGB
pixel on the same ray. The combined frame is depth-owned with the RGB blitted in
full-frame (`sensor:S-07`), and this run records the depth-to-color registration
status as

```text
depth_to_color_registration: SKIP: no depth registration source
```

The host does have libfreenect and `<libfreenect/libfreenect_registration.h>`, and
that header does export `freenect_camera_to_world`, `freenect_copy_registration`
and `freenect_destroy_registration` — but those go **camera → world**, i.e. depth
intrinsics plus a mechanical baseline. Nothing in the installed package supplies a
depth-to-**color** image transform or a registration source to build one, so no
warp is claimed and none is applied. Evidence:
`.omo/evidence/big-fix/kinect-depth-color-alignment.txt`.

## Pipeline parameter application, callbacks and thread count

Canonical rule: one owner per live parameter, one snapshot per processed frame,
and no subscriber ever runs while a controller lock is held.

Current behavior in `src/app/PipelineController.cpp`, **fixed by big-fix Todo 24**
(CPU only — the CUDA / HIP worker-loop, GPU dispatch, TSDF resize, callback and
thread-count execution paths keep their own shape and stay deferred as
`cross-backend:A28`):

- **Owner snapshot per frame.** `hyperparamsSnapshot()` is the only read path to
  `hyperparams_`, and both workers call it exactly once per processed frame,
  after that frame is popped and before anything consumes it (`trackingLoop()`
  `:754`, `integrationLoop()` `:1075`). No preprocess / `track()` / `integrate()` call runs under
  `hyper_mutex_`, which is a leaf guard around the copy alone.
- **Application is serialized by lifecycle.** `setHyperparams()` (`:271`) runs
  under `control_mutex_`, the same lock `start()` / `stop()` / `reset()` take, so
  it can never interleave with a start or a shutdown. It then takes
  `hyper_mutex_`, `tracker_mutex_`, `tsdf_mutex_` (exclusive, because
  `setParams()` reallocates and clears voxels against the integration, raycast
  and meshing readers) and `preprocessor_mutex_` **one at a time, never nested**,
  so no pair of component locks can invert in either direction. A band change
  clears the volume, which is why the integrated-frame count restarts from zero.
- **Each component lock owns only its own work.** `tracker_mutex_` spans
  `params()` / `setParams()` and the whole `track()` solve — and in
  relocalization the recovery-parameter swap around the hypothesis tracks,
  restored before the lock is released — with `gpu_mutex_` taken *before*
  `tracker_mutex_` on the GPU branches. `preprocessor_mutex_` covers `process()`,
  `setSrScale()`, `resetTemporalState()` and `reset()`, which mutate the temporal
  EMA and the upscaled buffer. `tsdf_mutex_` keeps its existing shared-read /
  exclusive-write split. Queue, pose, metrics and callback locks are never held
  together with a component lock.
- **Callbacks are copied, never shared.** The controller has exactly two
  callback setters, `setFrameReadyCallback()` (`:312`) and
  `setMeshReadyCallback()` (`:317`), and both assign under
  `callback_mutex_`; `frameReadyCallbackCopy()` (`:322`), `meshReadyCallbackCopy()`
  (`:327`) and `hasFrameReadyCallback()` (`:332`) read it under the same lock.
  Metrics are not a callback: the GUI pulls them through `metricsSnapshot()`
  (`:254`), and no `setMetricsCallback()` exists on the controller. Every call
  site — `dispatchUiFrame()` (`:595`, copy taken at `:599`), the meshing-loop
  publish inside `meshingLoop()` (`:1325`, copy taken at `:1439`) and `stop()`'s
  final full-model view (`:469`) — takes the copy first and invokes **that copy** outside
  `callback_mutex_`, because a subscriber is arbitrary UI code that may
  legitimately re-enter the controller. `dispatchUiFrame()` captures only the
  copy and the shared frame in the queued lambda, never `this`, so a later setter
  cannot change what an in-flight dispatch delivers.
- **Thread counts move only at safe points.** `setNumThreads()` (`:355`)
  publishes the request first; while running it parks the value
  (`pending_num_threads_`, then `threads_pending_` as the release/acquire edge)
  and `applyPendingThreadCount()` applies it in `trackingLoop()` after the pop
  and before the frame is processed (called at `:749`, definition `:348`), so the tracker is never resized
  mid-`track()`. While stopped it re-checks `running_` under `control_mutex_`
  and applies immediately, so a `start()` that won the race cannot have a worker
  already tracking with the old count.
- **Terminal state is published last.** `stop()` stores `Stopped` only after all
  three workers have joined (`stop()` at `:377`, joins at `:433-438`, the store
  at `:443`), because `trackingLoop()` stores
  `Running` / `TrackingLost` as it finishes a frame; storing it before the join
  let a mid-frame worker overwrite the terminal state after `stop()` had already
  returned.

Locked by `tests/pipeline_hyperparams_contract.cpp` (seam-driven, CPU label, no
device).

## Pipeline lifecycle state, queue backpressure and mesh requests

Canonical rule: a start begins with no motion model and no published mesh, every
queue is bounded with a named eviction policy that is counted, and a mesh result
can only be published if it was extracted against the current session's volume.

Current behavior in `src/app/PipelineController.cpp`, **fixed by big-fix Todo 25**
(CPU only — the CUDA / HIP worker loops keep their own shape and stay deferred as
`cross-backend:A28`):

- **A start clears the motion model.** `startInternal()` (`:64`) writes
  `last_pose_ = Identity` under `pose_mutex_` (the write at `:89`) and un-arms the mesh request block
  (`mesh_requests_.shutdown = false`, `:236`), so the first tracked frame is
  scored against identity and the world origin, never against the pose the
  previous session ended on. `reset()` (`:501`) does the same (`:542`) plus
  `frame_count_ = 0` and a fresh `metrics_`, both **inside** `metrics_mutex_`
  (`:552-556`, the zeroing at `:554`) — pre-Todo-25 those two writes were unsynchronized against the
  metrics reader. Neither `start()` nor `reset()` requires the other first.
- **Both queues retain the newest work.** `raw_queue_` capacity
  `kRawQueueCapacity = 3` (`include/app/PipelineController.h:216`),
  `integration_queue_` capacity `kIntegrationQueueCapacity = 3` (`:217`). When a
  push finds the queue full the **oldest** entry is displaced, so a slow consumer
  sheds stale input instead of growing memory without bound and instead of
  forcing the producer to block on a live capture callback. The displaced frame is
  destructed **outside** the queue lock — `displaced.reset()` in `onRawFrame()`
  at `src/app/PipelineController.cpp:673`, and the same shape in
  `enqueueForIntegration()` at `:722` —
  because a pooled `FrameData` deleter re-enters the sensor free-list.
- **Eviction is counted, not silent.** `dropped_frames`
  (in `onRawFrame()` at `:650`, incremented at `:689`), `dropped_integration_frames`
  (in `enqueueForIntegration()` at `:702`, incremented at `:725`) and `dropped_pre_model_frames` (frames
  discarded while waiting for the first model, in `trackingLoop()` at `:729`,
  incremented at `:823`) live inside
  `metrics_` and are written only under `metrics_mutex_`, which is a leaf: no
  queue lock is ever taken while it is held. They reach the UI through
  `metricsSnapshot()` and are zeroed by `reset()` with the rest of `metrics_`.
  Tracking and integration never block on a full queue, so the counters are the
  only visible sign of overload — which is why they are canonical surface.
- **Frames move by ownership, not by copy.** `enqueueForIntegration()` (`:702`)
  takes the `shared_ptr<FrameData>` by value and the tracking loop passes it with
  `std::move` (`:802`, `:1032`). Pre-Todo-25 the tracking loop copied the
  `shared_ptr` into the queue and kept a live local, so the pooled buffer could
  return to `acquireFreeData()` while still queued.
- **Mesh requests are versioned, and nobody waits for a result.** `requestMesh()`
  (`:1460`) increments `requested` under `mesh_requests_.mtx` and returns the
  version; `meshingLoop()` (`:1325`) waits on the condition variable, claims the
  newest requested version, extracts, and publishes with `(version, generation)`
  tags. The invariant is `served <= claimed <= requested`. A requester that needs
  the answer — PLY/GLB export (`exportPLY()` `:1540`, `exportGLB()` `:1547`, both
  delegating to `exportMesh()` `:1502`) — calls
  `awaitMeshVersion(version, 5s)` (`:1494`, invoked at `:1512`), a **bounded** wait released by
  `stop()`; the integration loop's cadence never waits at all.
- **Cadence is a clock, not a frame count.** `requestMeshIfCadenceDue()` (`:1484`)
  is called once per integrated frame (`integrationLoop()`, `:1321`) and requests at most once per
  `mesh_cadence_us_` (default 500000 µs, `include/app/PipelineController.h:319`),
  with the clock primed to the epoch at start so every session issues exactly one
  bootstrap request. The pre-Todo-25 `MESH_TRIGGER_FRAMES` counter made mesh rate a
  function of frame count, so it drifted with tracking quality.
- **A superseded result cannot be published.** `invalidateMeshState()` (`:1470`)
  bumps `generation` and drains `requested`/`claimed`; a worker that was already
  extracting captured its generation at claim time and drops the result instead of
  publishing it (`stale_drops`). `served` and `served_generation` are deliberately
  left alone, because they describe the mesh that *is* published: inventing a
  served version would wake a waiter for a mesh that was never produced, so a
  waiter on a drained version times out instead — the same visible outcome the
  pre-Todo-25 export path had when its flag was cleared. `stop()` (`:377`) sets
  `mesh_requests_.shutdown` (`:416`), which releases every waiter and every parked
  worker.

Locked by `tests/pipeline_state_contract.cpp` and
`tests/pipeline_mesh_cadence_contract.cpp` (seam-driven, CPU label, no device).

## Pipeline publication

Canonical rule: a published model frame implies a raycast-written buffer.

Current behavior in `src/app/PipelineController.cpp`: `emitCpuPreview()` is
defined under `#ifndef HIP_ENABLED`, and the `#elif defined(HIP_ENABLED)` raycast
block ends in an empty `else { // Fall through to CPU raycast }`. Under HIP with
`use_gpu_ == false` no raycast runs at all, yet `model_buffers_.swap()` and
`model_ready_.store(true)` still publish, so tracking consumes an all-zero or
stale model frame. Deferred as `pipeline:PC-02`.

Canonical depth band reaches the consumer: `min_depth` / `max_depth` must be
read inside the integration loop, per processed frame, not snapshotted before it.
CPU satisfies this since big-fix Todo 24: `integrationLoop()` reads
`hyperparamsSnapshot()` at `src/app/PipelineController.cpp:1075`, immediately
after popping a frame and before `integrate()` sees it, so a live slider change
takes effect on the next integrated frame. The pre-Todo-24 cache
(`d_min`/`d_max` stored once before the `while (running_.load())` loop) is gone.
CUDA / HIP keep the stale-read shape and remain deferred as
`cross-backend:A28`.

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
