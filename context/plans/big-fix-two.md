# big-fix-two - Work Plan

## Plan revision

- Date: 2026-09-22.
- Base: `de83ab5` on `idev`. Covers the 54 commits since `3841b9d` ("port to qt6"), most of which executed `context/plans/big-fix.md`.
- Inputs: 8 research lanes and 1 adversarial critic. Lane prefixes: HANG, PERF, GPU, TRACK, FUSE, SENSOR, REVIEW, SOTA, GAP (critic).
  - Scratch evidence lives under `/tmp/claude-1000/-home-iphands-prog-slop-kin-vendor-azu/35279940-71d1-49f4-a61c-479fd635138a/scratchpad/<LANE>/`. It is not in the repo, so re-derive any number before quoting it in a commit.
- Execution mode: **two lanes**.
  - The **CPU lane** is the reference oracle and must stay usable.
  - The **CUDA lane** is the production real-time lane. It is enabled by default only after its Phase 0 gates pass.
  - HIP is deleted.
- User request: "`./context/pans/bif-fix-two.md`". This was read as the typo it appears to be, and the file is written at `context/plans/big-fix-two.md`, next to `big-fix.md` (GAP-08).

## TL;DR (for humans)

**Why the app "captures a few frames then hangs"**
- Commit `c311c69` added a 50 ms RGB/depth pairing gate. It divides libfreenect timestamps by 1000, assuming microseconds.
- The Kinect v1 clock is actually 60 MHz, so the gate is really **0.83 ms**. Depth runs at 29.968 fps and RGB at 29.996 fps, so their phase slides by 1,868 ticks per frame. The gate is therefore open for about 54 frames out of every ~1,071.
- Replaying a real 80 s trace through the shipped code pairs **162/2400 frames, with a 34 s gap**. A real `PipelineController` running under libfakenect pairs **0 frames in 12 s** while the UI says "Running".
- The fix is one line: `/60000`. With it, pairing goes to 2396/2400 and 28 fps.

**The second wall behind it**
- Once pairing is fixed, the CPU lane integrates only **about 1.9 frames/s**, which knocks tracking out of the ICP basin.
- The tracking-lost state then freezes the preview, so it still looks like a hang.
- On an idle host, CPU integrate plus raycast cost **about 310 ms per frame**:
  - 150 ms comes from a serial sort-and-fold (`7438640`);
  - 156 ms comes from a uniform half-voxel raycast (`5233721`).
- Both are also accuracy bugs:
  - the SDF is sampled at one point and stored at another, giving a half-voxel bias. That causes up to 54 mm of drift with a still camera, and an ATE of 165 mm vs 3.6 mm once fixed;
  - the raycast accepts back faces, so 100% of hits are phantom surfaces when a surface is viewed from behind.

**The CUDA binary**
- `build/` is the CUDA binary the user most likely launches. It was never gated.
- On the default GPU (the RTX PRO 6000, with 458 MB free because vLLM is running) it hits OOM and **silently falls back to the slow CPU path**.
- On the RTX 4060 it runs at **29 integrated fps**, then **aborts on Stop** with an uncaught `cudaMalloc` exception, and the scan is lost.
- It also has an inside-out mesh, corrupt marching-cubes table rows, and unweighted ICP curvature.

**What this plan does**
1. **Phase 0 (ship together):** fix pairing, the dual-thread libusb stop, the off-thread GL, the stop-time crash, and the metrics-lock stall. Replace integration with voxel-projective (KinectFusion Alg. 1), make the raycast front-face only with space skipping, give ICP the correct reference pose, seed the model from frame 1, add real relocalization hypotheses, make CUDA device selection visible, and apply the CUDA one-liners. Add a fakenect end-to-end smoke test that would have caught `c311c69`.
2. **Phase 1:** a deterministic test and benchmark harness: synchronous seam, synthetic room with ground truth, fakenect/TUM/ICL-NUIM sources, ATE/RPE/Chamfer metrics, and perf and soak labels.
3. **Phase 2:** performance.
   - CPU: replace the 29 ms preprocessing chain with about 3 ms; sequential fusion worker; OpenMP budgets; allocation-free steady state; dirty-brick meshing.
   - CUDA: backend interface, stream-ordered pipeline, deterministic reductions, edge-owned GPU marching cubes; delete HIP.
4. **Phase 3:** accuracy. Registered depth with runtime intrinsics (today a 525 px focal length is applied to IR depth whose true focal length is about 576, a ~10% lateral scale error). Noise-aware truncation and weights, carving, one shared SDF sampler, mesh cleanup, noise-aware and degeneracy-aware ICP, photometric term, calibration, and an IMU gravity prior.
5. **Phase 4:** room scale. Voxel-block hashing on CPU and CUDA, streaming, a fern relocaliser, and session recording with offline pose-graph refinement.
6. **Phase 5:** UX.
7. **Phase 6:** cleanup, style and docs.

**Scale:** 70 todos. Every surviving finding is mapped in the traceability table.

## Measurement baseline (idle host, critic re-baseline; VERIFIED)

The other lanes measured at load average 45–68 on 32 cores. Only the idle numbers below are authoritative for prioritisation.

| Stage (CPU, 640×480, 256³ at 1 cm, synthetic room) | 32 threads | 16 threads | 8 threads |
|---|---|---|---|
| preprocess (SR on / off) | 28.9 / 23.4 ms | 34.5 / 28.3 | 65.6 / 52.1 |
| buildFramePyramid | 4.0 | 3.1 | 2.7 |
| ICP track (3 levels) | 8.6 (ok=1 but 9.3 mm error on 42.7 mm of motion) | 8.5 | 9.7 |
| **integrate** | **151.7** (flat across thread counts, so serial) | 153.9 | 149.4 |
| **raycast 640×480** | **156.2** | 157.0 | 280.0 |
| usageFraction (every frame, under `metrics_mutex_`) | 14.7 | 15.4 | 15.2 |
| MC full extract | 42.5 | 39.7 | 50.0 |

| End to end (real controller) | Result |
|---|---|
| CPU lane, HEAD, libfakenect dump | **0 paired frames in 12 s**, `state=Running` |
| CPU lane, `/60000` fix | 26 paired fps, **~1.9 integrated fps** |
| CUDA lane, `/60000` fix, RTX 4060 (`CUDA_VISIBLE_DEVICES=1`) | **29 integrated fps**, then **abort in `stop()`** (rc 134) |
| CUDA lane, default device (PRO 6000, 458 MB free) | OOM, silent CPU fallback, 1.8 integrated fps |
| PERF patched prototype (projective integrate, adaptive raycast, `ModelFrame::pose`, usage throttle, SR off) | 12–13 integrated fps under load, `tracking_ok` for the whole 12 s |

Prototype numbers:
- Voxel-projective integrate: **2.9–6.8 ms** (SOTA, PERF), versus 150 ms today.
- Adaptive, front-face raycast: **31–43 ms** (FUSE, PERF), versus 156 ms.
- CUDA at 192³ on the 4060: integrate 0.42 ms, raycast 1.1 ms, ICP 0.6–1.2 ms.

## Goals and measurable targets

Each target has a Phase gate, and a harness that measures it (Phase 1).

| # | Goal | Target | Justification | Measured by |
|---|---|---|---|---|
| G1 | No hang | 10 min fakenect loop replay plus 10 min synthetic soak, in both lanes: 0 intervals > 200 ms without a published frame while the sensor is live; paired ≥ 99.5% of depth frames; RSS growth < 50 MB; `stop()` < 300 ms; 0 aborts | c311c69 regression; GAP-02 crash | `pipeline_soak`, `smoke_fakenect` (T1.5, T0.4) |
| G2 | Pairing | ≥ 99.9% of depth frames published over a 4000-frame tick sweep including a wrap; every accepted pair has \|Δ\| ≤ 16.7 ms | half-period bound by construction | `kinect_pairing_realtrace_contract` (T0.2) |
| G3 | CPU throughput (reference lane) | Phase 0: ≥ 10 integrated Hz. Phase 2: ≥ 20 Hz, stretch 30 Hz. Measured at 16 threads, 256³ at 1 cm, synthetic moving room, idle host | stage sum 3 + 8 + 7 + 15 ms after the fixes | `pipeline_soak` CPU |
| G4 | CUDA throughput | 30 Hz sustained (sensor-limited) with ≥ 99% of frames integrated; fusion step (upload + ICP + integrate + raycast) p99 < 10 ms on the RTX 4060 and < 5 ms on the PRO 6000; sensor-to-integrated latency p95 < 40 ms | measured 29 fps even with sync copies; per-kernel costs about 1 ms | `pipeline_soak --backend cuda`, nsys |
| G5 | VRAM | Room at 1 cm < 1 GB; must run with only 1.4 GB free (4060 alongside llama-server) | GAP-01, GPU-03 host contention | GPU gate test |
| G6 | Static drift | 300 static synthetic frames: \|Δt\| < 1 mm and \|Δθ\| < 0.05° (today 20–54 mm) | HANG-12, FUSE-01 | `fusion_drift_contract` |
| G7 | Synthetic tracking | 150 noisy frames through the full chain: ATE RMSE < 5 mm (TRACK-02 prototype 3.6 mm); ideal model < 0.5 mm; 0 lost frames at 2 mm/frame and 1°/frame | TRACK-02, REVIEW-12 | `tracking_trajectory_synthetic` |
| G8 | Fusion accuracy | Plane: \|bias\| < 0.05·vs. Noisy sphere at 1 cm: mean \|err\| < 1.0 mm, RMS < 1.5 mm (prototype 0.60 / 1.18). Synthetic room with GT poses: Chamfer < 5 mm at 1 cm, completeness@1 cm > 95%, < 0.01% of vertices with error > 20 mm | FUSE-17 prototype | `fusion_accuracy_contract` |
| G9 | TUM RGB-D ATE (Kinect v1, registered) | Phase 3 gate: fr1/xyz < 3 cm, fr1/desk < 6 cm, fr2/xyz < 3 cm, fr3/long_office < 8 cm. Stretch after T3.7/T3.10: fr1/desk < 3 cm, fr3/office < 4 cm. fr1/room < 15 cm live, < 8 cm after refine | Frame-to-model KinFu-class systems report roughly 2–6 cm on fr1/desk / fr3/office; ElasticFusion 2.0 / 1.1 / 1.7 cm on fr1/desk, fr2/xyz, fr3/office (Whelan RSS 2015; literature recall, SUSPECTED exact values) | `eval_tum` + C++ ATE (Umeyama), cross-checked with `evo_ape` |
| G10 | ICL-NUIM living room kt0–kt3 | With GT poses: mesh→GT mean < 1 cm at 1 cm voxels. Tracked: < 2 cm | SOTA-12 | `eval_icl` |
| G11 | Real-device metric accuracy | Flat wall at 1–3 m: plane distance error ≤ 1% (≤ 0.5% after T3.11 calibration); 90° corner at 90 ± 1°; box edges within 1%; colour edges within 2 px of depth edges | SENSOR-09/10/11 (today about 95.3° corners and 10% lateral scale) | manual protocol plus fakenect golden |
| G12 | Room-scale mesh | Synthetic 6×5×2.7 m room with a 360° pan and a 3 m walk: no loss, ATE < 2 cm, Chamfer < 5 mm, RSS < 1 GB (CPU hash at 1 cm), triangle output not truncated; 0 non-manifold edges and 0 degenerate triangles after cleanup | FUSE-08, TRACK-12, SOTA-08 | `room_scale_synthetic` |
| G13 | Preprocessing | ≤ 5 ms on the CPU at 8 threads; ≤ 0.5 ms on the GPU | SENSOR-03 target of 3 ms | `bench_preprocess` |

## Revisited big-fix decisions

| big-fix decision | New decision | Why |
|---|---|---|
| D1/D4/D5: CPU-only, `.cu` untouched, CUDA deferred | **Two lanes.** CPU is the reference oracle; CUDA is production. `.cu` edits are allowed and gated by `ctest -L gpu`. | nvcc 13.4 builds sm_89 and sm_120 with 0 warnings (GPU, REVIEW). The shipped `build/` is CUDA and was never gated (GAP-01/02, GAP-07). CUDA is the only lane measured at 30 fps. |
| D2: missing compilers are fine | Keep this for HIP only, and delete HIP. `hipcc` is absent (GPU-20's "/usr/sbin/hipcc" is REFUTED). | AUTO has to resolve deterministically. |
| Canonical "unobserved voxels sampled as +1.0 for meshing" | **Replace.** One `SdfSampler`: a trilinear sample is *unknown* unless all 8 corners have weight ≥ `w_min`. MC skips cubes with an unknown corner. The raycast stops on unknown samples. | FUSE-11, SOTA-04: 100% phantom hits at 180°. |
| Raycast accepts +→− and −→+ crossings (`5233721`, locked by `tsdf_raycast_contract` D and `thin_feature` A/D) | **Only +→−, both samples valid.** | PERF-16, FUSE-06, SOTA-04. |
| "Deterministic" integration via a global sort (`7438640`, Todo 14) | **Voxel-projective, one writer per voxel.** Deterministic by construction. | 150 ms serial vs 3–7 ms; also fixes the bias and the weight semantics. |
| ICP damping: fixed Tikhonov 0.01/0.1/1.0 | **Scale-aware LM** (μ·trace(A/N)/6) plus a degeneracy projection. | TRACK-06: damping is inert because A scales with N. |
| ICP success = converged ∨ step ≤ 1e-3 (`e6999ff`) | **Quality tiers GOOD/POOR/FAILED** (see T0.12, T3.7). | Reconciles REVIEW-02 (too strict) with TRACK-06 (accepts degenerate solutions). |
| Depth = 11-bit disparity, "raw 0 and ≥ 2047 invalid", FX = 525 constexpr | **`FREENECT_DEPTH_REGISTERED` uint16 mm** with runtime `CameraIntrinsics` (defaults from `freenect_copy_registration`, override from a per-serial JSON). MM plus extrinsics later (T3.11). | TRACK-03, SENSOR-09/10, SOTA-05, REVIEW-08, FUSE-05. |
| SignalConditioner chain (bilateral RGB, median, EASU/RCAS SR, CAS, guided depth, hole fill, EMA, requantise) | **Delete.** Use LUT → float metres → flying-pixel mask → tracking-only bilateral. Integration uses the unfiltered (masked) depth. | SENSOR-03..08, TRACK-04/05: fake ramps, invented geometry, smear, 29 ms, and SR output never consumed. |
| Colour: float sRGB [0,1] in the voxel, blended in sRGB with the geometry weight | Colour in a separate plane, blended in linear light, with its own weight; uint8 or 16-bit fixed-point storage (CPU and CUDA identical). | FUSE-15, GPU-13, PERF-12. |
| Configure-time "anti-synthetic" regex gates (74 `FATAL_ERROR` from about `tests/CMakeLists.txt:720`) | **Delete** (T0.1). Linking the real library is the proof. | They grep test text and block any refactor (REVIEW-13, critic §2). |
| Test acceptance on Debug `build-cpu-big-fix` only | The gate runs **Release and Debug**, CPU and CUDA, plus a real-binary fakenect smoke test. | GAP-07; F1–F4 never ran. |
| Three worker threads with retain-latest queues | **One sequential fusion worker** (preprocess → track → integrate → raycast) plus an async mesher on dirty-brick snapshots. `ModelFrame::pose` is still kept. | Removes model staleness by construction (REVIEW-09, SOTA-09; critic §2.5). |
| Dense 256³ cube with origin (−1.28, −1.28, 0) | Interim: auto-centred, gravity-aligned volume. Target: 8³ voxel-block hash. | The camera starts on the volume face (TRACK-12, FUSE-08). |

## Phase 0 root cause: the HANG (P0)

### Mechanism (VERIFIED end to end)

1. **Timestamp unit**
   - `src/sensor/KinectSensor.cpp:155` and `:195` compute `timestamp / 1000.0` and treat the result as ms.
   - The gate is `include/sensor/KinectSensor.h:67` (`kMaxFrameSyncDeltaMs = 50.0`); the check is `pairPendingLocked` at `KinectSensor.cpp:218-254`.
2. **The clock really is 60 MHz**
   - Vendored `libfreenect/OpenNI2-FreenectDriver/src/VideoStream.hpp:51` says: "60MHz clock … overflows in ~70s".
   - A live device (045e:02ae) measures a depth period of 2,002,155 ticks and an RGB period of 2,000,287 ticks, i.e. 59,999,325 ticks/s against the host clock.
   - The wrap interval is therefore 2³²/60e6 = **71.58 s**. The code and docs say 71.6 minutes.
3. **Phase slip**
   - The period difference is 1,868 ticks per frame, so the offset cycles every ~35.7 s.
   - A "50 ms" window is 100k ticks wide (±50k), so it is open for ~54 of every ~1,071 frames.
   - Replays:
     - `ts80.txt`: 162/2400 paired, bursts `37 17 0 0 0 0 0 14 40 0 0 0 0 0 0 54 0` per 5 s, longest gap 33.97 s.
     - Fakenect dump: 0/165.
     - `e2e` under libfakenect: 0 paired frames in 12 s.
4. **Why it is silent**
   - Rejected pairs `return nullptr` with no counter (`:233-243`), and pool exhaustion is silent too (`:151`, `:191`).
   - `capture_fps` holds the last instantaneous value (`PipelineController.cpp:680-693`), and the state stays `Running`.
5. **Introduced by `c311c69`.** Base `3841b9d` also divided by 1000 but had no gate. `tests/kinect_pairing_contract.cpp:81,243-269` injects microsecond values, so the test locks the bug in.
6. **Scope:** both the CPU and CUDA binaries share the sensor code.

### Second layer: the apparent hang after the unit fix

- CPU integration runs at about 1.9 Hz, so each processed frame sees about 10× the motion of a 30 Hz frame. ICP then fails.
- The integration queue is cleared on loss (`PipelineController.cpp:1034-1048`).
- The preview is emitted only from `integrationLoop` (`:1143-1186`; the live emission was removed at `:790-792`), so the screen freezes.
- Three further bugs keep the lost state from ever recovering:
  - relocalization hypotheses `h1 == h3 == prev_pose`, and `h2` is +5 cm along world z (`:908`, `:915`);
  - the ICP reference pose does not match the pose the model was rendered at (TRACK-01);
  - frames are dropped until the first raycast exists (`:816-827`).
- **Pairing, throughput, reference pose and relocalization must ship in the same phase.** Otherwise the user trades a hard hang for a frozen TrackingLost screen.

### Other Phase 0 hang/crash risks

| ID | Status | Risk |
|---|---|---|
| GAP-02 | reproduced | `stop()` → `extractGlobalPointCloudGPU` → uncaught `cudaMalloc` OOM → `terminate` (rc 134). Reached from Stop, Close, Reset and the destructor. |
| HANG-10 / REVIEW-03 | code path VERIFIED | Reset on a `std::thread` → `stop()` → a synchronous `on_frame` → `makeCurrent()` off the GUI thread. |
| HANG-08 | VERIFIED | A 15 ms O(volume) scan holds `metrics_mutex_`, which the libusb event thread also takes. Capture-path p99 was 22–30 ms under load. |
| HANG-11 | PLAUSIBLE, low likelihood | `freenect_stop_*` runs on the caller thread while the capture thread pumps libusb. Fix it as hygiene. |

### Regression tests (all must land in Phase 0)

1. `kinect_pairing_realtrace_contract`
   - Synthetic periods of 2,002,155 and 2,000,287 ticks, 4,000 frames, 20 start phases over 0–2e6 ticks, one run starting at `0xFFF00000` so it crosses the wrap. Plus a fixture of about 300 lines embedded from `ts80.txt`.
   - Asserts: ≥ 99.9% published; longest unpublished run ≤ 1; every |Δ| ≤ 16.7 ms + 1 tick; no loss at the wrap.
   - HEAD fails; the fix passes.
2. `smoke_fakenect` (both lanes, Release and Debug)
   - Real binary path: `KinectSensor` → libfreenect API → `LD_PRELOAD` libfakenect with a **generated** dump (T0.4).
   - Asserts: ≥ 25 paired fps; integrated ≥ 20 fps (CUDA) or ≥ 10 fps (CPU after Phase 0); `stop()` < 300 ms; rc 0; non-empty exported mesh.
   - HEAD gives 0 paired frames.
3. `pipeline_watchdog_contract`
   - 300 synthetic moving frames at 30 Hz through the seam.
   - Asserts: `frame_count == 300` within 11 s; integrated ≥ 150; queues drain in 1 s; `onRawFrame` and `metricsSnapshot` max < 1 ms; `stop()` < 300 ms. CTest TIMEOUT 30 s.
4. `sensor_stop_cycle_contract`: 200 start/stop cycles under fakenect, each stop < 300 ms (gated on the dump being available).
5. `pipeline_stall_visibility_contract`
   - Inject depth only, or RGB at +100 ms skew.
   - Asserts: within 1.5 s, `SensorStalled` or `rgb_valid=false` depth-only publishing is observed; `pair_rejects` or `depth_only_published > 0`; `capture_fps == 0` after 1.5 s with no frames at all.

## Phase 0 status (2026-09-22)

`[x]` done, `[~]` partial (what is left is named on the todo line). Measured
after Phase 0 with libfakenect replaying the recorded dump through the real
controller: CUDA 29-30 integrated fps (both GPUs, device auto-selected), CPU
13-16 fps (was ~2, and 0 paired frames before T0.2); stop() 1-76 ms with no
abort; synthetic closed-loop trajectory ATE 4.3 mm; relocalization recovers a
12 deg / 3 cm offset on the first frame. `scripts/gate.sh`: CPU Release 47/47,
CPU Debug 47/47, CUDA builds.

## Field report: cap_001, handheld photosphere room take (2026-09-23)

Counter-clockwise turn with repeated up/down sweeps; trimmed 1286 frames (42.9 s).
Replayed with `azu_replay` (CUDA, Room preset).

- **Operator speed.** Accelerometer pitch: sweeps of -83..+66 deg; mean rate
  60-100 deg/s and peaks 150-180 deg/s, rising through the take. spin360-slow was
  ~12 deg/s. This is past what frame-to-model ICP on a Kinect v1 holds, so the
  capture guide now asks for <= ~30 deg/s (docs/RECORD_AND_REPLAY.md,
  context/capture_data.md).
- **Rolling shutter (new, not in the todos).** ICP RMS tracks pitch rate (r = 0.58),
  not depth (median depth 1.1-2.4 m, where sensor noise is 2-9 mm). The opt-in unwarp
  `AZU_RS_READOUT_MS` cut mean RMS in the worst sweep segment from 15.1 to 8.7 mm
  at 20-33 ms. 60 ms overshoots; a negative value is worse. See T3.14.
- **First loss at frames 352-419 with every variant tried** (one early run: 480).
  Tried: the unwarp, RMS gate 25/35 mm, `AZU_PREPROCESS=minimal`, velocity model,
  no degeneracy hold, legacy intrinsics. What happens at ~405: a fast sweep pushes RMS past
  15 mm, Poor frames stop integration, the view pitches up onto a mostly bare
  wall the model does not cover, and ICP diverges (33 deg in one frame).
  Constant-velocity prediction is not the problem: median error 0.32 deg, p90
  0.95 deg.
- **Relocalization accepted wrong poses and integrated them.** Re-acquired on a
  Poor fit at 35 mm RMS with a 17 deg tilt error. Also re-acquired three times
  with 87-93 deg tilt error (likely the floor fitted to a wall); after the first,
  28 of the next 30 frames graded Good and were integrated; the mesh shows a rotated second
  copy of the room. Fixed in part: relocalization exits only on a Good fit. The
  rest needs gravity (T3.13): a tilt gate of ~15 deg against the accelerometer
  would have rejected all four.
- **GPU ICP is not repeatable near a failure.** Four identical runs: 397-908 Good
  frames, 89-359 deg of yaw. T2.15 is a prerequisite for trusting end-to-end A/B.
  The luckiest run meshes nearly the whole room.
- **azu_replay default.** Without `--preset` it used a 2.56 m object box in front
  of the camera, so an operator's run lost the turn at once. The Room preset is
  now the default (`--preset none` gives the old behaviour).

Suggested order for the next session:
1. ~~T3.13 gravity check~~ first slice done (2026-09-23, see T3.13).
2. T3.14, unwarp on by default for Kinect input.
3. T2.15.
4. T4.5.
5. T5.1 speed warning.
6. T4.6 offline refine (several passes over a recording).

## Execution rules

- **Git (repo and `/home/iphands/prog/slop/CLAUDE.md`)**
  - History is append-only: no `--amend`, `rebase`, `reset --hard`, `revert` or force push, and **never push**.
  - Chain every edit and commit with `&&`, and re-read files before claiming them in a commit message.
  - Small commits, one logical change each. No co-author trailers.
- **Tests travel with their fixes.** Every fix that changes locked behaviour rewrites the locking contract **in the same commit**, or the gate goes red. The contracts that lock current bugs:
  - `kinect_pairing_contract` (µs unit)
  - `tsdf_integration_race_contract` (multi-hit oracle, lines 301-325)
  - `tsdf_integration_min_depth_contract` (192-218)
  - `color_convergence_contract` K6 (499-507)
  - `tsdf_raycast_contract` D (back face)
  - `tsdf_subvoxel_thin_feature_contract` A/D
  - `icp_numeric_policy_contract` (convergence-gated success, damping)
  - `depth_ema_determinism_contract` and `sr_upscaled_contract` (deleted with the EMA and SR)
  - `pipeline_state_contract` (drop counts)
- **Bench numbers in commit messages** must come from an idle host (load average < 2) with the benchmark protocol below. Numbers taken under load are labelled as such.
- **CUDA tests** use `SKIP_RETURN_CODE 77` when there is no device or free VRAM is below the test's need. Never fail CI because vLLM or llama-server is holding memory.
- **Deferral dossier.** `docs/CUDA_HIP_DEFERRED_CHANGES.md` becomes a *worklist*. Each row closed by this plan gets struck with the closing commit hash; T6.3 deletes the file once it is empty.

## Todos

Format: **ID. Title**, then what/why, sources, files, implementation notes, acceptance, effort (S ≤ ½ day, M ≤ 2 days, L > 2 days), and dependencies.

### Phase 0: hang, crash and core correctness (ship as one release)

- [~] **T0.1. Unblock the test gate: delete the regex source gates, add shared synthetic-scene support, run the gate in Release and Debug** — done in 7051315, 40c8eaf (SyntheticScene); partial: gate.sh (CPU Release+Debug ctest, CUDA build) and tests/support/SyntheticScene.h done; configure-time source-text gates NOT removed (needs maintainer go-ahead), Metrics lib and GPU ctest lane open
  - Why: the 74 configure-time `FATAL_ERROR` greps (for example "must literally call `startWithoutSensorForTests(`") fail any refactor. There is no ground-truth scene helper. The gate ran Debug only.
  - Sources: critic §2 "configure-time gates", REVIEW-13, GAP-07 (Release part), HANG-13 (flake context).
  - Files: `tests/CMakeLists.txt` (≈ `:719-1465`), new `tests/support/SyntheticScene.{h,cpp}`, `tests/support/Metrics.{h,cpp}`, `scripts/test-cpu-big-fix.sh` → `scripts/gate.sh`.
  - Notes:
    - Remove every `file(READ)`/`string(REGEX)`/`FATAL_ERROR` that inspects test source text. Keep the link-level guarantee: seam tests link `azu_test_pipeline`, which compiles `src/app/PipelineController.cpp`.
    - Add an `azu_add_test(name SOURCES … LIBS … LABELS … TIMEOUT …)` helper.
    - `SyntheticScene`:
      - an analytic SDF (planes, box, sphere, tilted slab, room-with-furniture), sphere-traced into depth;
      - Kinect noise σ(z) = 0.0012 + 0.0019·(z−0.4)² plus disparity quantisation via round-trip through the current raw curve (and, after T3.1, mm quantisation);
      - seeded `std::mt19937` (seed 42);
      - takes `CameraIntrinsics` and an optional RGB texture.
      - Start from `PERF/bench.cpp`, `TRACK/probe.cpp`, `FUSE/probe2.cpp` and `REVIEW/probe/e2e_probe.cpp`.
    - `Metrics`: ATE (Umeyama SE3 alignment, Horn), RPE, mesh→SDF signed error, Chamfer and 95th-percentile Hausdorff via a hash grid, completeness@τ, non-manifold, degenerate and component counts.
    - `gate.sh`: configure and build `cpu-release` and `cpu-debug` (and `cuda-release` when nvcc exists), run `ctest -L cpu` (and `-L gpu`), and print a summary.
  - Acceptance:
    - `grep -c FATAL_ERROR tests/CMakeLists.txt` ≤ 5, and each remaining one is about toolchain or config, not source text.
    - All 37 existing tests pass in both build types.
    - A metrics self-test: ATE of GT vs GT is 0; GT + 3 mm noise gives ATE = 3 mm ± 5%.
  - Effort: M. Depends on: none.

- [x] **T0.2. Fix the timestamp unit and make pairing depth-led, nearest-neighbour and wrap-safe** — done in 07e0cdf
  - Sources: HANG-01, SENSOR-01, SENSOR-02, REVIEW-01, HANG-18 (header comment), SENSOR-19 (RGB double copy).
  - Files: `include/sensor/KinectSensor.h`, `src/sensor/KinectSensor.cpp`, `include/sensor/FrameData.h` (`RawFrame`), `tests/kinect_pairing_contract.cpp`, new `tests/kinect_pairing_realtrace_contract.cpp`, `tests/data/kinect_ts_trace.txt` (about 300 lines).
  - Notes:
    - Constants: `constexpr double kKinectTickHz = 60e6; constexpr double kTicksPerMs = 60000.0; constexpr double kMaxColorSkewMs = 17.0;` (rename from `kMaxFrameSyncDeltaMs`).
    - `RawFrame` gains `uint32_t depth_ticks, rgb_ticks; uint64_t depth_ticks_unwrapped; int64_t host_steady_ns; bool rgb_valid;`.
    - Unwrap per stream: `if (ts < last && last - ts > 1u<<31) hi += 1ull<<32;`.
    - Delta: `int32_t d = int32_t(depth_ticks - rgb_ticks); double dms = d / kTicksPerMs;`.
    - Pairing:
      - keep a 3-slot RGB ring;
      - on depth, choose the RGB with minimal |d|;
      - if the newest RGB is older than depth − 16.7 ms, hold the depth frame until the next RGB arrives or 40 ms of host time passes, then choose the nearer;
      - accept when |d| ≤ 17 ms, else publish **depth-only** (`rgb_valid=false`);
      - geometry never waits on RGB.
    - Swap RGB vectors instead of making the two 921 KB memcpys (`:194`, `:246`).
    - Rewrite `kinect_pairing_contract` in ticks.
    - Also add the pitfall to `/home/iphands/prog/slop/context/pitfalls.md`, using the template "libfreenect timestamps are 60 MHz ticks", with sources `azu: KinectSensor.cpp (pairPendingLocked)`.
  - Acceptance:
    - Regression test 1 (above).
    - `sim.py` equivalence: `ts80.txt` gives ≥ 2396/2400.
    - Removing RGB samples 100–200 still publishes 100% of depth frames, and exactly those frames have `rgb_valid=false`.
  - Effort: S. Depends on: none.

- [x] **T0.3. Make starvation and stalls visible** — done in a7693e2; metric flag `sensor_stalled` instead of a new PipelineState
  - Sources: HANG-02, GAP-03.
  - Files: `KinectSensor.{h,cpp}`, `include/app/PipelineMetrics.h` (or equivalent), `PipelineController.cpp`, `src/gui/MetricsPanel.cpp`.
  - Notes:
    - Atomics in `KinectSensor`: `depth_callbacks, rgb_callbacks, pairs_published, depth_only_published, pair_rejects, pool_exhausted, last_publish_steady_ns, last_delta_ms`.
    - `PipelineMetrics` gains the same fields plus `stalled`, `backend` and `backend_reason`.
    - Compute `capture_fps` and `tracking_fps` from counter deltas over a 1 s window on the 5 Hz GUI timer, not from an instantaneous 1/dt.
    - Watchdog: while Running and `now − last_publish > 1 s`, raise a rate-limited WARN with the counters, set `PipelineState::SensorStalled`, and show red "NO FRAMES for N s (depth cb X, rejects Y)".
    - Log the pairing rate and median |Δ| every 150 frames.
  - Acceptance: regression test 5.
  - Effort: S. Depends on: T0.2.

- [x] **T0.4. End-to-end fakenect smoke test in both lanes (real binaries)** — done in 589a122
  - Sources: GAP-07, SENSOR-17(d), HANG test suite #4, REVIEW-01 (fakenect replay test).
  - Files: new `tools/make_fake_dump.cpp` (uses `SyntheticScene`), `tests/smoke_fakenect.cpp` (template: `CRITIC/e2e_gpu.cpp`, `HANG/e2e.cpp`), `tests/CMakeLists.txt`.
  - Notes:
    - Do **not** check in the 252 MB real dump. Instead, generate a 60-frame fakenect dump at test time from `SyntheticScene`:
      - `INDEX.txt` plus `d-<time>-<ts>.pgm` and `r-<time>-<ts>.ppm`;
      - fakenect parses the timestamp from the name (`fakenect.c:142`);
      - timestamps use the real periods of 2,002,155 and 2,000,287 ticks with a 19.8 ms phase offset, starting 2 s before a uint32 wrap;
      - depth PGM is written little-endian (SENSOR-17 pitfall);
      - include a `device.json` with registration parameters copied from `SENSOR/rec/device.json`, so REGISTERED mode works after T3.1.
    - Run the real `PipelineController::start()` with `LD_PRELOAD=<libfreenect build>/lib/fakenect/libfakenect.so FAKENECT_PATH=<generated>`.
    - Replay loops by default (`FAKENECT_LOOP`, `fakenect.c:540`).
    - Optional label `real_dump`: when `AZU_REAL_DUMP=/path` is set, replay a real recording as well.
    - Skip with 77 when libfakenect is absent.
  - Acceptance:
    - The assertions in regression test 2.
    - The test fails on `de83ab5` + the harness (0 paired frames) and passes after T0.2.
  - Effort: S–M. Depends on: T0.1, T0.2.

- [~] **T0.5. Make `usageFraction` O(1) and take the metrics path off mutexes** — done in 8ad7ed1; partial: O(1) usage counter outside metrics_mutex_; lock-free metrics/seqlock open
  - Sources: HANG-08, PERF-06, FUSE-07, REVIEW-09 (race part), critic HANG-08 (use-after-free via `setParams`).
  - Files: `src/tsdf/TSDFVolume.cpp:588-594`, `include/tsdf/TSDFVolume.h`, `PipelineController.cpp:680-693`, `:1296-1304`, `:1396-1400`.
  - Notes:
    - `std::atomic<int64_t> observed_voxels_`, updated from per-thread partial counts of 0→>0 weight transitions in integration, and reset in `unlocked_reset`.
    - Make the `PipelineMetrics` counters atomics. `onRawFrame` (libusb thread) must never take a mutex. `metricsSnapshot` reads atomics plus a seqlock for pose and strings.
    - Remove the unlocked full scan entirely.
  - Acceptance:
    - A contract checks `usageFraction` against a brute-force count after random integrations, and the call takes < 1 µs.
    - Seam timers over 300 frames: `onRawFrame` and `metricsSnapshot` max < 1 ms.
    - TSan (noomp preset) runs `setHyperparams` while running and reports nothing.
  - Effort: S. Depends on: none.

- [x] **T0.6. Never touch GL off the GUI thread** — done in c148a56
  - Sources: HANG-10, REVIEW-03.
  - Files: `PipelineController.cpp:469-471` (the synchronous `on_frame` in `stop()`), `dispatchUiFrame` (`:595-609`), `src/gui/MainWindow.cpp:212-218`, `:308-312`, `src/gui/OpenGLWidget.cpp:54-58`.
  - Notes:
    - `stop()` publishes its final frame via `dispatchUiFrame()` (queued).
    - The `MainWindow` frame callback marshals with `QMetaObject::invokeMethod(this, …, Qt::QueuedConnection)`.
    - Add `Q_ASSERT(QThread::currentThread() == thread())` in `onFrameReady`, `onMeshReady` and `OpenGLWidget::update*`.
    - Fix the false comment at `MainWindow.cpp:212-214`.
  - Acceptance: a seam test calls `reset()` from a `std::thread` while running (with a `QCoreApplication`); the recorded hook thread id is never that thread.
  - Effort: S. Depends on: none.

- [x] **T0.7. `stop()` never throws or blocks the GUI; drop the final full-volume point cloud** — done in c148a56
  - Sources: GAP-02, HANG-15, GPU-07 (stop part), GPU-03 (worker-thread throws).
  - Files: `PipelineController.cpp:445-497`, `:459-463`, `TSDFVolume.cpp:600-642` (`extractGlobalPointCloud*`), `include/utils/CudaUniquePtr.h:40`, `MainWindow.cpp:158`, `:355`.
  - Notes:
    - Delete the stop-time `extractGlobalPointCloud` call. Show the last published mesh instead.
    - Mark `stop()` `noexcept`. Wrap all worker-thread GPU entry points in try/catch that sets `gpu_fault_` and a `last_error` string. Full `KF_CUDA_TRY` comes in T2.12.
    - Run Stop through `startBackgroundOp` (as Reset already does).
    - Long loops check `running_` per row or chunk.
    - Stop must **not** free the GPU volume. It is freed only on Reset or quit (full persistence in T2.17).
    - Env hook `AZU_CUDA_MALLOC_FAIL_AFTER=N` in `make_cuda_unique` for tests.
  - Acceptance:
    - CUDA lane: with the fail-after hook, `stop()` returns, the process survives, state is `Stopped` or `Error`, and the last mesh can be exported.
    - `stop()` < 100 ms with workers busy.
    - GAP-02's reproduction (4060, 290 frames, Stop) exits with rc 0.
  - Effort: S. Depends on: none.

- [x] **T0.8. Fix `KinectSensor` lifecycle hygiene** — done in a7693e2; getLatestFrame()/releaseFrame() kept: a configure-time source gate requires them
  - Sources: HANG-11 (P2, per critic), SENSOR-14, REVIEW-17, HANG-19 (`setFrameCallback` lock, ignored return values, `releaseFrame` no-op), SENSOR-19 (dead APIs, `stats_.median_filtered` double count).
  - Files: `KinectSensor.cpp:45-108`, `:293`, `KinectSensor.h:83`.
  - Notes:
    - `stop()`: `running_=false` → join the capture thread → `freenect_stop_depth/video` → close. Only one thread ever pumps libusb.
    - `init()`: on failure, call `freenect_shutdown(ctx_)` and set `ctx_=nullptr` (fixes the leak at `:56-59`).
    - Check `freenect_start_*` return values (`:85-86`).
    - `setFrameCallback` takes `sync_mutex_`.
    - Delete `releaseFrame()`, `getLatestFrame()` and `ready_frame`.
    - Fix the `stats_.median_filtered` double increment (`:358`, `:474`), unless that code is deleted by T2.1.
  - Acceptance: regression test 4 (200 cycles, each < 300 ms); 50 cycles on the real device, if attached.
  - Effort: S. Depends on: T0.4.

- [x] **T0.9. Voxel-projective TSDF integration (CPU canonical)** — done in 450901f, eff9464
  - Sources: PERF-01, PERF-04, HANG-04, HANG-12, FUSE-01, FUSE-02, FUSE-03, TRACK-02, GPU-05 (CPU side), SOTA-01, SOTA-02, REVIEW-05.
  - Files: `src/tsdf/TSDFVolume.cpp:213-398` (`integrateCPU`), `include/tsdf/TSDFVolume.h`, tests `tsdf_integration_race_contract`, `tsdf_integration_min_depth_contract`, `color_convergence_contract` (K6), new `fusion_bias_contract`, `fusion_drift_contract`.
  - Notes:
    - Frustum AABB in voxel coordinates: take the 8 frustum corners at `min_depth − τ` and `max_depth + τ`, plus the camera centre, and clamp to the volume.
    - Loop: `#pragma omp parallel for collapse(2) schedule(dynamic,16)` over (z, y) inside the AABB.
    - Per x-row: compute `c = R_cw·(origin + (x0,y,z)·vs) + t` once, then advance `c += R.col(0)·vs`.
    - Per voxel:
      - `u = floor(fx·c.x/c.z + cx + 0.5)`, and `v` likewise;
      - read `d = depth[v·W+u]`;
      - reject non-finite values and anything outside `[min_depth, max_depth]`;
      - `sdf = d − c.z`; skip if `sdf < −τ`;
      - `tn = min(1, sdf/τ)`.
    - Exactly **one** update per voxel per frame, with `w_new = 1` for now (noise weights come in T3.3).
    - **Single coordinate convention: the corner**, `voxelToWorld(i) = origin + i·vs`, shared by the integrator, `getTSDF`, MC and the raycast. Add a comment in `TSDFVolume.h` saying so.
    - Colour update only when `|sdf| < 0.5·τ` (tightened in T3.2). Keep the NaN-colour guard.
    - Diagnostics become integer per-thread partial sums, reduced in fixed order.
    - `max_weight` now counts **frames**, not ray samples (about 40 per frame before). Retune presets (for example Human 64 → 64 frames is fine; default 128 → 100).
    - Record this in `docs/CANONICAL_SEMANTICS.md`.
    - Delete the `Candidate` buffer, the sort and the fold.
  - Acceptance:
    - Byte-identical output across `OMP_NUM_THREADS` = 1, 4, 32.
    - After N identical frames, weight == min(N, max_w).
    - Plane at z = 1 m, 10 noise-free frames: |mean signed MC vertex SDF| < 0.5 mm (today 6.67 mm).
    - `fusion_drift_contract` (40-frame closed loop, `FUSE/drift.cpp`): |cam z| < 0.5 mm (today 54 mm).
    - `bench_integrate` (perf label): 256³ p50 ≤ 15 ms at 16 threads, idle.
  - Effort: M. Depends on: T0.1.

- [~] **T0.10. Raycast: AABB clip, space skipping, front-face only, one shared validity rule** — done in eea2cb1; partial: no brick space skipping yet (adaptive step only); MC still samples unobserved as +1 (shared SdfSampler open); raycast <= 15 ms needs T2.8
  - Sources: PERF-02 (a, b, d), PERF-16, HANG-05 (with the critic's correction that `t_far` is 2.5 m at defaults), FUSE-06, FUSE-11, SOTA-03 (stage 1), SOTA-04, TRACK-08 (1–4), SOTA-19 (default band), FUSE-09 (min-weight in the sampler).
  - Files: `TSDFVolume.cpp:400-499` (raycast, `computeNormal`), `:509-549` (`getTSDF`), `src/meshing/MarchingCubes.cpp:62,81-85,217-258` (`sampleCorner`), new `include/tsdf/SdfSampler.h`, tests `tsdf_raycast_contract` D, `tsdf_subvoxel_thin_feature_contract` A/D, new `tsdf_raycast_backside_contract` (from `SOTA/phantom.cpp`), `raycast_mc_consistency`.
  - Notes:
    - `SdfSampler::trilinear(p) → {float f; bool valid;}`. `valid` requires all 8 corners to have weight ≥ `w_min` (param `mesh_min_weight`, default 1 now, 3 after T3.5). Use one `floor` for the trilinear weights, and inline it.
    - MC skips any cube with an invalid corner.
    - Slab-clip each ray to `[origin, origin + dims·vs]` ∩ `[min_depth, max_depth]`, passed per call. Remove the defaults `0.3/5.0` from `integrate()` and `raycast()`.
    - Step `max(0.5·vs, 0.8·τ)` while the sample is invalid or `f ≥ 0.999`; otherwise `max(0.5·vs, 0.8·f·τ)`.
    - Accept only when `f_prev > 0 && f_cur ≤ 0` and both samples are valid. If `f_prev < 0` (the ray entered from behind), terminate with no hit.
    - Refine the hit with one secant step.
    - The normal comes from 6 samples at the hit only.
    - Write all outputs on every exit path.
  - Acceptance:
    - Phantom probe: 0 off-surface hits at 120° and 180°; < 0.1% at 45° (today 2.8% / 96% / 100%).
    - ≥ 99.5% of hit pixels on the synthetic room fall within 1 mm of the uniform-step reference.
    - Raycast hit → MC mesh distance < 0.1·vs.
    - `bench_raycast`: 256³ p50 ≤ 45 ms at 16 threads idle (the ≤ 15 ms target needs T2.8).
  - Effort: M. Depends on: T0.9.

- [x] **T0.11. `ModelFrame::pose` becomes the ICP reference pose** — done in 40c8eaf
  - Sources: TRACK-01, PERF-03, REVIEW-04.
  - Files: `include/tracking/ICPTracker.h:44-69`, `src/tracking/ICPTracker.cpp:154-166` (dead `R_ref`, `t_ref`, `R_rel_T`), `PipelineController.cpp:838`, `:861`, `:1147-1154`, `solve_gpu`.
  - Notes:
    - Add `Eigen::Matrix4f pose; uint64_t source_frame_id;` to `ModelFrame`, and set them right after the raycast, before `model_buffers_.swap()`.
    - Call `track(pyr, *model_ref, estimate, model_ref->pose)`. `prev_pose` is used only as the initial estimate.
    - Apply the same change to the GPU path.
    - Delete the dead locals.
  - Acceptance: the `refmismatch` contract (model at P_int, live frame at P_int·Δ with 0.2 s lag at a pan of 0.5 rad/s plus 0.15 m/s) converges within 1 mm / 0.05°. Today it fails with 0 inliers.
  - Effort: S. Depends on: T0.1.

- [x] **T0.12. Tracking quality tiers, v1** — done in 02cc7db
  - Sources: REVIEW-02 (PLAUSIBLE, per critic), TRACK-06 (acceptance part), SOTA-10(b) (partial), critic §2.2.
  - Files: `ICPTracker.cpp:48-60`, `:88-91`, `include/tracking/ICPShared.h`, `PipelineController.cpp:975-986`, `:1034-1048`, `icp_numeric_policy_contract`.
  - Notes:
    - Add `enum class TrackQuality {Good, Poor, Failed}` to `ICPResult`, together with `inlier_ratio = inliers/valid_live`, `rms_mm`, and the existing fields.
    - GOOD: finite, `inlier_ratio ≥ 0.3`, `rms ≤ max(3σ(z̄), 1 cm)`, and inside the motion gate (0.15 m / 30°, scaled by Δt after T3.9). Integrate.
    - POOR: finite and inside the gate, but not GOOD. Update the pose, **do not integrate**.
    - FAILED: anything else. Enter Lost only after **3 consecutive** FAILED frames.
    - When the finest level fails, keep the last good coarser level's pose instead of discarding it (fixes the overwrite at `:48-56`).
    - `converged` is informational, not a gate.
    - Unify the pose caps (`:977` 0.15/0.52 against `ICPShared.h:66-67` 0.2/0.5) into one `TrackingPolicy`.
  - Acceptance:
    - The synthetic trajectory at 2 mm and 5 mm per frame and 1°/frame gives 0 Lost events and ATE < 5 mm.
    - A `plane` scene with 10 cm motion reports POOR or FAILED, never GOOD.
  - Effort: S. Depends on: T0.11.

- [x] **T0.13. Seed the model from frame 1, add real relocalization hypotheses, show a live preview while lost** — done in bb40bcb, 02cc7db; no GUI "LOST" overlay; status label + live preview only
  - Sources: HANG-09, HANG-03, TRACK-01 (fix 3: re-raycast on loss), TRACK-09 (steps 1, 4, 5), SOTA-11 (duplicate hypothesis), PERF-15 (duplicate hypothesis), REVIEW-15 (h1 == h3), HANG-19 (dead hypothesis), TRACK-17 (two ICP runs on failure).
  - Files: `PipelineController.cpp:816-827`, `:892-943`, `:953-956`, `:1143-1186`, `src/gui/OpenGLWidget.cpp`.
  - Notes:
    - First frame: fill `model_buffers_` from its own vertex and normal maps (world = camera = identity), set `model_ready_`, and integrate it.
    - Hypotheses, run at level 2 only, then refine the best one at full resolution:
      - the last GOOD pose;
      - the constant-velocity prediction;
      - `model_ref->pose`;
      - a 3×3 yaw/pitch grid of ±10° about the last GOOD pose.
      - Delete h3, and fix h2 so it moves along camera-forward.
    - On entering Lost: request a "raycast-only" model at the last GOOD pose.
    - Use one ICP run with the better initialisation (chosen at level 2) instead of predicted-then-prev.
    - Live preview from the tracking/fusion thread at ≤ 10 Hz through a latest-only slot (T2.6 finalises it). Overlay "LOST: return to the last view" with the last GOOD raycast ghosted.
  - Acceptance:
    - 20 start phases give a first-ICP success rate ≥ 99%.
    - A 1 s injection gap mid-run, with the camera moving: recovery within 2 s of returning within 10 cm / 10°.
    - `drop_pre_model == 0`.
  - Effort: M. Depends on: T0.11, T0.12.

- [~] **T0.14. Remove super-resolution from the per-frame path** — done in 667692e; partial: upscale opt-in and off in the pipeline; sr_scale/GUI combo/sources not deleted
  - Sources: HANG-06 (SR part), SENSOR-04, REVIEW-06, PERF-07(1), GAP-06 (partial), SOTA-14 (SR claim REFUTED, see appendix).
  - Files: `src/sensor/SignalConditioner_omp.cpp:331-341`, `:367`, `src/sensor/SuperResolution.cpp`, `PipelineController.cpp:773-784`, `include/app/FusionHyperparams.h:16` (`sr_scale`), `src/gui/ControlPanel.cpp:229-234`, `sr_upscaled_contract`.
  - Notes:
    - Delete `applySuperResolutionToRgb`, the guidance CAS, `sr_scale` and the combo box.
    - Delete `sr_upscaled_contract` in the same commit.
    - `git rm` SuperResolution only if nothing else references it. Otherwise it moves to `tools/` in T6.4.
  - Acceptance: preprocessing is ≥ 5 ms faster idle at 32 threads (28.9 → ≤ 24 ms); `grep -r applyEASU src` returns 0.
  - Effort: S. Depends on: none.

- [x] **T0.15. CUDA device selection, VRAM budget, visible backend, and backend resolved once** — done in a719dae; no free-memory stub seam test
  - Sources: GAP-01, GPU-03, GAP-06, GPU-21 (context).
  - Files: new `src/gpu/GpuContext.{h,cu}`, `PipelineController.cpp:114-160` (startInternal), `src/main.cpp:361-364` (`--backend`), Preprocessor backend selection, `MetricsPanel.cpp`.
  - Notes:
    - `GpuContext::select(budget)`:
      - if `AZU_CUDA_DEVICE` or `--backend cuda:N` is set, use that device;
      - else iterate `cudaGetDeviceCount` and pick the device with maximal `cudaMemGetInfo` free memory such that free ≥ budget and compute capability ≥ 8.9;
      - then call `cudaSetDevice`.
    - `budget = volume + MC buffers + model buffers + pyramids + 64 MB`. Point-cloud scratch is gone after T0.7.
    - Preallocate everything in `start()`.
    - If nothing fits: fall back to CPU, **or** offer a resolution shrink. Record the reason.
    - `PipelineMetrics::backend`, `backend_reason`, `gpu_device`, `gpu_free_mb`. The MetricsPanel shows, for example, "Backend: CPU (CUDA OOM on dev0: 458 MB free, need 1.2 GB)".
    - Resolve the backend once and pass it to `Preprocessor`. With no device there must be 0 per-frame `cudaMalloc` retries.
    - `--gl-info` or a new `--probe-gpu` flag prints the choice.
  - Acceptance:
    - A seam test with an injected `freeMem` stub picks the fitting device, or CPU with a reason.
    - `CUDA_VISIBLE_DEVICES=""` for 5 s gives 0 CUDA error lines.
    - With default visibility on this host, the app picks the 4060 (or reports why not), and the UI shows the backend.
  - Effort: S–M. Depends on: T0.7.

- [x] **T0.16. CUDA one-line correctness fixes** — done in a4159da; GPU ICP within 0.54 mm of CPU, not the 0.05 mm target (GPU raycast is still the old algorithm)
  - Sources: GPU-01, GPU-02, GPU-04, REVIEW-10 (colour domain), GPU-13 (partial: rounding and domain).
  - Files: `src/meshing/MarchingCubes_cuda.cu:35-327`, `:62`, `:484-486`, `:517-520`; `src/tracking/ICPTracker_cuda.cu:223`, `:230`; `src/tsdf/TSDFVolume_cuda.cu:135-163`, `:334-336`.
  - Notes:
    - MC winding: emit `c_tri_table[idx][i + 2 − k]`.
    - Delete the local tables. `#include "meshing/MarchingCubesTables.h"` and `cudaMemcpyToSymbol` from `tables::edge_table` and `tri_table`, checking returns.
    - ICP: `local_A += w·J·Jᵀ`; `kHuberK` from `ICPShared.h`.
    - Colour: `×255` on upload and `÷255` on download, round to nearest with `lrintf` and clamp.
  - Acceptance (`ctest -L gpu`, lane from T1.7; until then, a scratch probe run recorded in the commit):
    - outward faces ≥ 0.999 and signed volume > 0 on the sphere;
    - the device symbol table is byte-equal to the header;
    - the 4 ICP motion cases are within 0.05 mm / 0.01° of CPU;
    - colour round trip within ±1 LSB.
  - Effort: S. Depends on: none (tests: T1.7).

### Phase 1: test and benchmark harness

- [ ] **T1.1. Synchronous stepping seam and counter-based waits**
  - Sources: REVIEW-11, HANG-13 (reclassified as load flake), PERF-20 (flaky `pipeline_state`).
  - Files: `PipelineController.{h,cpp}` (seam), `tests/pipeline_*_contract.cpp`.
  - Notes:
    - `processFrameSyncForTests(RawFrame)` runs preprocess → track → integrate → raycast (→ optional mesh) inline on the caller, with no workers.
    - Replace the wall-clock budgets (`pipeline_test_seam_smoke kBudgetMs=5000`, `pipeline_state_contract.cpp:130,217`, `pipeline_mesh_cadence_contract.cpp:132`, 8 `sleep_for` calls) with generation-counter waits bounded only by the CTest TIMEOUT.
    - `pipeline_state_contract`: record `rawQueueDepthForTests()` before the burst, and assert `drops == depth_before + burst − capacity`.
    - Drop the forced `-O2` on the test libraries (`tests/CMakeLists.txt:86-100`) once no budgets remain.
  - Acceptance: `ctest -j32 --repeat until-fail:20` is green on cpu-release, cpu-debug, cpu-asan and cpu-tsan-noomp.
  - Effort: M. Depends on: T0.1.

- [ ] **T1.2. Accuracy contracts on synthetic ground truth**
  - Sources: REVIEW-12, TRACK-16(1), FUSE-17, PERF-04 (test), PERF-20 (`tsdf_accuracy`, `icp_ate`), SOTA-01 (test c), SOTA-02 (test), HANG-12 (test), TRACK-02 (test).
  - Files: `tests/fusion_accuracy_contract.cpp`, `tests/tracking_trajectory_synthetic.cpp`, `tests/e2e_synthetic_scan.cpp` (all label `accuracy`, deterministic, default run).
  - Notes:
    - Scenes: plane, sphere + wall, box, room with furniture, degenerate `plane` and `corridor`, textured plane.
    - With GT poses (fusion only): bias, mean/RMS |err|, Chamfer, Hausdorff95, completeness@1 cm, degenerate/non-manifold/component counts, weight invariant.
    - Tracked (full chain, via T1.1): ATE and RPE, lost count, static drift.
    - The seam smoke test must assert integrated ≥ 90% of injected frames (today it passes with 1 of 16).
    - Each threshold is from the Goals table (G6–G8). A threshold that is stricter than current behaviour lands as `WILL_FAIL`-free only together with its fix. Until then, tag it `accuracy_pending` and keep it out of the default run.
  - Acceptance:
    - The suite runs in < 60 s in Release.
    - On `de83ab5` it fails on bias, drift and ATE (a recorded red run).
    - After Phase 0 it meets G6–G8 except the items owned by Phase 3.
  - Effort: M. Depends on: T0.1, T1.1.

- [ ] **T1.3. `DepthSource` abstraction: Kinect, FakenectDump, Synthetic, TUM, ICL-NUIM**
  - Sources: SENSOR-17, SOTA-12(a, b), TRACK-16(2, 4), FUSE-17 (ICL loader).
  - Files: `include/sensor/DepthSource.h`, `src/sensor/{KinectSource,FakenectDumpSource,SyntheticSource,TumSource,IclNuimSource}.cpp`.
  - Notes:
    - Interface: `bool next(RawFrame&, bool blocking)`, `CameraIntrinsics intrinsics()`, `optional<Pose> groundTruth(t)`, `bool realtime()`.
    - `FakenectDumpSource`:
      - reads `INDEX.txt`, PGM (**little-endian** despite the P5 header), PPM and `device.json` directly;
      - emits frames as fast as they are consumed;
      - uses the recorded timestamps.
    - `TumSource`:
      - reads `rgb.txt`, `depth.txt` and the association within 20 ms;
      - depth PNG/5000 → m (already registered, with the factor pre-applied);
      - fr1 intrinsics 517.3/516.5/318.6/255.3; fr2 and fr3 per the TUM page.
    - `IclNuimSource`: TUM format with fy = −480.0 (handle the sign), and depth scale 5000.
    - The pipeline consumes a `DepthSource`, and `KinectSensor` becomes `KinectSource`.
    - Datasets download via `scripts/fetch_datasets.sh` into `${XDG_CACHE_HOME}/azu/datasets`, outside the repo. Tests skip with 77 if absent.
  - Acceptance:
    - Round trip: `SyntheticSource` → `make_fake_dump` → `FakenectDumpSource` gives identical depth.
    - TUM fr1/xyz loads 798 frames with GT.
  - Effort: M–L. Depends on: T0.1, T0.4.

- [ ] **T1.4. `eval_tum` / `eval_icl` headless tools and the accuracy benchmark**
  - Sources: TRACK-16(2, 3), SOTA-12, SOTA-13 (test), G9, G10.
  - Files: `tools/eval_dataset.cpp`, `tests/bench_datasets.cmake` (label `bench`, non-gating in CI, gating locally), `scripts/bench.sh`.
  - Notes:
    - Runs the real pipeline synchronously via T1.1 (deterministic).
    - Writes `timestamp tx ty tz qx qy qz qw`.
    - In-tree ATE (Umeyama) and RPE (Δ = 1 s), mirroring `evaluate_ate.py` and `evaluate_rpe.py`.
    - Also prints per-stage p50/p95 ms.
    - Sequences: fr1/xyz, fr1/desk, fr1/room, fr2/xyz, fr3/long_office_household, fr3/nostructure_texture_near/far, fr3/structure_notexture_far; ICL-NUIM lr kt0–3 with mesh→GT surface distance.
    - Results are appended to `bench/results.csv` (commit hash, host, load average, device).
  - Acceptance:
    - A GT-as-estimate run gives ATE ≈ 0.
    - GT + 5 mm noise gives 5 mm ± 5%.
    - Agreement with `evo_ape tum --align` within 1%.
  - Effort: M. Depends on: T1.3.

- [ ] **T1.5. Perf and soak labels**
  - Sources: PERF-20, HANG-03 (acceptance), HANG test suite #2/#3, GAP-04, G1, G3, G4.
  - Files: `tests/bench_stages.cpp`, `tests/pipeline_soak.cpp`, `tests/pipeline_watchdog_contract.cpp`, `tests/sensor_stop_cycle_contract.cpp`.
  - Notes:
    - `bench_stages`: per-stage budgets on the synthetic room at 256³, reporting p50/p95.
    - `pipeline_soak`: the real threaded controller at 30 Hz for 10 s (default) or 600 s (`AZU_SOAK_SECONDS`). Asserts:
      - integrated ≥ threshold (CPU 10 Hz in Phase 0, 20 Hz in Phase 2; CUDA 29 Hz);
      - `tracking_ok` in ≥ 95% of 1 s samples;
      - 0 stalls > 200 ms;
      - RSS growth < 50 MB.
    - Label `perf` (not default).
    - The watchdog test is default.
  - Acceptance: on `de83ab5` + T0.2 only, the soak fails at 1.9 Hz; after Phase 0 it passes at the Phase 0 threshold.
  - Effort: M. Depends on: T1.1.

- [ ] **T1.6. Build hygiene: native flags, warnings, presets, sanitizers, freenect detection**
  - Sources: PERF-09, REVIEW-14, GPU-17, GPU-19 (`-diag-suppress`), REVIEW overview (98 warnings).
  - Files: `CMakeLists.txt:24-28`, `:51`, `:119`, new `CMakePresets.json`, `tests/CMakeLists.txt:167`, `:184`, `tests/logger_timer_contract.cpp:83-85`.
  - Notes:
    - Native: `option(AZU_NATIVE ON)` → `target_compile_options(<tgt> PRIVATE $<$<CONFIG:Release>:-march=native -fno-math-errno>)` on the app and `azu_test_core`. The `if(NOT CMAKE_CXX_FLAGS_RELEASE)` branch never runs.
    - Warnings: an `azu_warnings` INTERFACE target with `-Wall -Wextra -Wshadow -Wconversion`, and `-Werror` in the `ci` preset. Fix the 98 warnings over Phase 6.
    - Presets: `cpu-release`, `cpu-debug`, `cpu-asan` (clang at `/usr/lib/llvm/23/bin/clang++`; Gentoo gcc lacks libasan), `cpu-tsan-noomp` (`-DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON`), `cuda-release` (archs `89;120`).
    - Freenect: `option(REQUIRE_FREENECT ON)` → FATAL_ERROR if missing. Also `find_library(freenect HINTS /usr/local/lib)` and append (never replace) `PKG_CONFIG_PATH`.
    - Fix the undefined `${AZU_PIPELINE_TEST_DEFS}`.
    - TSan guard around the `operator new` override in `logger_timer_contract`.
    - Drop `-diag-suppress=20012,20015` in `ci`.
  - Acceptance:
    - `compile_commands.json` for `cpu-release` contains `-march=native`.
    - All presets build.
    - Configure with `PKG_CONFIG_PATH=/opt/cuda/pkgconfig` still finds freenect or fails loudly.
  - Effort: S. Depends on: none.

- [ ] **T1.7. Library split and GPU test lane**
  - Sources: GPU-12, REVIEW-10(3).
  - Files: `CMakeLists.txt:261`, new targets `kfusion_core` (CPU, shared headers `ICPShared.h`, `DepthValidity.h`, `MarchingCubesTables.h`, `ColorMath.h`), `kfusion_cuda`, app; `tests/gpu/*`.
  - Notes:
    - Tests are enabled for any backend.
    - GPU tests run when `CUDA_AVAILABLE AND BUILD_TESTING`. `SKIP_RETURN_CODE 77` on no device or insufficient free VRAM.
    - Fixtures from `GPU/probe/probe.cpp` and `icp_probe.cpp`.
    - Tolerances:
      - exact: depth LUT, MC tables, topology on noise-free fixtures;
      - |mean signed| < 0.1·vs; p99 < 1·vs; outward ≥ 0.999; colour ±2 LSB;
      - ICP vs truth ≤ 1 mm / 0.1°; CPU vs GPU ≤ 0.05 mm / 0.01°.
    - Perf numbers are reported, not asserted.
  - Acceptance: `ctest -L gpu` passes with both `CUDA_VISIBLE_DEVICES=0` and `=1` (or skips with 77 when VRAM is short).
  - Effort: M. Depends on: T0.16.

- [ ] **T1.8. Retarget the remaining implementation-pinning contracts to behaviour**
  - Sources: REVIEW-13, critic "tests that lock the current bugs".
  - Files: `icp_numeric_policy_contract`, `pipeline_state_contract`, `glb_writer_contract` (1125 lines; trim it).
  - Notes: any locking contract not already rewritten by its fix's commit is rewritten here as an invariant: weights, pair rate, ATE, "drops are counted", tier semantics.
  - Acceptance: no contract asserts an internal step count, march-step multiplicity or damping constant.
  - Effort: S–M. Depends on: T0.9–T0.12.

### Phase 2: performance

#### CPU

- [ ] **T2.1. Replace SignalConditioner with a minimal, float, tracking-only chain**
  - Sources: SENSOR-03 (P1, per critic), SENSOR-05, SENSOR-06, SENSOR-07, SENSOR-08, PERF-07(2–4), HANG-06 (rest), TRACK-04, TRACK-05(1, 2), REVIEW-07, SOTA-06, SENSOR-12 (RGB filter removal), SENSOR-19 (CUDA/HIP conditioner duplicates), GAP-06 (code deletion).
  - Files:
    - rewrite `src/sensor/SignalConditioner_omp.cpp` → `src/sensor/DepthPrep.cpp`;
    - `SignalConditioner_cuda.cu` → `DepthPrep_cuda.cu`;
    - delete `_hip.hip`;
    - `include/sensor/DepthValidity.h` (delete `cpuDepthMetersToRaw`);
    - `FrameData.cpp:193`, `:300-303`;
    - `PipelineController.cpp:766-784`;
    - tests `depth_ema_determinism_contract` (delete), new `depth_prep_contract`.
  - Notes:
    1. Convert once: `uint16 raw/mm → float m` via a LUT (2048 entries for 11-bit; mm/1000 after T3.1), band gate [min, max], invalid → 0. `RawFrame::depth` stays untouched.
    2. Flying-pixel mask: invalidate a pixel if any 4-neighbour jump exceeds `max(0.03, 0.05·z)` (reuse `depthJumpThreshold`, `ICPShared.h:45-50`), then erode by 1 px next to invalid pixels. **No hole fill, no EMA, no guided filter, no RGB guidance, no requantisation.**
    3. Tracking copy: 5×5 bilateral on float, σ_s = 2 px (KinFu uses 4.5 px at r = 3–6; tune on T1.2), σ_r = 3·σ(z_center) (or a fixed 30 mm). Precompute `float Ks[25]` and a range LUT indexed by quantised (Δ/σ_r)². No `expf` in the loop. It feeds only the vertex, normal and pyramid maps.
    4. Integration consumes the step-2 depth.
    5. RGB: no bilateral and no median. Pass it through.
    6. Normals are computed per pyramid level from that level's vertices.
  - Acceptance:
    - `bench_preprocess` median ≤ 5 ms at 8 threads (today 52–66 ms at 8 threads idle).
    - Synthetic 10/40/100 mm steps at 1 m and 3 m: 0 in-between pixels more than 1 px from the edge (today a 7-px ramp).
    - Shadow band: 0 filled pixels.
    - Noisy plane: σ reduced ≥ 30% with |bias| < 1 mm.
    - Output has > 64 distinct values (not code-snapped).
    - Noisy single frame: `angF/valid_model` < 10% and inliers ≥ 85% of `valid_live` (today about 66% rejected).
    - `traj` with the prep chain in the loop: ATE ≤ 1.2× without it (today the EMA adds 2.4 mm).
  - Effort: M. Depends on: T1.2.

- [ ] **T2.2. One sequential fusion worker and an asynchronous mesher**
  - Sources: REVIEW-09, SOTA-09, critic §2.5, TRACK-01 (removes lag by construction).
  - Files: `PipelineController.cpp:242-244` (threads), `:651-727` (queues), `include/app/PipelineController.h:216`.
  - Notes:
    - The sensor pushes into a single-slot latest-wins mailbox, counting displaced frames as `dropped_sensor`.
    - The fusion worker runs, per frame: prep → pyramid → track → (GOOD) integrate → raycast at the new pose → publish the preview.
    - The mesher (T2.7) runs on dirty-brick snapshots.
    - Keep `ModelFrame::pose` for async GPU variants.
    - Add an invariant check: `integrated_frame_id − tracked_frame_id ≤ 1`.
  - Acceptance:
    - The soak shows the invariant holds 100% of the time.
    - Every sensor frame is either processed or counted as dropped.
    - CPU integrated Hz ≥ Phase 2 target (G3) at 16 threads idle.
  - Effort: M. Depends on: T0.9–T0.13, T2.1.

- [ ] **T2.3. OpenMP thread budgets, passive waiting, and no parallel regions for tiny loops**
  - Sources: HANG-07, PERF-08, TRACK-07(3), REVIEW-18, SENSOR-19 (OMP), GAP-09 (Threads spin box only reaches ICP). Critic verdict: PLAUSIBLE at P2; decide by measurement.
  - Files: `src/main.cpp` (`setenv("OMP_WAIT_POLICY","passive",0)`, `GOMP_SPINCOUNT=10000` before the first region), each worker's entry (`omp_set_num_threads(budget)`), `ICPTracker::setNumThreads` → `ThreadBudgets{fusion, mesher}`, `ControlPanel` Threads spin box.
  - Notes:
    - After T2.2 there are 2 workers. The budget defaults (fusion = physical cores − 2, mesher = 2) are chosen by an **idle in-pipeline sweep** (fusion ∈ {8, 12, 16, 24, 30}) and recorded in `bench/results.csv`.
    - `if(n > threshold)` clauses on loops under about 100 µs (`buildFrameData`, `computeNormals`).
  - Acceptance:
    - Soak Hz at the chosen budget ≥ Hz with defaults.
    - Under `stress -c 16`, per-stage medians stay within ±20% of idle × 1.5.
  - Effort: S. Depends on: T2.2.

- [ ] **T2.4. Allocation-free steady state: persistent pyramid, pooled frames**
  - Sources: HANG-14, PERF-10, TRACK-07(1), REVIEW-18 (allocations).
  - Files: `FrameData.{h,cpp}:37-42`, `:259`, `:333-340`, `PipelineController.cpp:922`, `:951`, `:1158`, `:1214`, `:1264`.
  - Notes:
    - A `FrameData(w, h)` constructor (no default 640×480).
    - Level 0 is a non-owning view.
    - `downsample(src, dst&)` works in place, using nearest-valid (not mean) at depth jumps (GPU-16).
    - The pyramid is a worker member.
    - The preview uses a pooled 320×240 buffer.
  - Acceptance:
    - `perf_pyramid` ≤ 1 ms.
    - A `mallinfo2` delta hook shows 0 allocations > 1 MB per steady-state frame in the fusion loop.
  - Effort: S. Depends on: T2.2.

- [ ] **T2.5. ICP inner-loop efficiency**
  - Sources: PERF-15, TRACK-07(2, 4), TRACK-11 (double accumulation).
  - Files: `ICPTracker.cpp:176-212`.
  - Notes:
    - `struct alignas(64) LocalAcc` with `double` sums, allocated once.
    - One `omp parallel` region per level containing the iteration loop.
    - At level 0, subsample every 2nd pixel after the first two iterations. Stop early per T3.8.
  - Acceptance:
    - `icp_numeric_policy_contract` passes (as retargeted).
    - Synthetic 30-frame ATE ≤ 5 mm.
    - `track()` p95 < 8 ms at 640×480 idle.
  - Effort: S. Depends on: none.

- [ ] **T2.6. Latest-only preview mailbox and persistent VBOs**
  - Sources: HANG-17, PERF-14.
  - Files: `PipelineController.cpp:595-609` (`dispatchUiFrame`), `OpenGLWidget.cpp` (`uploadPointCloud`).
  - Notes:
    - A `std::atomic<std::shared_ptr<PreviewFrame>>` slot plus `atomic<bool> pending`. Post the queued lambda only on the false → true transition; the GUI takes the latest frame.
    - Allocate the VBO once and update it with `glBufferSubData`.
  - Acceptance: stall the GUI hook for 2 s → ≤ 1 pending delivery, the newest frame is the one delivered, RSS growth < 20 MB.
  - Effort: S. Depends on: T0.6.

- [ ] **T2.7. Incremental dirty-brick meshing with an edge-owned weld, meshed outside the lock**
  - Sources: PERF-05, HANG-16, FUSE-13, PERF-18.
  - Files: `src/meshing/MarchingCubes.cpp:203-214`, `:346`, `PipelineController.cpp:1325-1402`, `src/export/GLBExporter.cpp:135-167`, `include/meshing/MeshData.h`, GL per-brick buffers.
  - Notes:
    - Integration sets `dirty[brick]` (8³ bricks; the dense volume gets a brick index view).
    - The mesher swaps the dirty set, copies each brick plus a 1-voxel apron under a short lock (or a sequence number), then meshes lock-free.
    - Weld by edge ownership: each cube owns its +x, +y and +z edges, with per-slice 2D index arrays. No `unordered_map`, and parallel over z-slabs.
    - Skip bricks whose tsdf min and max share a sign.
    - Keep a `unordered_map<brickKey, BrickMesh>` cache. The GL side uses one VBO range per brick, or rebuilds a concatenated index buffer.
    - Set `MeshData::welded = true` so GLB skips its re-weld.
    - Adaptive cadence: wait at least 2× the last extraction time.
  - Acceptance:
    - `mesh_incremental_contract`: after 50 frames, the incremental mesh equals the full MC after a canonical sort.
    - `perf_mesh_update` ≤ 10 ms for a single-frame dirty set.
    - Fusion-thread lock wait p99 < 1 ms while meshing.
    - `glb_writer_contract` passes with the weld skipped.
  - Effort: L. Depends on: T0.10, T2.2.

- [ ] **T2.8. Brick-occupancy space skipping and raycasting at tracking resolution**
  - Sources: PERF-02(c), SOTA-03 (stages 2, 3), TRACK-08(5), FUSE-06 (block stepping).
  - Files: `TSDFVolume.cpp` raycast, occupancy bitmap.
  - Notes:
    - A 1-bit-per-8³-brick "any weight > 0" bitmap maintained in integration (2.4–3.4% of bricks are occupied, measured). 3D-DDA over bricks.
    - The tracking model is raycast at 320×240 (level 1, with scaled intrinsics) when the CPU lane is used. Full resolution is only for display, on demand.
  - Acceptance: `bench_raycast` 256³ ≤ 15 ms at 16 threads idle, with accuracy unchanged per the T0.10 checks.
  - Effort: M. Depends on: T0.10.

- [ ] **T2.9. Compact voxel layout**
  - Sources: PERF-12, SOTA-19 (voxel bandwidth), FUSE-18 (32-bit indices), SOTA-08 (short term).
  - Files: `include/tsdf/TSDFVolume.h` (`struct Voxel`), `VoxelGPU.h`, all readers.
  - Notes:
    - Geometry plane `{int16 sdf_q (metric, ±τ_max, after T3.3); uint16 weight}` = 4 B.
    - Colour plane `{uint16 r, g, b (linear, 16-bit fixed); uint16 w_c}` = 8 B. uint8 is allowed on GPU per T3.2 if parity holds within ±1 LSB.
    - Indices are `size_t`, or block-local after T4.2.
    - The raycast and MC read only the geometry plane.
  - Acceptance:
    - `color_convergence_contract` converges within 0.5 byte.
    - Determinism holds.
    - Raycast and MC ≥ 1.5× faster idle.
    - The 512³ volume is ≤ 1.1 GB.
  - Effort: M. Depends on: T0.9, T3.3 (sdf units).

- [ ] **T2.10. Volume (re)allocation off the GUI thread; split live vs volume parameters**
  - Sources: PERF-11, GAP-05.
  - Files: `TSDFVolume.cpp:90-145` (`setParams`, `sameParams`, reset `:118`), `PipelineController.cpp:293-295`, `MainWindow.cpp:117`, `include/app/FusionHyperparams.h`, `FusionUiModel.cpp`.
  - Notes:
    - `LiveParams` (depth band, ICP thresholds, mesh min weight) apply at a frame boundary with no clear.
    - `VolumeParams` (res, vs, τ, origin, max_weight) need a confirmation dialog and run through `startBackgroundOp`.
    - Remove the depth band from the `TSDFParams` identity (it is passed per call, per T0.10).
    - Parallel first-touch fill instead of resize + fill.
  - Acceptance:
    - Changing `max_depth` while running leaves `usageFraction` unchanged.
    - Changing `voxel_size` requires `confirmVolumeReset`.
    - `setHyperparams` returns on the GUI thread in < 50 ms.
    - A 512³ reset completes in < 500 ms.
  - Effort: S. Depends on: T0.5.

#### CUDA

- [ ] **T2.11. `IFusionBackend` interface; remove the triplicated `#ifdef` blocks**
  - Sources: REVIEW-10(2), REVIEW-15 (backend part), GPU-21 (recommendation).
  - Files: new `include/fusion/IFusionBackend.h` (`prep`, `pyramid`, `track`, `integrate`, `raycast`, `extractDirty`, `download`), `src/fusion/CpuBackend.cpp`, `src/fusion/CudaBackend.cu`, `PipelineController.cpp:114-184`, `:445-497`, `:1087-1128`, `:1189-1290`, `:1372-1401`.
  - Acceptance: `grep -c "#if.*CUDA_ENABLED" src/app/PipelineController.cpp` == 0, and both lanes pass their tests.
  - Effort: M. Depends on: T1.7, T2.2.

- [ ] **T2.12. CUDA fault handling**
  - Sources: GPU-18, GAP-02 (item 3).
  - Files: all `.cu` files (`CUDA_CHECK` → `KF_CUDA_TRY` returning `Status`), `PipelineController` (`gpu_fault_` → orderly stop, CPU fallback, UI error), `solve_gpu` (`preprocessor_mutex_` at `:865-871`), null checks at GPU entry points.
  - Acceptance: fault injection (a null depth pointer, and a malloc failure via `AZU_CUDA_MALLOC_FAIL_AFTER`) is reported in the UI, with no terminate and no silent lost-tracking loop.
  - Effort: M. Depends on: T2.11.

- [ ] **T2.13. Port the canonical CPU algorithms to CUDA**
  - Sources: GPU-05 (GPU side), GPU-06, GPU-10, GPU-21.
  - Files: `TSDFVolume_cuda.cu:33`, `:52-100`, `:182-183`, `:300-321`; `ICPTracker_cuda.cu:336`, `:467-469`; `ICPShared.h` (`icp::solveStep`).
  - Notes:
    - Integration kernel: pass `[min, max]` (not hard-coded 0.1 / 0.3 / 5.0) and use corner-convention nodes.
    - Raycast kernel: identical to T0.10 (AABB, skipping, front face, validity).
    - Share one host `icp::solveStep(A, b, level)` between both lanes: damping, caps, Rodrigues, `projectToSO3`, `final_step`, quality tiers.
  - Acceptance: the GPU-12 parity suite passes; `det(R) = 1 ± 1e-5`.
  - Effort: M. Depends on: T0.9, T0.10, T0.12, T2.11.

- [ ] **T2.14. Stream-ordered frame pipeline with a single upload**
  - Sources: GPU-08.
  - Files: `SignalConditioner_cuda.cu:434-487` (→ `DepthPrep_cuda.cu`), `TSDFVolume.cpp:172-176`, `TSDFVolume_cuda.cu:203`, `:382`, `:461`, `:479`, `ICPTracker_cuda.cu:362`, `:393`, `:396`.
  - Notes:
    - A pinned (`cudaHostAlloc`) ring of raw uint16 depth (600 KB) and RGB (900 KB).
    - One H2D copy per frame. The band gate in `rawToMeters`, then prep, pyramid, ICP, integrate and raycast, all on `cuda_stream_`.
    - No `cudaDeviceSynchronize` on the hot path; events mark stage boundaries.
    - Capture the ICP level-iteration as a CUDA Graph.
    - Tracking and integration read the **same** band-gated buffer.
  - Acceptance:
    - nsys over 300 replayed frames: ≤ 2 H2D copies per frame, 0 device syncs, GPU time per frame < 5 ms on the 4060.
    - Latency p95 < 40 ms (G4).
  - Effort: M. Depends on: T2.13.

- [ ] **T2.15. Deterministic GPU ICP reduction**
  - Sources: GPU-15, SOTA-16 (reduction design).
  - Files: `ICPTracker_cuda.cu:272` (and the `s_reduce[8][34]` assumption).
  - Notes: warp `__shfl_down_sync` plus a block reduce into a `float[blocks][27]` buffer, then a second kernel sums in fixed order in fp64. Add `static_assert` on the block size.
  - Acceptance: 5 repeats give bit-identical poses (today |Δpose| up to 2e-4).
  - Effort: S. Depends on: T2.13.

- [ ] **T2.16. Edge-owned GPU marching cubes**
  - Sources: GPU-09.
  - Files: `MarchingCubes_cuda.cu:387-393`, `:434-438`, `:454-459`, `:582-627`, `MarchingCubes.h:73` (`max_triangles_`).
  - Notes:
    - Classify cubes and compute owned-edge crossing flags with the `SdfSampler` rule.
    - Exclusive scans for vertex ids and triangle counts.
    - Emit vertices once per edge (rejecting non-finite values), then the index buffer.
    - Size buffers from the scan: no cap, no host weld.
    - Per dirty block after T4.3.
  - Acceptance:
    - CPU vs GPU on the frontier, sphere and table fixtures: identical vertex and triangle counts, positions within 1e-5·vs.
    - Euler characteristic of the closed sphere is 2.
    - Triangle count on the noisy scene within 5% of CPU (today 2×).
  - Effort: M. Depends on: T0.10, T2.13.

- [ ] **T2.17. The GPU volume persists across stop; explicit download**
  - Sources: GPU-07.
  - Files: `PipelineController.cpp:473-484`, `:1502-1517` (`exportMesh` waiting 5 s on a dead mesher), `TSDFVolume_cuda.cu:118-163`.
  - Notes:
    - Free the volume only on reset or quit.
    - On stop, run a final `extractGPU`, publish the mesh, then join.
    - `downloadVolume()` packs on the device to the T2.9 layout.
    - `exportMesh` after stop uses the cached mesh, or extracts synchronously.
  - Acceptance:
    - scan → stop → export gives a non-empty mesh with the same triangle count;
    - stop → start keeps `usage > 0`.
  - Effort: M. Depends on: T0.7, T2.13.

- [ ] **T2.18. CUDA-GL interop preview**
  - Sources: GPU-14, SOTA-16 (cross-GPU note).
  - Files: `OpenGLWidget.cpp`, `CudaBackend`.
  - Notes:
    - Register the preview VBO/PBO with `cudaGraphicsGLRegisterBuffer` only when the GL context's GPU == the CUDA device. Check `cudaGLGetDevices`.
    - Otherwise use pinned async copies of the downsampled preview.
    - Throttle to the display rate.
  - Acceptance: the fusion thread spends < 0.2 ms on the preview (`ScopedTimer`).
  - Effort: M. Depends on: T2.14.

- [ ] **T2.19. Delete HIP; AUTO means CUDA or CPU**
  - Sources: GPU-20 (the hipcc-on-PATH claim is REFUTED; the recommendation stands).
  - Files: 5 `*_hip.hip`, `HipUniquePtr.h`, `CMakeLists.txt:59-80` (`gfx90c`), HIP `#elif` branches, `rocm.md`.
  - Acceptance: `grep -r HIP_ENABLED src include` == 0; AUTO on this host selects CUDA.
  - Effort: S. Depends on: none.

### Phase 3: accuracy

- [ ] **T3.1. Registered depth in mm and runtime `CameraIntrinsics` everywhere**
  - Sources: TRACK-03, SENSOR-09, SENSOR-10 (Option A), SENSOR-11 (Option A), FUSE-05, SOTA-05, REVIEW-08.
  - Files:
    - `KinectSensor.cpp:69-72` → `FREENECT_DEPTH_REGISTERED`;
    - `include/sensor/FrameData.h:14-17` (delete the constexpr FX/FY/CX/CY) → `struct CameraIntrinsics {float fx, fy, cx, cy; int w, h; CameraIntrinsics scaled(int level) const;}` carried in `FrameData` and `ModelFrame`;
    - `ICPTracker.cpp:237`, `PipelineController.cpp:1093-1244`, `DepthValidity.h` (→ mm validity: 0 = invalid, band clamp), the GPU `rawToMeters` kernel;
    - `depth_domain_contract`, `cpuDepthMetersToRaw` users.
  - Notes:
    - Default K = RGB (525/525/319.5/239.5). Log it at startup. Override from `~/.config/azu/kinect_<serial>.json` (T3.11).
    - TUM and ICL sources supply their own K.
    - REGISTERED has about 13% fewer valid pixels (FOV crop, z-buffer cracks).
    - Fakenect replays REGISTERED from `device.json` (`fakenect.c:79-88`, `:250-256`). Update `make_fake_dump`.
    - SyntheticScene quantisation switches to 1 mm steps plus disparity-equivalent noise.
  - Acceptance:
    - Synthetic 90° corner and 1 m cube rendered with K and reconstructed with K: 90 ± 0.3°, edges within 0.5%.
    - A fakenect checkerboard-on-box golden: colour edges within 2 px of depth edges.
    - TUM fr1/xyz ATE is measured and recorded.
    - Real device (manual, G11): corner 90 ± 1°.
  - Effort: M. Depends on: T2.1, T1.3.

- [ ] **T3.2. Colour: lock exposure, fuse separately in linear light**
  - Sources: SENSOR-12, FUSE-15, SOTA-14 (online part), GPU-13, FUSE-05 (AE/AWB).
  - Files: `KinectSensor.cpp` (`freenect_set_flag` for `FREENECT_AUTO_EXPOSURE`, `AUTO_WHITE_BALANCE` and `AUTO_FLICKER` OFF after about 1.5 s warm-up, run on the capture thread; `libfreenect.h:112-114`, `:672`), `TSDFVolume` colour update, `include/utils/ColorMath.h` (`__host__ __device__ srgbToLinear`, `linearToSrgbU8`, round to nearest, reject non-finite).
  - Notes:
    - Colour weight `w_c = cosθ·(z_ref/z)²·edge_mask`, where the edge mask is 0 within 3 px of a depth discontinuity or on saturated pixels. Cap about 32.
    - Update only when |sdf| < 0.5·vs·√3 and `rgb_valid`.
    - A GUI toggle "Lock exposure" (default on).
  - Acceptance:
    - Textured synthetic plane with ±10% gain jitter: mean ΔE76 < 3; bleed width ≤ 1 voxel.
    - Static scene with the lock on: luma drift < 2/255.
    - GPU colour within ±1 LSB of CPU.
  - Effort: S–M. Depends on: T3.1, T2.9.

- [ ] **T3.3. Noise- and quantisation-aware truncation and weights; metric SDF storage; 0.4–4 m range**
  - Sources: FUSE-04, SOTA-07, SENSOR-13, FUSE-14, GPU-10 (weight part), SOTA-02 (noise-aware weight).
  - Files: `TSDFParams`, the integration kernels (CPU and CUDA), `include/sensor/NoiseModel.h` (`sigmaZ(z, θ)`), `FusionHyperparams.h:15` (`max_depth`), `FusionUiModel.cpp` (`kTruncationVoxelRatioMin/Max`).
  - Notes:
    - σ(z, θ) = 0.0012 + 0.0019·(z−0.4)² + (0.0001/√z)·θ²/(π/2−θ)² (Nguyen 2012). It fits the device measurement (6.97 mm at 2–2.5 m; SENSOR-13).
    - τ(z) = clamp(max(3·vs, 3·σ(z), q(z)), τ_min, τ_max), where q(z) is the quantisation step: 0.00307·z² for 11-bit, and for REGISTERED the underlying disparity step still applies.
    - Store the **metric** SDF clamped to ±τ_max.
    - `w = (σ(z0)/σ(z))²·max(cosθ, 0.1)`. Reject cosθ < 0.1 and pixels within 2 px of a discontinuity.
    - nvblox-style dropoff behind the surface: `w *= clamp((sdf+τ)/(τ−vs), 0, 1)` when `sdf < −vs`.
    - max_weight 64–100 frames.
    - Integrate 0.4–3.5 m, track to 4.0 m.
  - Acceptance:
    - Synthetic walls at 1, 2, 3 and 4 m with real quantisation, 30 frames: |bias| < 1 mm everywhere, RMS(4 m) < 6 mm, < 0.1% of vertices > 20 mm off.
    - Floor at 70–80° incidence: bias < 1 mm, RMS < 3 mm.
    - Near-wall RMS does not regress when far walls are added.
  - Effort: M. Depends on: T0.9, T3.1.

- [ ] **T3.4. Free-space carving**
  - Sources: FUSE-10, GPU-10 (carving).
  - Notes:
    - In the projective kernel, voxels with `sdf > τ` that already have weight are fused with tsdf = +1 at weight 0.5·w.
    - Optional sign-disagreement decay.
  - Acceptance: a box present in frames 0–29 and removed in frames 30–89 → at frame 90, no vertex lies within 2 cm of its former surface. Cost increase < 10%.
  - Effort: S. Depends on: T0.9.

- [ ] **T3.5. Mesh quality: min weight, cleanup, degenerate triangles, cap, smoothing and decimation**
  - Sources: FUSE-09, FUSE-12, SOTA-15.
  - Files: `MarchingCubes.cpp:127-131`, `MarchingCubes.h:62`, new `src/meshing/MeshCleanup.cpp`, vendored meshoptimizer (MIT) under `third_party/`.
  - Notes:
    - `mesh_min_weight` default 3 (in frames).
    - Clamp the MC interpolation t to [1e-4, 1 − 1e-4]. Drop triangles with repeated indices or area < 1e-12.
    - Union-find components; drop those with < 500 triangles or < 0.01 m².
    - Size the triangle cap from available memory (for example 50M), never a fixed 2M.
    - At export time: Taubin smoothing (λ = 0.5, μ = −0.53, 5–10 iterations), then `meshopt_simplify` with target error 0.5·vs.
    - Small-hole fan fill for loops < 20 edges.
    - Optional Screened Poisson for a "watertight" export later.
  - Acceptance:
    - Noisy fixtures: < 0.01% of vertices with error > 20 mm (today 1.07%).
    - 0 degenerate triangles (today 116 on the sphere); 0 non-manifold edges; no component smaller than N_min.
    - Decimation gives ≥ 5× reduction at < 2 mm RMS.
    - The room fixture exports with `truncated == false`.
  - Effort: M. Depends on: T0.10.

- [ ] **T3.6. Noise-aware ICP weighting and per-level gates**
  - Sources: TRACK-11, SOTA-10(c), critic risk (Room preset gates of 60° / 0.20 m).
  - Files: `ICPShared.h:34`, `:209` (`kHuberK`), `ICPTracker.cpp:276` (`abs(dot)`), `include/tracking/ICPTracker.h:19-24`, `FusionUiModel.cpp:161-171`.
  - Notes:
    - Residual weight 1/σ(z_live)². Huber k = 1.345 in normalised units (or Tukey c = 4.685 at the fine level).
    - dist_threshold {0.1, 0.15, 0.2} m and angle {20°, 30°, 35°}, fine → coarse.
    - Signed normal test: `dot(n_live_world, n_model) > cosθ`, with live normals flipped to face the camera.
    - Room preset gates: 25° / 0.1 m.
  - Acceptance:
    - Synthetic noisy `traj` ATE ≤ 0.8× baseline.
    - Thin-wall scene: 0 cross-surface matches.
  - Effort: S. Depends on: T3.3 (σ model).

- [ ] **T3.7. Degeneracy-aware solve and full quality tiers**
  - Sources: TRACK-06, SOTA-10(b).
  - Files: `ICPShared.h:261-275` (`dampingForHessian`), `ICPTracker.cpp`, `MetricsPanel`.
  - Notes:
    - Eigen-decompose A/N (it is already computed).
    - For eigenvalues λ_i < τ·λ_max (start at τ = 1e-3; tune on the probe), project the update onto the well-conditioned subspace and take the motion prior along the degenerate directions.
    - Column-scale rotation by the mean depth.
    - LM λ = 1e-4·trace(A/N)/6.
    - Report `degenerate_dims`. A degenerate dimension with no photometric or IMU term caps the tier at POOR.
    - UI: "Featureless surface: add objects".
  - Acceptance:
    - `plane` and `corridor` probes report POOR/FAILED or `degenerate_dims > 0` whenever the error is > 5 mm (today ok=1 at 85 mm / 10°).
    - The room scene stays GOOD at < 0.5 mm.
  - Effort: M. Depends on: T0.12.

- [ ] **T3.8. Convergence criteria and pyramid bias**
  - Sources: GPU-16 (SUSPECTED; applies to both lanes), TRACK-17 (`final_step` units), TRACK-05(3).
  - Notes:
    - Converge on separate thresholds ‖t‖ < 1e-4 m and ‖ω‖ < 1e-4 rad, **and** a relative residual decrease < 1e-4, instead of a mixed-unit norm.
    - Nearest-valid downsampling (from T2.4).
    - Per-level normals.
    - Coarse levels may use more iterations (InfiniTAM style).
  - Acceptance: the identity-pose fixture gives < 0.3 mm / 0.05° with default iterations (today 2.45 mm / 0.58°); room replay drift at rest < 1 mm over 300 frames.
  - Effort: S–M. Depends on: T2.4.

- [ ] **T3.9. Time-aware motion model and dropped-frame detection**
  - Sources: TRACK-10, SENSOR-15.
  - Files: `PipelineController.cpp:203-206`, `:338`, `FrameData` (timestamp).
  - Notes:
    - Carry `t_s` from the unwrapped ticks.
    - ξ = log(T_{k−1}⁻¹T_k)/Δt_prev; prediction T_k·exp(ξ·Δt_now) (SE3 log/exp helper).
    - Scale the motion gate by Δt. Decay the velocity after a loss.
    - Count a dropped frame when Δticks > 1.5 × 2,002,155.
  - Acceptance:
    - Replay with every 10th frame removed: metrics report exactly 10% dropped.
    - With every 3rd frame dropped: ATE ≤ 1.2× the undropped run and no extra failures.
  - Effort: S. Depends on: T0.2.

- [ ] **T3.10. Photometric term (coloured ICP)**
  - Sources: TRACK-13, SOTA-10(a).
  - Notes:
    - The raycast stores linear luminance in `ModelFrame`. Build a 3-level intensity and Sobel gradient pyramid.
    - J_rgb = ∇I·∂π/∂p·[I | −[p]×] with a Huber weight.
    - w_rgb = 0.1 relative to depth (ElasticFusion icpWeight 10), with both terms normalised by their residual σ.
    - Only pixels with |∇I| above a threshold contribute.
    - CPU first, then CUDA.
  - Acceptance:
    - Textured-plane probe: in-plane error < 2 mm (plain ICP > 10 mm).
    - TUM fr3/nostructure_texture_near ATE < 5 cm.
    - The degenerate tier is lifted when the texture constrains the DOF.
  - Effort: L. Depends on: T3.1, T3.2, T3.7.

- [ ] **T3.11. Calibration tool, per-device file, and DEPTH_MM + extrinsics (Option B)**
  - Sources: SENSOR-18, SENSOR-10 (Option B), SENSOR-11 (Option B).
  - Files: `tools/azu-calibrate` (C++ or `scripts/calibrate.py` with OpenCV), `~/.config/azu/kinect_<serial>.json` {K_ir, K_rgb, dist, T_rgb_ir, depth_a, depth_b, distortion map path}.
  - Notes:
    - Factory defaults on every start via `freenect_copy_registration`: fx_ir = ref_dist/(2·ref_pix) (575.8 on this device), the raw-to-mm table, the baseline (2.4 cm). Log them, keyed by `camera_serial`.
    - Intrinsics: RGB checkerboard; IR via `FREENECT_VIDEO_IR_8BIT`; `stereoCalibrate`; check the 3–4 px IR/depth offset.
    - Depth correction: a wall at 6–8 tape-measured distances, fit `1/z_true = a/z + b`.
    - Mode B: `FREENECT_DEPTH_MM`, IR K, and colour by projecting voxels through `K_rgb·T_rgb_ir`.
  - Acceptance: after calibration, the wall test is ≤ 0.5% error to 3 m; box edges ≤ 0.5%, angles ≤ 0.5°; reprojection RMS < 0.5 px.
  - Effort: L. Depends on: T3.1.

- [ ] **T3.12. Optional depth-distortion LUT**
  - Sources: SOTA-18.
  - Notes: a per-pixel multiplier `m(u, v, z_bin)` learned from a refined flat-wall scan (CLAMS / Herrera), applied during conversion.
  - Acceptance: flat-wall RMS planarity at 2.5 m improves by ≥ 30%.
  - Effort: M. Depends on: T3.11.

- [~] **T3.13. IMU gravity alignment and prior** — first slice done 2026-09-23. Done:
  - accelerometer on every frame: `RawFrame::accel`; KinectSensor polls it live at up to 100 Hz and averages per frame; libfakenect replays 'a' records; azu_replay feeds its samples;
  - `tracking/Gravity.h`: axis map, tilt error, verdicts;
  - relocalization refuses poses > 15 deg off gravity (`AZU_GRAVITY_TILT_DEG`);
  - a tracked Good frame > 15 deg off is graded Poor (not integrated);
  - `tilt_err_deg` trace column;
  - two fixes found on the way: the constant-velocity prediction replayed the relocalization jump, and re-acquisitions now serve a 3-Good-frame probation before integrating (from T4.5).

  Measured on cap_001 (CUDA, 4 runs each, unwarp 33 ms):
  - HEAD-like: 393-672 Good frames, 91-252 deg of yaw; 2 of 4 runs collapse at ~100 deg.
  - Now: 496-675 Good, 169-250 deg; every run recovers; re-acquisitions within 8.1 deg of gravity.
  - Correct poses: tilt error p50 2.3, p90 4.8, p99 6.9 deg.
  - spin360-slow unchanged.

  Not done: the soft roll/pitch residual in ICP, the gravity-aligned world and floor origin, the 15 deg tilted-replay acceptance.
  - Sources: TRACK-15, SENSOR-16 (accelerometer part).
  - Files: `KinectSensor.cpp` (the capture thread polls `freenect_update_tilt_state` and `freenect_get_mks_accel` at 10–20 Hz; `libfreenect.h:495`, `:561`), the tracking prior.
  - Notes:
    - Low-pass filter the accelerometer.
    - At start, rotate the world so −g maps to −Y and put the origin at floor level.
    - Soft roll/pitch residual `(R_cwᵀ·g_world) × ĝ_meas` with weight 1/(0.02 rad)², applied only when ‖a‖ ≈ g ± 0.3.
    - Verify that the fakenect `a-*.dump` files replay the accelerometer (SUSPECTED).
  - Acceptance:
    - A replay tilted by 15°: exported floor normal within 2° of +Y.
    - Corridor probe with synthetic gravity: rotation error < 0.2°.
  - Effort: M. Depends on: T0.8.

- [ ] **T3.14. Rolling-shutter unwarp on by default for Kinect input**
  - Sources: cap_001 field report (above).
  - Files: `include/sensor/RollingShutter.h` (today CPU + OpenMP, ~1 ms per frame, opt-in via `AZU_RS_READOUT_MS`), `PipelineController` (hook before preprocessing), `make_fake_dump` (emulate the readout so synthetic takes exercise it).
  - Notes:
    - Readout ~30 ms, fitted on cap_001 (20-33 ms best). Confirm on a second take.
    - Second pass: re-unwarp with the ICP step, then re-solve, when the step is large.
    - GPU path: unwarp the uploaded raw frame in the conditioner.
    - A hyperparameter, not an environment variable, once on by default. Off for synthetic sources until make_fake_dump emulates the readout.
  - Acceptance:
    - `rolling_shutter_contract` stays green.
    - On cap_001, mean RMS over tracked frames 230-300 at or below 9 mm.
    - spin360-slow no worse (tracked frames, loop check).
  - Effort: S-M. Depends on: none.

### Phase 4: room scale

- [ ] **T4.1. Auto-centred, gravity-aligned volume and preset origins**
  - Sources: PERF-17, FUSE-08 (interim), TRACK-12(3), SOTA-08 (short term), critic Room-preset risk.
  - Files: `FusionUiModel.cpp:141-180`, `TSDFParams`, `PipelineController.cpp:797` (first pose).
  - Notes:
    - At the first frame: `origin = cam_pos − 0.5·L·(1,1,1)` for Room. For object presets: `cam_pos + R·(0,0,d_obj) − L/2`.
    - Per-axis dims (`Eigen::Vector3i`).
    - Gravity-aligned via T3.13 when available.
  - Acceptance: a preset contract checks the camera origin lies in the middle 50% of x and y; the synthetic 360° pan inside a 5 m volume never raycasts empty.
  - Effort: S. Depends on: T0.9.

- [ ] **T4.2. CPU voxel-block hash**
  - Sources: PERF-13, FUSE-08, TRACK-12(1), SOTA-08 (long term), HANG-16 (sparse part).
  - Files: new `include/tsdf/BlockMap.h`, `src/tsdf/BlockVolume.cpp`, behind `IFusionBackend` / the volume interface.
  - Notes:
    - 8³ blocks, key = packed int3 in a uint64 (21 bits per axis), `ankerl::unordered_dense` (or open addressing with a Morton hash) mapping to an index in a pooled `std::vector<Block>`.
    - Block = 512 voxels in the T2.9 layout plus dirty flag, min/max sdf, and last-seen frame.
    - Allocation: per valid pixel (subsampled 2×), a 3D-DDA over the segment [d − τ(d), d + τ(d)]; thread-local key vectors, then sort/unique, then allocate.
    - Integration: the T0.9 kernel per visible block, one block per task.
    - Raycast: two-level (skip absent blocks, cache the current block pointer).
    - Meshing: T2.7 per dirty block, with a halo.
    - Unit tests: 1M random keys for insert, find and collision.
  - Acceptance (G12): synthetic 6×5×2.7 m room with a 360° pan and a 3 m walk: RSS < 1 GB at 1 cm, integration median < 10 ms, mesh bias < 2 mm, completeness > 95%, allocated block count within ±10% of the analytic estimate.
  - Effort: L. Depends on: T0.9, T0.10, T2.7, T2.9.

- [ ] **T4.3. CUDA voxel-block hash**
  - Sources: GPU-11, SOTA-16, SOTA-08.
  - Notes:
    - Device open-addressing hash with `atomicCAS` insertion (or stdgpu), a block pool with a free list, and a visible-block list from frustum culling.
    - One CUDA block (8×8×8 threads) per visible block.
    - Per-block min/max range image for the raycast.
    - GPU marching cubes per dirty block (T2.16).
    - In-house implementation borrowing the nvblox (Apache-2.0) and InfiniTAM *designs*. Do not copy InfiniTAM, ElasticFusion or BundleFusion code (non-commercial licences).
  - Acceptance: synthetic room replay at 30 Hz sustained; VRAM < 1 GB at 1 cm (G5); CPU/GPU parity on block fixtures.
  - Effort: L. Depends on: T4.2 (semantics), T2.13–T2.16.

- [ ] **T4.4. Streaming and range**
  - Sources: FUSE-08 (streaming), TRACK-12(4), GPU-11(2) (host streaming).
  - Notes:
    - Blocks further than 5 m from the camera and idle for N frames move to a host cold store (zstd). Reload them when they re-enter the frustum.
    - Tracking range 4 m, integration ≤ 3.5 m.
  - Acceptance: a synthetic 12 m corridor walk keeps RSS/VRAM bounded (< 1.5× that of a 6 m room), and the final mesh is complete.
  - Effort: M. Depends on: T4.2 or T4.3.

- [ ] **T4.5. Keyframe database and fern relocaliser** (probation after a re-acquisition landed with T3.13)
  - Reach today: the +-10 deg hypothesis grid around the last good pose. At ~15 deg on the synthetic room it re-acquired with the orientation right and the position 21.6 cm off, and that wrong basin passed probation (pipeline_gravity_contract notes).
  - Sources: TRACK-09 (steps 2, 3), SOTA-11.
  - Notes:
    - 80×60 normalised depth (and luma).
    - 500 ferns × 4 binary tests; dissimilarity = 1 − equal/ferns.
    - Add a keyframe when GOOD and the dissimilarity > 0.2 (cap 1000).
    - On loss: take the k = 3 nearest keyframes, raycast the model at each pose, run coarse-to-fine ICP (plus RGB after T3.10), and accept after 3 consecutive GOOD frames.
    - The GUI shows the retrieved keyframe as a ghost.
  - Acceptance:
    - A 1 s blackout plus a 90° jump: recovery within 30 frames, pose error < 3 cm / 2°, and no integration during the gap.
    - Lens covered for 30 frames while moving 30 cm / 20°: recovery within 10 frames after uncovering, and ATE < 10 mm after recovery.
  - Effort: M. Depends on: T0.13, T3.7.

- [ ] **T4.6. Session recording and offline "Refine & Export"**
  - Sources: SOTA-13 (A), TRACK-14, SOTA-14 (offline colour map), G9 (fr1/room refine target).
  - Notes:
    - Always-on session directory: a fakenect-compatible raw dump, a TUM-format trajectory, intrinsics and `reg_info`. Raw depth is about 600 KB per frame; keep every frame at ≤ 3 min, then every third.
    - Refine, v1: offline via Open3D (pip, Python) reconstruction-system defaults:
      - fragments of 100 frames, coloured ICP, FPFH RANSAC global registration;
      - pose graph with a line process;
      - reintegration at 5 mm voxels;
      - optional SLAC;
      - colour map optimisation (Zhou & Koltun 2014).
    - Refine, v2: an in-tree C++ pose graph (Ceres or g2o, SE3 factors with ICP-Hessian information) re-integrating into the T4.2 volume.
  - Acceptance:
    - A synthetic 3 m loop around the room: end-point error < 1 cm after refinement.
    - TUM fr1/room and fr3/long_office: ATE reduced about 2× (G9 refine target).
  - Effort: M (v1), L (v2). Depends on: T1.3, T1.4.

- [ ] **T4.7. Online submaps and loop closure (stretch)**
  - Sources: SOTA-13 (B, C), TRACK-14 (online), GPU-11(7).
  - Notes: InfiniTAM-v3-style submaps: a new local TSDF when the view leaves the current one, relocaliser-based inter-submap loop detection, a pose graph over submap poses, and fusion at mesh time. BundleFusion-style de-/re-integration is an alternative.
  - Acceptance: live fr1/room ATE < 8 cm without the offline step.
  - Effort: L. Depends on: T4.5, T4.6.

### Phase 5: UX

- [ ] **T5.1. Scan UX: tracking light, coverage, speed warnings, auto pause and resume, one workflow**
  - Sources: SOTA-17, HANG-09 (overlay), T0.3 (stalled display).
  - Notes:
    - Green/amber/red from the tiers (T0.12, T3.7), plus an ICP residual heat overlay.
    - Raycast shaded by weight (unobserved red, low yellow), plus a top-down minimap.
    - Speed warning: > 0.3 m/s or > 30°/s shows "slow down".
    - Auto pause on loss, with a keyframe ghost and auto resume.
    - Room-size presets (small/medium/large) set vs and extent.
    - "Start scan → Finish → Refine & Export (GLB/PLY/OBJ)" with progress.
  - Acceptance: a scripted fakenect replay with fast segments makes `FusionUiModel` emit SLOW_DOWN, LOST and RELOCALIZED at the expected frames (deterministic).
  - Effort: M. Depends on: T3.7, T4.5.

- [ ] **T5.2. Kinect LED and tilt**
  - Sources: SENSOR-16 (LED/tilt part).
  - Notes:
    - On the capture thread: `freenect_set_tilt_degs(0)` at start; LED green while tracking, red while lost.
    - Fall back to CAMERA-only if opening the motor fails (models 1473 and K4W).
  - Acceptance: the fakenect/no-motor path does not error; manual check on the device.
  - Effort: S. Depends on: T0.8.

- [ ] **T5.3. Export formats**
  - Sources: FUSE-16.
  - Notes:
    - GLB `COLOR_0` as normalised UNSIGNED_SHORT linear (8 B per vertex instead of 16), plus `KHR_materials_unlit`.
    - PLY comments with `voxel_size`, units and frame convention.
    - Remove the redundant weak-hash weld (the exact-edge weld from T2.7 is canonical).
    - If `extractGlobalPointCloud` survives T0.7, project its points to the zero crossing.
    - Add an OBJ writer for the refine output.
  - Acceptance: glTF-Validator reports 0 errors; the GLB is ≤ 60% of its current size on the sphere fixture.
  - Effort: S. Depends on: T2.7.

### Phase 6: cleanup, style and docs

- [ ] **T6.1. Split `PipelineController`**
  - Sources: REVIEW-15.
  - Notes:
    - Split into `FrameSource`/`DepthSource` (T1.3), `FusionEngine` (T2.11, synchronous `processFrame`), `MeshService` (versioned requests), `PipelineRunner` (threads and lifecycle) and `PreviewPublisher`.
    - Move seam hooks out of production code into `friend class PipelineTestAccess` in `tests/` (T0.1 removed the regex gates that would otherwise block this).
    - One `TrackingPolicy` struct for all constants.
    - Move the GL bootstrap from `main.cpp` (466 lines) to `src/gui/GlBootstrap.cpp`.
    - clang-format the tree.
  - Acceptance: no source file > 600 lines; tests green; no behaviour change (soak and accuracy numbers equal within noise).
  - Effort: L. Depends on: Phase 2.

- [ ] **T6.2. One synchronisation owner**
  - Sources: PERF-19, FUSE-18, HANG-19 (double locking).
  - Notes: the volume is externally synchronised (documented), and `TSDFVolume::mutex_` is removed. After T4.2, brick-level sequence numbers replace it.
  - Acceptance: TSan (noomp) is clean across the full suite.
  - Effort: S. Depends on: T2.2.

- [ ] **T6.3. Docs consolidation and correction**
  - Sources: REVIEW-16, HANG-18, REVIEW-10 (README hardware claims).
  - Notes:
    - README covers usage and build. `docs/ARCHITECTURE.md` covers pipeline, semantics (merge the updated `CANONICAL_SEMANTICS`) and tuning.
    - Move `REGRESSION_REVIEW`, `CAMERA_CONTROLS_UPDATE`, `rocm.md` and `KNOWLEDGE_BASE` to `docs/history/` or delete them.
    - Fix the false statements: "no NVIDIA hardware", "real-time", "30 ms CPU budget", "microseconds / 71.6 minutes", the "atomicAdd kernel" description.
    - Replace performance claims with `bench/results.csv` numbers.
    - Strip "big-fix" and "Todo N" narration from code comments (135 + 171 references) and the 60-line history comment in the gate script. Remove the untracked `.omo/evidence` references.
    - Delete `docs/CUDA_HIP_DEFERRED_CHANGES.md` once every row is closed.
  - Acceptance:
    - `grep -rn "big-fix\|[Tt]odo [0-9]" src include | wc -l` == 0.
    - `grep -rn "71.6 min\|microsecond" include src docs` == 0 for timestamps.
  - Effort: S–M. Depends on: most phases.

- [ ] **T6.4. Style sweep**
  - Sources: TRACK-17, SOTA-19, SENSOR-19 (remaining), GPU-19, HANG-19 (remaining), GAP-09, REVIEW-16 (ELF).
  - Notes:
    - Delete the empty `src/tracking/PyramidLevel.cpp`, or move downsampling into it.
    - Per-level thresholds in `ICPParams`.
    - `error` reported as RMS in mm; fix the `advice` m² comparison.
    - GPU hygiene: duplicate include (`MarchingCubes_cuda.cu:4,10`), the misleading "Image-Centric" comment, dead `R_cw`/`t_cw`, dead `d_rgb`.
    - Fix the 98 warnings.
    - `udev/99-kinect.rules`: `MODE="0660"`.
    - `git rm test_colors` (the ELF) and `test_colors.cpp`, or move the source to `tools/`.
    - Move `SuperResolution` to `tools/`, or delete it.
    - Use zero-copy `freenect_set_depth_buffer` / `freenect_set_video_buffer` double buffering.
  - Acceptance: the `ci` preset builds with `-Werror`; `git ls-files | xargs file | grep ELF` is empty.
  - Effort: S–M. Depends on: T1.6.

## Traceability (every surviving finding → todo)

| Finding(s) | Todo |
|---|---|
| HANG-01, SENSOR-01, SENSOR-02, REVIEW-01 | T0.2 |
| HANG-02, GAP-03 | T0.3 |
| HANG-03 | T0.13 (preview), T0.9/T0.10 (throughput), T1.5 (acceptance) |
| HANG-04, HANG-12 | T0.9 |
| HANG-05 | T0.10, T2.8 |
| HANG-06 | T0.14 (SR), T2.1 (rest) |
| HANG-07 | T2.3 |
| HANG-08 | T0.5 |
| HANG-09 | T0.13 |
| HANG-10 | T0.6 |
| HANG-11 (P2) | T0.8 |
| HANG-13 (reclassified as flake) | T1.1 |
| HANG-14 | T2.4 |
| HANG-15 | T0.7 |
| HANG-16 | T2.7, T4.2 |
| HANG-17 | T2.6 |
| HANG-18 | T0.2, T6.3 |
| HANG-19 | T0.8, T0.13, T6.2, T6.4 |
| HANG regression suite (1–4) | T0.2, T0.4, T0.8, T1.5 |
| PERF-01, PERF-04 | T0.9 (T1.2 tests) |
| PERF-02 | T0.10 (a, b, d), T2.8 (c) |
| PERF-03 | T0.11 |
| PERF-05 | T2.7 |
| PERF-06 | T0.5 |
| PERF-07 | T0.14 (1), T2.1 (2–4) |
| PERF-08 | T2.3 |
| PERF-09 | T1.6 |
| PERF-10 (P2) | T2.4 |
| PERF-11 | T2.10 |
| PERF-12 | T2.9 |
| PERF-13 | T4.2 (T4.1 preset part) |
| PERF-14 | T2.6 |
| PERF-15 | T2.5, T0.13 (duplicate hypothesis) |
| PERF-16 | T0.10 |
| PERF-17 | T4.1 |
| PERF-18 | T2.7 |
| PERF-19 | T6.2 |
| PERF-20 | T1.1, T1.2, T1.5 |
| PERF-21 | T2.11–T2.18, T4.3 |
| GPU-01, GPU-02, GPU-04 | T0.16 |
| GPU-03 | T0.15 (T0.7 worker throws) |
| GPU-05 | T0.9 (CPU), T2.13 (GPU) |
| GPU-06 | T2.13 |
| GPU-07 | T0.7 (stop part), T2.17 |
| GPU-08 | T2.14 |
| GPU-09 | T2.16 |
| GPU-10 | T2.13 (gates, raycast), T3.3 (weights), T3.4 (carving) |
| GPU-11 | T4.3 (T2.9 voxel, T2.16, T2.18, T4.4, T4.7) |
| GPU-12 | T1.7 |
| GPU-13 | T0.16 (domain), T3.2 |
| GPU-14 | T2.18 |
| GPU-15 | T2.15 |
| GPU-16 | T3.8, T2.4 |
| GPU-17 | T1.6 |
| GPU-18 | T2.12 |
| GPU-19 | T6.4, T1.6 |
| GPU-20 (recommendation) | T2.19 |
| GPU-21 (recommendation; claim partly refuted) | T0.15, T2.11–T2.14 |
| TRACK-01 | T0.11, T0.13 (re-raycast on loss), T2.2 |
| TRACK-02 | T0.9 |
| TRACK-03 | T3.1 |
| TRACK-04 | T2.1 |
| TRACK-05 | T2.1 (1, 2), T3.8 (3) |
| TRACK-06 | T0.12 (tiers), T3.7 (degeneracy) |
| TRACK-07 | T2.3, T2.4, T2.5 |
| TRACK-08 | T0.10 (1–4), T2.8 (5) |
| TRACK-09 | T0.13 (1, 4, 5), T4.5 (2, 3) |
| TRACK-10 | T3.9 |
| TRACK-11 | T3.6, T2.5 (double accumulation) |
| TRACK-12 | T4.1, T4.2, T4.4 |
| TRACK-13 | T3.10 |
| TRACK-14 | T4.6, T4.7 |
| TRACK-15 | T3.13 |
| TRACK-16 | T1.2, T1.3, T1.4 |
| TRACK-17 | T0.13, T3.8, T6.4 |
| FUSE-01, FUSE-02, FUSE-03 | T0.9 |
| FUSE-04 | T3.3 |
| FUSE-05 | T3.1, T3.2 |
| FUSE-06 | T0.10, T2.8 |
| FUSE-07 | T0.5 |
| FUSE-08 | T4.1 (interim), T4.2, T4.4 |
| FUSE-09 | T0.10 (sampler min weight), T3.5 |
| FUSE-10 | T3.4 |
| FUSE-11 | T0.10 |
| FUSE-12 | T3.5 |
| FUSE-13 | T2.7 |
| FUSE-14 | T3.3 |
| FUSE-15 | T3.2 |
| FUSE-16 | T5.3 |
| FUSE-17 | T1.2, T1.3 |
| FUSE-18 | T6.2, T2.9 |
| SENSOR-03 (P1) | T2.1 |
| SENSOR-04 | T0.14 |
| SENSOR-05, SENSOR-06, SENSOR-07, SENSOR-08 | T2.1 |
| SENSOR-09 | T3.1 |
| SENSOR-10 | T3.1 (A), T3.11 (B) |
| SENSOR-11 | T3.1 (A), T3.11 (B) |
| SENSOR-12 | T2.1 (RGB filters), T3.2 (AE/AWB) |
| SENSOR-13 | T3.3 |
| SENSOR-14 | T0.8 |
| SENSOR-15 | T3.9 |
| SENSOR-16 | T3.13 (accelerometer), T5.2 (LED/tilt) |
| SENSOR-17 | T0.4, T1.3 |
| SENSOR-18 | T3.11 |
| SENSOR-19 | T0.2 (RGB swap), T0.8, T2.1, T2.3, T6.4 |
| REVIEW-02 | T0.12 |
| REVIEW-03 | T0.6 |
| REVIEW-04 | T0.11 |
| REVIEW-05 | T0.9 |
| REVIEW-06 | T0.14 |
| REVIEW-07 | T2.1 |
| REVIEW-08 | T3.1 |
| REVIEW-09 | T2.2, T0.5 |
| REVIEW-10 | T0.16 (colour), T2.11, T1.7, T4.2/T4.3, T6.3 (README) |
| REVIEW-11 | T1.1 |
| REVIEW-12 | T1.2 |
| REVIEW-13 | T0.1, T1.8 |
| REVIEW-14 | T1.6 |
| REVIEW-15 | T6.1, T2.11 |
| REVIEW-16 | T6.3, T6.4 |
| REVIEW-17 | T0.8 |
| REVIEW-18 | T2.3, T2.4 |
| SOTA-01, SOTA-02 | T0.9 (T3.3 weights) |
| SOTA-03 | T0.10 (stage 1), T2.8 (stages 2, 3) |
| SOTA-04 | T0.10 |
| SOTA-05 | T3.1, T3.2 |
| SOTA-06 | T2.1 |
| SOTA-07 | T3.3 |
| SOTA-08 | T2.9, T4.1, T4.2, T4.3 |
| SOTA-09 | T2.2 |
| SOTA-10 | T3.10 (a), T0.12/T3.7 (b), T3.6 (c) |
| SOTA-11 | T0.13, T4.5 |
| SOTA-12 | T1.3, T1.4 |
| SOTA-13 | T4.6 (A), T4.7 (B, C) |
| SOTA-14 | T3.2 (online), T4.6 (offline); the SR claim is refuted |
| SOTA-15 | T3.5 |
| SOTA-16 | T4.3, T2.15, T2.18 |
| SOTA-17 | T5.1 |
| SOTA-18 | T3.12 |
| SOTA-19 | T0.10 (band defaults), T2.9, T6.4 |
| GAP-01 | T0.15 |
| GAP-02 | T0.7 (T2.12) |
| GAP-04 | Priority order encoded in Phase 0/2; T1.5 |
| GAP-05 | T2.10 |
| GAP-06 | T0.15, T0.14, T2.1 |
| GAP-07 | T0.4, T0.1 (Release gate) |
| GAP-08 | this file's path; pitfall in T0.2 |
| GAP-09 | T6.4, T2.3 (Threads spin box) |
| Critic §2 (configure gates, locking contracts) | T0.1, T1.8, execution rules |

## Verification strategy

- **Per commit:** `scripts/gate.sh`, which runs cpu-release and cpu-debug `ctest -L cpu`, plus `-L gpu` on cuda-release when nvcc exists. Fixes that change locked behaviour carry the rewritten contract.
- **Red-before-green:**
  - T0.2, T0.4, T0.9, T0.10, T0.11 and T1.2 record a failing run on the pre-fix tree in the commit body (test name and assertion).
  - Pure refactors (T2.11, T6.1) record unchanged `accuracy` and `perf` numbers.
- **Phase gates:**

| Gate | Must pass |
|---|---|
| Phase 0 exit | `smoke_fakenect` both lanes, both build types; `kinect_pairing_realtrace_contract`; `pipeline_watchdog_contract`; `pipeline_soak` CPU ≥ 10 Hz, CUDA ≥ 29 Hz; G6 static drift; GAP-02 repro exits rc 0; manual: real Kinect 5 min scan, Stop, Export, on both backends |
| Phase 1 exit | the whole suite green 20× with `--repeat until-fail:20` under `-j32`, asan and tsan-noomp; `eval_tum` running on fr1/xyz |
| Phase 2 exit | G3, G4, G13; 10 min soak (G1) on both lanes; nsys criteria from T2.14 |
| Phase 3 exit | G7, G8, G9 gate values, G10, G11 (manual protocol below) |
| Phase 4 exit | G5, G12; T4.5 relocalisation; fr1/room refine target |

- **Manual real-device protocol (G11), recorded in `bench/manual.md`:**
  - flat wall at 1.0 / 2.0 / 3.0 m (tape) → RANSAC plane distance and RMS;
  - cardboard box of known size → edge lengths and corner angles;
  - checkerboard on the box → colour vs depth edge offset;
  - a 2 min room pan → no Lost, mesh visually closed.

## Benchmark protocol

- **Host:** load average < 2 before start (check `uptime`). Record the CPU governor and `nproc`.
- **Threads:** fix `OMP_NUM_THREADS` (16 and 32) and `OMP_WAIT_POLICY=passive`.
- **GPU:** record `nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv` and `CUDA_VISIBLE_DEVICES`. vLLM and llama-server occupancy changes free VRAM and SM contention, so report it with every number.
- **Stages:** 3 runs; report the median of per-run p50 and p95. Warm-up of 30 frames is excluded.
- **Record per run:** commit, preset, dataset/scene, resolution/voxel, results. Append to `bench/results.csv`.
- **Commit messages** quote only idle-protocol numbers, and label loaded numbers as loaded.

## Commit strategy

- Append-only history. Never amend, rebase, reset away, revert or force-push. **Never push.**
- A mistake is corrected by a new commit that states what was wrong.
- `edit && build && ctest && git commit`. A failed step must never be followed by a commit that claims success.
- Re-read the files a message claims to have changed before committing.
- One logical change per commit; a todo may have several commits. Suggested prefixes: `fix(sensor)`, `fix(tsdf)`, `perf(tsdf)`, `test(harness)`, `feat(cuda)`, `refactor(pipeline)`, `docs(big-fix-two)`.
- Order within Phase 0:
  1. T0.1
  2. T0.2 + T0.3, then T0.4 (it proves the hang fixed)
  3. T0.5, T0.6, T0.7, T0.8 (independent, small)
  4. T0.9 → T0.10 → T0.11 → T0.12 → T0.13
  5. T0.14
  6. T0.15 → T0.16
  - The Phase 0 gate runs before anything is tagged as usable.
- No co-author trailers.
- Mark each closed row in the deferral dossier with its commit hash.

## Risks

| Risk | Mitigation |
|---|---|
| The pairing fix alone turns the hang into a frozen TrackingLost at about 2 Hz | Phase 0 ships pairing, projective integration, the raycast, the ref pose, tiers and relocalization together (critic §4). |
| The new integrator changes `max_weight` semantics (per frame vs about 40 samples per frame) | Retune presets in T0.9. `color_convergence_contract` is re-derived. |
| Front-face raycast plus the loose Room ICP gates (60°, 0.2 m) cause mis-association | T3.6 tightens them to 25° and 0.1 m; interim change in T0.12 if the soak shows it. |
| CUDA VRAM contention from vLLM and llama-server on both cards | T0.15 budget and selection; skip-77 tests; hashing in T4.3 brings it under 1 GB. The dense 256³ at 32 B/voxel (537 MB plus scratch) does not fit reliably today. |
| CUDA enabled by default while its P0 bugs remain (inside-out mesh, corrupt MC rows, ICP curvature, stop crash) | CUDA stays non-default until T0.7, T0.15 and T0.16 pass `-L gpu` and `smoke_fakenect`. |
| REGISTERED depth loses about 13% of valid pixels and adds z-buffer cracks | Accept for v1. Option B (MM plus extrinsics) is in T3.11; measure both on the G11 protocol. |
| Removing the regex gates could let a test bypass production code | The link-level check remains; seam tests link `azu_test_pipeline`. |
| Datasets are large and CI has no network | Download to a cache outside the repo; skip-77; synthetic and generated fakenect tests cover the default gate. |
| GPL-3 contamination from reference SLAM code | Borrow designs only from InfiniTAM, ElasticFusion and BundleFusion (non-commercial). nvblox, meshoptimizer and PoissonRecon are permissive. |
| Performance numbers taken under load mislead | Idle-host benchmark protocol; the critic's re-baseline is the source of truth. |
| The OpenMP budget change has unmeasured in-app impact | Decided by the idle in-pipeline sweep in T2.3; stays P2. |

## Appendix A: refuted or corrected claims (dropped as stated)

| Claim | Verdict | Why |
|---|---|---|
| HANG-13: "`pipeline_state_contract` fails every time in Release" | REFUTED | 10/10 passes on an idle host in two Release trees. It is load-dependent flakiness, handled as REVIEW-11 in T1.1. |
| SENSOR-03: "P0, 120–730 ms preprocessing" | REFUTED as stated | 28.9 ms at 32 threads idle (23.4 ms with SR off). Downgraded to P1; still most of a 33 ms budget. T2.1. |
| GPU-20: "hipcc on PATH at /usr/sbin/hipcc" | REFUTED | The file does not exist; `which hipcc` finds nothing; AUTO resolves to CUDA. The deletion recommendation stands (T2.19). |
| Several reports (HANG-05, etc.): "raycast marches to t_far = 5 m" | CORRECTED | The constructor syncs `max_depth` from hyperparameters (2.5 m default), so at most about 440 steps. The cost is still 156 ms idle. |
| HANG-07, PERF-08, TRACK-07, REVIEW-18: "10× ICP / buildFrameData slowdown from oversubscription" | CORRECTED | Load artifacts. ICP is 8.6 ms idle at 32 threads. The fix is kept at P2 pending an in-pipeline measurement. |
| TRACK-01 as *the* hang | CORRECTED | The primary cause is pairing (reproduced). TRACK-01 is a real secondary cause that bites when integration lags. |
| HANG-11 / SENSOR-14 / REVIEW-17: "stop can block up to 60 s" | Unlikely (PLAUSIBLE, low) | libusb wakes the event waiters promptly. The fix is kept as hygiene at P2. |
| REVIEW-02: "the success rule is the cause of false loss" | PLAUSIBLE only | The log line is more likely a symptom of the wrong ref pose plus the model bias. The tiering in T0.12 addresses both views. |
| GPU-21: "the CPU lane cannot be real-time" | CORRECTED | True for current code. The projective integrator (3–7 ms) plus the fixed raycast make ≥ 20 Hz feasible. The CPU stays the reference lane. |
| SOTA-14: "SR-upscaled RGB is used for texturing" | REFUTED | The only consumer is commented out (`PipelineController.cpp:782-783`), and SR output is discarded. |
| PERF-10 at P1 | Downgraded to P2 | Measured 3–4 ms idle. |
| Bias magnitudes (−14.4, +4.5, +6.7 mm; 20–54 mm drift) | Reconciled | They differ in sign convention and scene; all are 0.5–1.5 voxel with the same root cause and fix (T0.9). |
| TRACK-03 "fx should be about 576" | Scoped | True only for 11-bit or MM depth. With REGISTERED (T3.1), RGB K (about 525) is correct. Read the value from the device or calibration, never hard-code it. |

## Appendix B: conflicting proposals and chosen resolution

| Topic | Options | Chosen |
|---|---|---|
| Tracking acceptance | loosen (REVIEW-02) vs tighten (TRACK-06, SOTA-10) | GOOD/POOR/FAILED tiers; Lost after 3 consecutive FAILED (T0.12, T3.7) |
| Depth mode | REGISTERED + RGB K vs MM + IR K + voxel→RGB projection | REGISTERED first (T3.1), MM later (T3.11) |
| Threading | sequential worker vs 3 threads with fixed locks | Sequential fusion worker plus async mesher (T2.2) |
| Primary lane | CUDA now (GPU-21, REVIEW-10) vs CPU first | Fix the CPU canonical algorithms first (Phase 0). CUDA becomes default only after T0.7, T0.15 and T0.16 pass. Hash VRAM in T4.3. |
| Pair window | 16.7 / 17 / 20 ms | 17 ms with nearest-neighbour depth-led pairing; moot by construction |
| Thread budgets | 6/8/2, 10/16/4, 8/16/4 | Measured on an idle host in-pipeline (T2.3), exposed as a runtime setting |
| Voxel convention | corner vs centre | Corner (`origin + i·vs`) everywhere (T0.9) |
