# Record a capture, replay it, study it

## 1. Record (app closed: the Kinect is exclusive)

```bash
mkdir -p ~/kinect-rec                          # fakenect-record only creates the last
fakenect-record ~/kinect-rec/spin360-slow      # directory; it refuses an existing take
```

~46 MB/s (30 s ≈ 1.4 GB). Each take holds raw 11-bit depth (`d-*.pgm`), RGB
(`r-*.ppm`), accelerometer samples (`a-*.dump`, ~300 Hz) and the unit's factory
calibration (`device.json`). Timestamps are the device's 60 MHz ticks.

Protocol (handheld is fine):

1. start the recorder, get in position quickly
2. hold still for at least 10 s
3. do the capture (e.g. turn a full circle in ~30 s); keep every motion at
   about 30 deg/s or slower, including up/down sweeps (see below)
4. hold still for at least 10 s
5. move / reach for the keyboard and stop with Ctrl-C

Speed matters. The Kinect v1 depth camera is rolling shutter and the tracker
follows frame to frame, so fast motion both bends each frame and leaves too
little overlap. A slow turn at ~12 deg/s (spin360-slow) tracked ~358 deg and
lost it only at the loop closure; a photosphere-style take with up/down sweeps
at 60-100 deg/s (peaks 150-180 deg/s, cap_001) lost tracking after ~90 deg. For
a room with up/down sweeps, take one floor-to-ceiling sweep in 4-5 s, pause at
the top and bottom, turn ~20-30 deg between sweeps, and expect 2-3 minutes for
the whole room.

## 1b. Trim to the holds

```bash
./build-cuda/tools/azu_trim ~/kinect-rec/spin360-slow --dry-run   # review
./build-cuda/tools/azu_trim ~/kinect-rec/spin360-slow             # writes spin360-slow-trimmed
```

azu_trim finds the first and last still hold (depth change AND accelerometer
both quiet for >= 3 s, searched in the first/last 20 s), prints a 0.5 s timeline
of both ends with the cut marked, and writes `<take>-trimmed`: the entries between
the cuts as hardlinks, `device.json`, and `trim.json`. The positioning motion and
the keyboard reach are gone; the holds are kept (the start hold anchors the model,
the end hold is the loop-check reference). The original take is not modified.
If a hold is too short it reports the best candidate and writes nothing: rerun
with `--min-still 1.5`, or cut by hand with `--start S --end S` (seconds).
Use the trimmed take everywhere below.

## 2. See it in the GUI at recorded speed

```bash
FAKENECT_LOOP=0 LD_PRELOAD=/usr/local/lib/fakenect/libfakenect.so \
  FAKENECT_PATH=~/kinect-rec/spin360-slow-trimmed ./build-cuda/KinectFusionQt
```

Pick the Room preset ("Stand in the middle, turn around"): its volume surrounds
the camera. Object presets put the volume in front of the camera and cannot hold
a full turn.

## 3. Replay offline, in lockstep

```bash
./build-cuda/tools/azu_replay ~/kinect-rec/spin360-slow-trimmed --out /tmp/spin-cuda
./build-cuda/tools/azu_replay ~/kinect-rec/spin360-slow-trimmed --backend cpu --out /tmp/spin-cpu
scripts/trace_report.py /tmp/spin-cuda /tmp/spin-cpu -o /tmp/spin.html
```

Lockstep: every frame is fully tracked, integrated and raycast before the next,
so no frame is dropped and runs compare. CUDA runs are not bit-repeatable: the
GPU ICP sums floats in a varying order, and near a tracking loss runs diverge.
On cap_001, 4 identical runs tracked 89-359 deg of yaw after the same first loss.
Compare several runs, or only up to the first loss (big-fix-two T2.15).
Outputs per run: `summary.json`, `frames.csv`
(pose, accelerometer gravity, cumulative yaw, tilt error), `trace.csv` (per-frame
ICP counters, grades, timings), `mesh.ply`. The summary reports frames by grade,
lost frames, yaw tracked about gravity, tilt vs the accelerometer, a loop check
(last frame against a model of the first 30 frames, observable directions only)
and the end pose relative to the start.

The Room preset (384^3 x 2 cm centred on the camera, depth to 4 m) is the
default. Options: `--backend cpu|cuda`, `--preset room|helmet|chair|human|none`
(`none`: pipeline defaults, a 2.56 m box in front of the camera, which cannot
hold a turn), `--volume front|centred --res N --voxel M`,
`--min-depth/--max-depth`, `--intrinsics device|legacy` (device.json vs the old
525 px), `--max-frames N`.

## 4. A/B switches (environment, GUI and replay)

| Variable | Effect |
|---|---|
| `AZU_TRACE=<file>` | per-frame trace CSV from any run (GUI too) |
| `AZU_INTRINSICS=legacy` | force the old 525 px depth intrinsics |
| `AZU_PREPROCESS=minimal` | depth: spatial median only (no hole fill, guided filter, EMA) |
| `AZU_MOTION_MODEL=velocity` | keep velocity through failed frames |
| `AZU_DEGENERACY_REL=<ratio>` | unobservable-motion threshold (default 5e-3, 0 = off) |
| `AZU_RS_READOUT_MS=<ms>` | rolling-shutter unwarp of each depth frame with the predicted motion (try 30; off by default) |
| `AZU_CUDA_DEVICE=<n>` | pick the GPU |

## 5. Synthetic recordings

`build-b2-cpu/tests/make_fake_dump <dir> [frames] [pan|spin|protocol] [fx] [noise]`
writes a recording of the synthetic room (`spin`: a full turn at 1.5°/frame;
`protocol`: positioning, hold, turn, hold, reach with known boundaries), optionally
rendered with a given focal length (plus a matching `device.json`) and Kinect-like
depth noise. The end pose of a synthetic spin is exact drift.
