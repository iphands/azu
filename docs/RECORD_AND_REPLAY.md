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
3. do the capture (e.g. turn a full circle in ~30 s)
4. hold still for at least 10 s
5. move / reach for the keyboard and stop with Ctrl-C

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

## 3. Replay offline, deterministically

```bash
./build-cuda/tools/azu_replay ~/kinect-rec/spin360-slow-trimmed --preset room --out /tmp/spin-cuda
./build-cuda/tools/azu_replay ~/kinect-rec/spin360-slow-trimmed --preset room --backend cpu --out /tmp/spin-cpu
scripts/trace_report.py /tmp/spin-cuda /tmp/spin-cpu -o /tmp/spin.html
```

Lockstep: every frame is fully tracked, integrated and raycast before the next,
so runs repeat and compare. Outputs per run: `summary.json`, `frames.csv`
(pose, accelerometer gravity, cumulative yaw, tilt error), `trace.csv` (per-frame
ICP counters, grades, timings), `mesh.ply`. The summary reports frames by grade,
lost frames, yaw tracked about gravity, tilt vs the accelerometer, a loop check
(last frame against a model of the first 30 frames, observable directions only)
and the end pose relative to the start.

Options: `--backend cpu|cuda`, `--preset helmet|chair|room|human`,
`--volume front|centred --res N --voxel M`, `--min-depth/--max-depth`,
`--intrinsics device|legacy` (device.json vs the old 525 px), `--max-frames N`.

## 4. A/B switches (environment, GUI and replay)

| Variable | Effect |
|---|---|
| `AZU_TRACE=<file>` | per-frame trace CSV from any run (GUI too) |
| `AZU_INTRINSICS=legacy` | force the old 525 px depth intrinsics |
| `AZU_PREPROCESS=minimal` | depth: spatial median only (no hole fill, guided filter, EMA) |
| `AZU_MOTION_MODEL=velocity` | keep velocity through failed frames |
| `AZU_DEGENERACY_REL=<ratio>` | unobservable-motion threshold (default 5e-3, 0 = off) |
| `AZU_CUDA_DEVICE=<n>` | pick the GPU |

## 5. Synthetic recordings

`build-b2-cpu/tests/make_fake_dump <dir> [frames] [pan|spin|protocol] [fx] [noise]`
writes a recording of the synthetic room (`spin`: a full turn at 1.5°/frame;
`protocol`: positioning, hold, turn, hold, reach with known boundaries), optionally
rendered with a given focal length (plus a matching `device.json`) and Kinect-like
depth noise. The end pose of a synthetic spin is exact drift.
