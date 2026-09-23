# Record a capture, replay it, study it

## 1. Record (app closed: the Kinect is exclusive)

```bash
mkdir -p ~/kinect-rec
fakenect-record ~/kinect-rec/spin360-slow     # Ctrl-C to stop; refuses an existing dir
```

~46 MB/s (30 s ≈ 1.4 GB). Each take holds raw 11-bit depth (`d-*.pgm`), RGB
(`r-*.ppm`), accelerometer samples (`a-*.dump`, ~300 Hz) and the unit's factory
calibration (`device.json`). Timestamps are the device's 60 MHz ticks.

Spin protocol: stand still 3 s facing something with shape (furniture, a corner),
turn clockwise at a steady pace (≈30 s per turn, then ≈15 s), stop facing the
start and hold 3 s, Ctrl-C. Starting and ending on the same view makes the loop
check meaningful.

## 2. See it in the GUI at recorded speed

```bash
FAKENECT_LOOP=0 LD_PRELOAD=/usr/local/lib/fakenect/libfakenect.so \
  FAKENECT_PATH=~/kinect-rec/spin360-slow ./build-cuda/KinectFusionQt
```

Pick the Room preset ("Stand in the middle, turn around"): its volume surrounds
the camera. Object presets put the volume in front of the camera and cannot hold
a full turn.

## 3. Replay offline, deterministically

```bash
./build-cuda/tools/azu_replay ~/kinect-rec/spin360-slow --preset room --out /tmp/spin-cuda
./build-cuda/tools/azu_replay ~/kinect-rec/spin360-slow --preset room --backend cpu --out /tmp/spin-cpu
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

`build-b2-cpu/tests/make_fake_dump <dir> [frames] [pan|spin] [fx] [noise]` writes
a recording of the synthetic room (`spin`: a full turn at 1.5°/frame), optionally
rendered with a given focal length (plus a matching `device.json`) and Kinect-like
depth noise. The end pose of a synthetic spin is exact drift.
