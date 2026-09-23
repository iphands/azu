# Capture a Kinect take and trim it

Step by step: record a raw take with `fakenect-record`, then cut it down to the
still holds with `azu_trim`. The result is a normal recording for `azu_replay`
and the GUI replay. Full reference: `docs/RECORD_AND_REPLAY.md`.

## 0. Once: build the tools

```bash
cd ~/prog/slop/kin/vendor/azu
./scripts/build.sh --cuda        # builds build-cuda/KinectFusionQt and build-cuda/tools/{azu_trim,azu_replay}
mkdir -p ~/kinect-rec            # fakenect-record only creates the LAST directory of its path
```

If the parent directory is missing, `fakenect-record` prints
`Error: Cannot open file [.../INDEX.txt]` and then segfaults. Create it and rerun.

## 1. Capture

Close KinectFusionQt first: the Kinect can only be opened by one program.

```bash
fakenect-record ~/kinect-rec/<take>       # e.g. ~/kinect-rec/spin360-02
```

- Use a new name for each take. The recorder refuses a directory that already has an `INDEX.txt`.
- Disk use is about 46 MB/s, so a 60 s take is about 2.8 GB.

Protocol (handheld is fine):

1. Start the recorder, pick up the camera and get in position quickly.
2. **Hold still for at least 10 s**, pointed at something with shape (furniture, a corner).
3. Do the capture, e.g. turn a full circle clockwise in about 30 s, at chest height.
4. **Hold still for at least 10 s** facing where you started.
5. Reach for the keyboard and press **Ctrl-C** (the recorder only cleans up on SIGINT).

You get:
- `d-*.pgm`: raw 11-bit depth;
- `r-*.ppm`: RGB;
- `a-*.dump`: accelerometer, about 300 Hz;
- `device.json`: this unit's factory calibration.

## 2. Review where the holds are

```bash
./build-cuda/tools/azu_trim ~/kinect-rec/<take> --dry-run
```

It prints a 0.5 s timeline of the first and last 20 s, then the holds and the cut.
Each timeline row shows:
- time;
- depth-change median, with a `#` bar;
- accelerometer std in g, with an `=` bar;
- the share of still frames;
- `<< cut` where a cut lands.

A frame counts as still only when both are low:

| Signal | Still | Moving |
|---|---|---|
| depth change | < 0.015 (handheld still: 0.002-0.009) | turning 0.03-0.25 |
| accelerometer std | < 0.025 g (handheld still: 0.013-0.022 g) | turning 0.03-0.07 g; a keyboard reach 0.07-0.09 g |

The first hold must begin within the first 20 s and the last must end within
the last 20 s; each must last at least 3 s. Cuts go 0.25 s inside the holds.

## 3. Trim

```bash
./build-cuda/tools/azu_trim ~/kinect-rec/<take>
```

This writes `~/kinect-rec/<take>-trimmed`:
- `INDEX.txt` with only the entries between the cuts;
- those files as hardlinks (no extra disk space);
- `device.json`;
- `trim.json`, which records the cut, the holds and the thresholds used.

The original take is not modified. It keeps both holds (the start hold anchors
the model, the end hold is the loop-check reference) and drops getting into
position and the keyboard reach.

If it says **`no trim`** (exit code 3, nothing written), a hold was shorter than 3 s
or not found. Check the timeline, then do one of these:

```bash
./build-cuda/tools/azu_trim ~/kinect-rec/<take> --min-still 1.5     # accept shorter holds
./build-cuda/tools/azu_trim ~/kinect-rec/<take> --start 4.2 --end 51.0   # cut by hand (seconds, from the timeline)
```

Other knobs: `--search S` (how far from each end to look, default 20), `--depth-thresh F`,
`--accel-thresh G`, `--margin S`, `--out DIR`. To redo a trim, delete the `-trimmed`
directory first: azu_trim will not overwrite a recording.

## 4. Use the trimmed take

```bash
# Replay offline, deterministic, both backends
./build-cuda/tools/azu_replay ~/kinect-rec/<take>-trimmed --preset room --out /tmp/<take>-cuda
./build-cuda/tools/azu_replay ~/kinect-rec/<take>-trimmed --preset room --backend cpu --out /tmp/<take>-cpu
scripts/trace_report.py /tmp/<take>-cuda /tmp/<take>-cpu -o /tmp/<take>.html

# Watch it in the GUI at recorded speed (pick the Room preset)
FAKENECT_LOOP=0 LD_PRELOAD=/usr/local/lib/fakenect/libfakenect.so \
  FAKENECT_PATH=~/kinect-rec/<take>-trimmed ./build-cuda/KinectFusionQt
```

## Quick checklist

- [ ] `mkdir -p ~/kinect-rec`, app closed
- [ ] `fakenect-record ~/kinect-rec/<new-name>`
- [ ] get in position -> hold 10 s -> capture -> hold 10 s -> Ctrl-C
- [ ] `azu_trim <take> --dry-run`: both holds found?
- [ ] `azu_trim <take>` -> `<take>-trimmed`
- [ ] `azu_replay <take>-trimmed --preset room`
