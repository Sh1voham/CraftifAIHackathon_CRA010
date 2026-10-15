# SafeStride decision layer (`logic/`)

This is CPU-only Python. It turns the per-frame body keypoints from the PipeGen pose pipeline into:
- fall alerts (`FALL_CONFIRMED`, `PROLONGED_DOWN`, `RECOVERED`), written to `incidents.jsonl` and MQTT;
- gait metrics (speed, cadence, step-time CV, left/right symmetry, TUG time);
- a LOW / MODERATE / HIGH fall-risk band.

Perception lives in PipeGen; this package only makes decisions. It is a screening aid, not a medical device.

```
keypoints (stdin / growing JSONL / per-frame JSON folder / MQTT)
  -> io.adapt_stream / auto_adapt  (layout, timestamp unit, coordinates: the only vision-format code)
  -> io.select_person              (follow one person)
  -> filters.KeypointCleaner       (confidence gate, outliers, gap repair, One Euro, L_ref)
  -> features.FeatureTracker       (torso angle phi, hip drop speed v, hip height h, AR)
  -> fall.FallDetector             (STANDING/FALLING/DOWN/FALL_CONFIRMED + PROLONGED_DOWN)
  -> gait / tug / risk             (speed, cadence, CV, symmetry, TUG, risk band, baseline drift)
  -> run.Runner                    (incidents.jsonl, MQTT safestride/alerts|risk|status, blurred evidence frame)
```

See `docs/ARCHITECTURE.md` for the full data flow and `NOTES.md` for decisions and limitations.

Every command below works in **PowerShell** and in **bash** (Git Bash, Linux) unless shown twice.

## Install and test

```bash
pip install -r logic/requirements.txt
python -m pytest logic/tests -q
```

The allowed packages are numpy, scipy, opencv-python, paho-mqtt and pytest. The tests need no MQTT broker and no camera.

## Live demo (one command)

```bash
python -m logic.demo --selftest
```

`--selftest` replays a built-in synthetic fall and a fast sit through the whole stack and prints `SELFTEST PASS` or `SELFTEST FAIL`. It also checks MQTT delivery when a broker answers on `localhost:1883`.

```bash
# PipeGen appends JSON lines to a file
python -m logic.demo --source tail:C:/pipegen/out/keypoints.jsonl --camera builtin0 --model pose-fp16
# PipeGen writes one JSON file per frame
python -m logic.demo --source folder:C:/pipegen/out/frames
# PipeGen publishes keypoints on MQTT
python -m logic.demo --source mqtt:localhost:1883/pipegen/keypoints
# Stage fallback: replay a recorded clip at its real speed with an overlay window
python -m logic.demo --source file:docs/demo_fall.jsonl --realtime --overlay
python -m logic.demo --source file:docs/demo_walk.jsonl --realtime --overlay --person alice
```

- The demo prints one status line per second (state, torso angle, hip speed, hip height, risk band) and a `***` line for every event.
- Alerts go to `incidents.jsonl` and MQTT `safestride/alerts`. Use `--no-mqtt` without a broker; otherwise a missing broker just logs one warning.
- `--overlay` opens an OpenCV window. Add `--video clip.mp4` to draw on the video; without it you get a dark keypoint-only canvas. Press `q` to quit.
- Input format options (also on `run`, `overlay`):
  - `--ts-unit {auto,s,ms,us,ns}`: auto infers the unit from the median frame interval; DeepStream PTS is usually ns.
  - `--coords {auto,px,norm}` with `--frame-size 1280x720`: normalised coordinates need a frame size.
  - `--fps 30`: used only when frames carry frame numbers instead of timestamps.
  - `--kp-map map.json`: source index or name → COCO-17 name, for non-COCO models.
- The detected layout, timestamp unit and coordinate mode are logged once at startup. Run with `--log-level INFO` to see them.

**Demo note:** lying on the floor for 10 s or more raises `PROLONGED_DOWN` even without a fast fall. Don't lie down deliberately for long unless that is the point.

## Watch MQTT

```bash
mosquitto_sub -h localhost -t "safestride/#" -v
```

Topics:
- `safestride/alerts`: fall events, same JSON as `incidents.jsonl`.
- `safestride/risk`: gait and risk summaries.
- `safestride/status`: heartbeat every 5 s.

## Replay and pipe (scriptable runner)

```bash
python -m logic.run --input clip.jsonl --mqtt localhost:1883 --camera builtin0 --model pose-fp16
python -m logic.run --input clip.jsonl --start-wallclock 2026-10-15T20:41:00+05:30   # byte-identical output
```

Pipe frames in from stdin:

```bash
# bash
cat clip.jsonl | python -m logic.run --input - --mqtt localhost:1883
```

```powershell
# PowerShell
Get-Content clip.jsonl | python -m logic.run --input - --mqtt localhost:1883
```

Other `run` options:
- `--follow`: tail a growing file.
- `--record session.jsonl`: save the received frames in replay format, for labelling later.
- `--video clip.mp4`: save a **blurred** frame for each alert under `blurred/`.
- `--person alice --calib x1,y1,x2,y2,metres`: gait baseline and floor calibration.
- `--no-gait`: turn off gait/risk analysis.

## Evaluation

Synthetic benchmark:

```bash
python -m logic.synth --out data/synth --n 200 --seed 0
python -m logic.eval --data data/synth
python -m logic.eval --data data/synth --tune      # grid search -> logic/config.tuned.json
python -m logic.eval --data data/synth --bench     # frames per second on one core
```

Real recordings, named by class (`fall_01`, `sit_01`, `bend_01`, `lie_slow_stay_01`, `lie_slow_recover_01`, `walk_01`, `tug_01`, …):

```bash
python -m logic.eval --data data/real --labels-from-filenames
python -m logic.eval --data data/real --labels-from-filenames --tune
```

- Labels come from the name: `fall_*` expects `FALL_CONFIRMED`, `lie_slow_stay_*` expects `PROLONGED_DOWN`, everything else expects no event.
- A **margin report** shows, per clip and per class, the peak hip speed relative to V_DROP, peak torso angle, minimum hip height and longest time down. Use it to place thresholds between the fall and non-fall groups.
- `--tune` on real clips uses a coarse grid. It never writes `config.tuned.json` from fewer than 10 real clips; it prints a warning instead.

Regenerate the results document (no hand-typed numbers):

```bash
python -m logic.make_results            # writes docs/RESULTS.md; includes data/real if present
```

## Overlay and FP16/INT8 comparison

```bash
python -m logic.overlay --video clip.mp4 --input clip.jsonl                          # window
python -m logic.overlay --video clip.mp4 --input clip.jsonl --headless --save demo.mp4  # people blurred
python -m logic.compare_precision --a runs/fp16/ --b runs/int8/ --labels runs/labels.json --out precision.md
```

`compare_precision` aligns the two runs by timestamp and reports:
- the keypoint error normalised by torso length, overall and per keypoint;
- the share of keypoints whose confidence drops by more than 0.2;
- the fall alerts each run produces.

Latency, FPS and power come from PipeGen's speed reports.

## Data contract

Replay format, one frame per line, pixels, y pointing down, COCO-17 names:

```json
{"t": 12.345, "persons": [{"id": 0, "bbox": [x1, y1, x2, y2], "kp": {"l_shoulder": [412.0, 220.5, 0.91]}}]}
```

`auto_adapt` also accepts:
- `objects` / `detections` / `people` lists;
- keypoints as a dict by name (`left_shoulder` style names work), a list of 17 `[x, y, c]` triplets, a flat list, or `x`/`y`/`score` arrays;
- boxes as `[x, y, w, h]` or `rect_params`;
- `track_id` / `object_id` as the person id;
- `ts` / `timestamp` / `pts` / `frame_num` as the time field.

Examples: `docs/sample_frames.jsonl`, `docs/sample_incidents.jsonl`.

**At the real-data handoff, only `logic/io.py` may change.** Run one sample through `python -m logic.demo --source file:sample.jsonl --log-level INFO` and read the detected layout line.

## Configuration

Every tunable number is in `logic/config.json`, loaded into dataclasses by `logic/config.py`. Unknown keys raise an error. The risk-score ref/bad values and weights are illustrative design choices, not a validated clinical model.
