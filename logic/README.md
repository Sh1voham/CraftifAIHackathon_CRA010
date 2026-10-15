# SafeStride decision layer (`logic/`)

CPU-only Python that turns PipeGen's per-frame body keypoints into fall alerts
(MQTT + `incidents.jsonl`). Perception lives in PipeGen; this package only
makes decisions. It is a screening aid, not a medical device.

```
keypoints (JSONL / stdin / folder / MQTT)
  -> io.read_keypoints          (the only PipeGen-format code)
  -> io.select_person           (follow one person)
  -> filters.KeypointCleaner    (confidence gate, outliers, gap repair, One Euro, L_ref)
  -> features.FeatureTracker    (torso angle phi, hip drop speed v, hip height h, AR)
  -> fall.FallDetector          (STANDING/FALLING/DOWN/FALL_CONFIRMED + PROLONGED_DOWN)
  -> run.Runner                 (incidents.jsonl, MQTT safestride/alerts, stdout, blurred evidence frame)
  -> gait / tug / risk          (MQTT safestride/risk: speed, cadence, CV, symmetry, TUG, risk band, drift)
```

## Install

```bash
pip install -r logic/requirements.txt
```

Allowed packages only: numpy, scipy, opencv-python, paho-mqtt, pytest.

## Tests

```bash
python -m pytest logic/tests -q
```

## Replay a clip

```bash
python -m logic.run --input clip.jsonl --mqtt localhost:1883 --camera builtin0 --model pose-fp16
```

- `--realtime` replays at the original timestamps (good for demos).
- `--start-wallclock 2026-10-15T20:41:00+05:30` gives reproducible `ts` values (byte-identical output).
- `--video clip.mp4` saves a **blurred** frame for each alert under `blurred/`. Without it only keypoints are kept, which is the default and the privacy story for the pitch.
- `--config logic/config.tuned.json` uses tuned thresholds.
- Events are printed to stdout as JSON lines; logs go to stderr.

Watch MQTT in another terminal:

```bash
mosquitto_sub -h localhost -t "safestride/#" -v
```

MQTT is optional. If the broker is down, the detector logs one warning, keeps writing `incidents.jsonl`, and retries in the background.

## Live mode (PipeGen output)

```bash
# PipeGen app prints JSON lines on stdout
pipegen-app | python -m logic.run --input - --mqtt localhost:1883

# PipeGen appends to a JSONL file
python -m logic.run --input /path/keypoints.jsonl --follow --mqtt localhost:1883

# PipeGen writes one JSON file per frame into a folder
python -m logic.run --input /path/frames_dir/ --mqtt localhost:1883

# PipeGen publishes keypoints over MQTT
python -m logic.run --input mqtt://localhost:1883/pipegen/keypoints --mqtt localhost:1883
```

Add `--record session.jsonl` to save every received frame in replay format. You can label those recordings later and use them with `eval`.

**Demo note:** lying on the floor for 10 s or more raises `PROLONGED_DOWN` even without a fast fall. Don't lie down deliberately for more than a few seconds unless that is the point of the demo.

## Gait, TUG and risk band

`run.py` also computes walking speed, cadence, step-time CV, left/right symmetry and (when a sit–walk–turn–sit sequence is seen) TUG time. It publishes a LOW / MODERATE / HIGH risk band to `safestride/risk`.

```bash
python -m logic.run --input - --mqtt localhost:1883 --person alice --calib 412,700,892,700,3.0
```

- `--calib x1,y1,x2,y2,metres` gives two floor-marker pixels a known distance apart. Without it, speed uses a rough torso = 0.5 m scale and is reported with `calibrated=false`.
- `--person` keeps a walking-speed baseline in `logic/baselines.json` and flags drift: speed below 0.8 × baseline, or z < −2 over the last 5+ sessions.
- `--no-gait` turns this off.
- Walk side-on across the frame. Turns are discarded, and steps are pooled across passes, so several short passes work.

The scoring thresholds are illustrative design choices, not a validated clinical model.

## Synthetic data, evaluation and tuning

```bash
python -m logic.synth --out data/synth --n 200 --seed 0     # labelled clips + labels.json
python -m logic.eval --data data/synth                      # confusion matrix, F1, FA/h, delays, out/eval_results.csv
python -m logic.eval --data data/synth --tune               # grid search -> logic/config.tuned.json
python -m logic.eval --data data/synth --bench              # frames per second on one core
```

`--data` can be repeated, for example to pool real recordings with synthetic ones. For tuning, real recordings take priority over synthetic ones. A clip folder holds `<name>.jsonl` files (replay format) plus `labels.json`:

```json
{"fall_01": {"class": "fall", "variant": "forward", "expected_event": "FALL_CONFIRMED", "t_fall_start": 812.4},
 "sit_03":  {"class": "sit_fast", "variant": "chair", "expected_event": null, "t_fall_start": null}}
```

- `t_fall_start` uses the same clock as the frames' `t`.
- `expected_event` is `FALL_CONFIRMED` (fast fall), `PROLONGED_DOWN` (slow lowering, stays down) or `null`.
- Split, grid and match window live under `eval` in `config.json`.

## Data contract

Replay format, one frame per line, pixels, y pointing down, COCO-17 names:

```json
{"t": 12.345, "persons": [{"id": 0, "bbox": [x1, y1, x2, y2], "kp": {"l_shoulder": [412.0, 220.5, 0.91]}}]}
```

- `id` and `bbox` are optional.
- A missing keypoint may be absent or `[0, 0, 0]`.
- `kp` may also be a list of 17 `[x, y, c]` triples or 51 flat numbers in COCO order.

See `docs/sample_frames.jsonl` and `docs/sample_incidents.jsonl` at the repo root for examples.

**At the real-data handoff, change only `_from_pipegen()` and `PIPEGEN_NAME_MAP` in `io.py`.**

## Configuration

Every tunable number is in `logic/config.json`, loaded into dataclasses by `logic/config.py`. Unknown keys raise an error. The ref/bad values and weights used for risk scoring are illustrative design choices, not a validated clinical model.
