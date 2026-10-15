# SafeStride architecture

SafeStride does fall detection and gait-based fall-risk screening from one fixed camera, on an edge GPU. Perception, meaning the pose model, its TensorRT engine and the DeepStream app, is built and run in **CraftifAI PipeGen**. The repo's `logic/` package only turns PipeGen's keypoints into decisions.

## Data flow

```
 Camera (built-in /dev/video0, USB, RTSP or a video file)
   |
   v
 PipeGen-generated DeepStream app                      [pipeline side]
   - TensorRT pose engine (FP16; INT8 for the bonus)
   - per-frame body keypoints as JSON (file / folder / stdout / MQTT)
   |
   v
 logic/io.py           transports: tail_jsonl | watch_folder | mqtt_frames | stdin | file
                       adapt_stream/auto_adapt: layout, timestamp unit, coordinate mode,
                       keypoint names -> canonical Frame (seconds, pixels, COCO-17)
                       select_person: follow one person (nearest mid-hip)
   |
   v
 logic/filters.py      confidence gate (c < 0.3 -> missing); outlier rejection scaled by
                       elapsed time; short-gap repair (<= 5 frames, linear in t);
                       One Euro smoothing (real dt); L_ref = median torso length over 2 s
   |
   v
 logic/features.py     torso angle phi; hip drop speed v (torso lengths/s); hip height h above
                       the learned floor line; bbox aspect ratio
   |
   +--> logic/fall.py  STANDING -> FALLING -> DOWN -> FALL_CONFIRMED  (fast fall, ~3 s hold)
   |                   STANDING -> PROLONGED_DOWN                     (slow lowering, 10 s hold)
   |                   -> RECOVERED when upright again
   |
   +--> logic/gait.py  steady straight walking windows; speed (calibrated or rough); steps from
   |    logic/tug.py   ankle separation peaks; cadence, step-time CV, symmetry; TUG phases
   |    logic/risk.py  weighted sub-scores -> LOW / MODERATE / HIGH; per-person speed baseline
   |
   v
 logic/run.py / demo.py
   - incidents.jsonl (one JSON object per event, fsync per write)
   - MQTT safestride/alerts (QoS 1), safestride/risk, safestride/status (heartbeat)
   - optional evidence frame, blurred by logic/privacy.py before it is written
   - optional OpenCV overlay (logic/overlay.py)
```

All timing comes from the frame timestamps `t`. The algorithm modules never read the wall clock and never assume a frame rate, so a replay gives byte-identical output. Wall-clock time appears only at the output boundary (`incidents.py`, `run.py`, `demo.py`).

## Module responsibilities

| Module | Responsibility |
|---|---|
| `config.py`, `config.json` | Every tunable number, as frozen dataclasses; unknown keys are rejected |
| `io.py` | Replay JSONL, live transports, **all vision-format assumptions** (`auto_adapt`), person selection |
| `filters.py` | Keypoint cleaning, One Euro filter, body scale L_ref |
| `features.py` | Per-frame fall features, floor-line estimation |
| `fall.py` | Fall state machine and events (with confidence score) |
| `gait.py`, `tug.py`, `risk.py` | Gait metrics, Timed Up and Go segmentation, risk band and baseline drift |
| `pipeline.py` | Glue: select person → clean → features |
| `incidents.py`, `mqtt_out.py`, `privacy.py` | Durable JSONL log, resilient MQTT publisher, blur-before-save |
| `run.py` | Scriptable CLI runner (replay, stdin, tail, folder, MQTT) |
| `demo.py` | One-command live demo, status line, overlay, `--selftest` |
| `overlay.py` | OpenCV view; video export with people blurred |
| `synth.py`, `eval.py`, `make_results.py` | Synthetic clips with ground truth, evaluation and tuning, generated results |
| `compare_precision.py` | FP16 vs INT8 keypoint agreement (accuracy column of the Edge Optimization bonus) |

## Data contract

Canonical frame, one JSON object per line:

```json
{"t": 12.345, "persons": [{"id": 0, "bbox": [x1, y1, x2, y2], "kp": {"l_shoulder": [412.0, 220.5, 0.91]}}]}
```

- Coordinates are in pixels, x to the right, y downward.
- Keypoints use COCO-17 names: `nose l_eye r_eye l_ear r_ear l_shoulder r_shoulder l_elbow r_elbow l_wrist r_wrist l_hip r_hip l_knee r_knee l_ankle r_ankle`.
- `persons` may be empty, and `id`/`bbox` are optional.
- A missing keypoint is absent, `[0, 0, 0]`, or below confidence 0.3.

`io.auto_adapt` converts other common layouts into this form; see `logic/README.md`.

## Outputs

Incident record (`incidents.jsonl` and `safestride/alerts`):

```json
{"ts": "2026-10-15T20:41:07.563+05:30", "event": "FALL_CONFIRMED", "confidence": 0.996,
 "torso_angle_deg": 89.3, "hip_drop_speed_Lps": 4.15, "time_down_s": 3.03,
 "camera": "builtin0", "model": "pose-fp16", "frame_saved": null, "t_s": 7.563}
```

Risk summary (`safestride/risk`):

```json
{"ts": "...", "person": "alice", "walk_speed_mps": 0.85, "cadence_spm": 97.9, "step_cv_pct": 7.4,
 "symmetry_pct": 13.5, "tug_s": null, "risk_score": 0.54, "band": "MODERATE", "calibrated": false,
 "drift_alert": null}
```

## Privacy

By default only keypoints are processed and stored; a skeleton is not a face. An evidence frame is saved only when a video source is given, and `privacy.save_blurred` blurs every person (box expanded 15 % per side, Gaussian 51×51, σ 20) before writing. If no person can be located, it blurs the whole frame. Everything runs on the device, with no cloud and no external API.
