# SafeStride: fall detection and gait-based fall-risk screening on CraftifAI PipeGen

*Text for the submission form. Items marked `[FILL: ...]` can only be completed by the pipeline side, from PipeGen's outputs and reports. Remove this line before pasting.*

## Problem

Falls at home often go unnoticed for hours, especially for older people who live alone. A gradual decline in walking ability, which often comes before a fall, is rarely measured between clinic visits. (Hackathon problem #2: Fall Detection and Elder Care.)

## Who it helps

- Older adults living alone or in assisted living.
- Carers and family members, who get a prompt alert instead of discovering a fall later.
- Clinicians, who get a low-cost, privacy-preserving signal that someone's gait is changing and deserves a closer look.

## Track

Track A, built on CraftifAI PipeGen.

## How PipeGen is load-bearing

All perception is built, compiled and run in PipeGen. Our own code only consumes its keypoint output.

| Pipeline stage | What we used |
|---|---|
| Requirement | Pose estimation from a camera at real-time rate, per-frame keypoints as JSON, fall flag, blur before saving, MQTT alerts |
| Pipeline IR | `[FILL: node graph summary, e.g. source -> pose inference -> metadata sink]` (screenshot in repo: `[FILL: path]`) |
| Model | `[FILL: pose model name from the PipeGen catalog]`, `[FILL: number of keypoints / layout]` |
| Compiler | TensorRT `[FILL: FP16]` engine (`[FILL: INT8 engine if built]`) |
| Run | Input `[FILL: built-in camera /dev/video0 | USB | RTSP | video file]` on an NVIDIA `[FILL: RTX 5050]` |
| Speed (PipeGen report) | FP16: `[FILL: latency ms]` / `[FILL: FPS]` · INT8: `[FILL: latency ms]` / `[FILL: FPS]` |
| Output format | `[FILL: JSON metadata file / folder of per-frame JSON / MQTT topic]` |

The decision layer (`logic/`) reads that output directly. A tail of the growing JSON file, a watch on the frame folder, or an MQTT subscription all work. It detects the layout, timestamp unit and coordinate mode automatically.

## Outputs (actionable, beyond boxes on screen)

- **JSON incident log** `incidents.jsonl`: one line per event (`FALL_CONFIRMED`, `PROLONGED_DOWN`, `RECOVERED`). Each line has the time, a confidence score, torso angle, hip drop speed, time spent down, camera and model. Every write is flushed to disk.
- **MQTT** on a local Mosquitto broker, QoS 1:
  - `safestride/alerts`: fall events;
  - `safestride/risk`: walking speed, cadence, step-time variability, left/right symmetry, TUG time and a LOW / MODERATE / HIGH risk band;
  - `safestride/status`: heartbeat every 5 s, so a silent system is detectable.
- **Privacy:** only keypoints are kept by default. Any evidence frame is blurred before it is written, and nothing leaves the device.

## The decision logic in plain words

1. **Clean the keypoints.** Drop low-confidence points and reject impossible jumps. Fill gaps of a few frames, then smooth with a One Euro filter that uses the real frame timing. Measure everything in torso lengths, so it does not matter how far the person stands from the camera.
2. **Fast fall.** An alert needs three things:
   - the hips drop faster than 1.2 torso lengths per second;
   - within 1.5 s the torso is close to horizontal (over 60°) and the hips are near the floor;
   - the person stays down for 3 s.

   A fast sit or a bend does not alert: the torso stays upright, or the hips stay high.
3. **Slow fall.** Lying low and nearly horizontal for 10 s raises `PROLONGED_DOWN`, even without a fast drop. This covers someone who slides down a wall.
4. **Recovery.** Getting back upright logs `RECOVERED`. A 30 s cooldown prevents duplicate alerts.
5. **Gait and risk.** While the person walks side-on across the view, the system measures:
   - walking speed (calibrated with two floor markers, or a rough body-size scale);
   - cadence, step-time variability and left/right step symmetry;
   - the Timed Up and Go time, when a sit–walk–turn–sit sequence is seen.

   These become a weighted LOW / MODERATE / HIGH band. A per-person baseline flags a drop of walking speed below 80 % of that person's own usual speed. The weights and thresholds are illustrative design choices, not a validated clinical model.

## Bonus claimed

Edge Optimization: `[FILL: claim only if both FP16 and INT8 engines were built and reported]`.

| Precision | Latency (ms) | FPS | Avg power (W) | Energy (J/frame) | Keypoint error vs FP16 (/ torso length) | Falls caught |
|---|---|---|---|---|---|---|
| FP16 | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | 0 (reference) | `[FILL: x / N]` |
| INT8 | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL]` | `[FILL: from python -m logic.compare_precision]` | `[FILL: x / N]` |

The accuracy column comes from `python -m logic.compare_precision` on the same recorded clips with both engines.

## Evaluation results

These numbers are measured by `python -m logic.make_results` (full tables in `docs/RESULTS.md`). The synthetic numbers come from a seeded stick-figure generator: 200 clips, 90 fast falls, 10 slow lowerings and 100 non-fall activities (sitting fast, bending, lying down and getting up, walking, leaving the frame).

- **Synthetic held-out test split:**
  - every expected alert was raised and there were 0 false alarms;
  - fast-fall alert 3.51 s after the fall starts (mean);
  - the threshold grid search kept the default thresholds.
- **Synthetic gait:** walking speed within 0.2 % and cadence within 1 step/min of the generator settings. A deliberately asymmetric walk (left steps 15 % slower, true symmetry index 13.95 %) measured 14.3–14.5 %. TUG phases were found on 20/20 synthetic trials.
- **Throughput:** about 11,000 frames/s on one CPU core for the fall path, and about 3,500 with gait analysis (a camera delivers 30).
- **Real recordings:** `[FILL: n clips; falls caught x / y; false alarms on sit/bend/lie-and-get-up clips; from docs/RESULTS.md section 4]`.

Synthetic data is far easier than real video. The synthetic results show the logic behaves as designed; the real-clip results are the ones that matter.

## Honest limitations

- A single 2D camera with a narrow view. Falls straight towards or away from the camera, heavy occlusion and people out of frame are hard cases.
- Thresholds are starting values tuned on synthetic data and a small number of staged recordings, not on a clinical population.
- Gait metrics need side-on walking across the view, and speed in metres needs a two-marker floor calibration. Without it, speed is a rough estimate and is flagged `calibrated=false`.
- The risk band uses illustrative weights and is meant to tell a carer to look closer. It is a screening aid, not a medical device, and it does not diagnose.
- Monitoring a person requires their consent.
