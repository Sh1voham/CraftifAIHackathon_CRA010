# SafeStride decision layer — engineering notes

Status as of 15 Oct 2026. All numbers below are on **synthetic** data unless
stated otherwise. SafeStride is a screening prototype, not a medical device.

## Tier 1 status: done

| Acceptance check | Result |
|---|---|
| `pytest` | 96 passed, 3 skipped at the Tier 1 gate (the gait/tug/risk wall-clock checks skipped until those modules existed) |
| Synthetic benchmark (200 clips, seed 0, 70/30 stratified split) | test: TP 27, FN 0, FP 0, TN 30; recall 1.000, precision 1.000, F1 1.000, 0 false alarms (0.0/h) |
| Fast-fall alert delay after fall start | test mean 3.51 s, max 3.60 s (train mean 3.51 s, max 3.63 s); target 3.0–3.6 s mean ✔ |
| Slow-lowering `PROLONGED_DOWN` delay after descent start | test mean 13.3 s (≈4 s descent + 10 s hold) |
| Throughput (one core, `--bench`) | ~11,500 frames/s for pipeline + detector (8,400 on an earlier run; timings vary); JSON parsing ~23,000 frames/s; full runner with gait ~4,000 frames/s (target ≥ 2000) ✔ |
| Byte-identical replay | ✔ (`test_cli_replay_is_byte_identical`, with fixed `--start-wallclock`) |
| `--input -` stdin mode | ✔ (subprocess test, including an unreachable MQTT broker) |
| No wall clock in algorithm modules | ✔ (`test_static.py` scans filters, features, fall, gait, tug, risk) |

Numbers are from the final code (after the Tier 2 outlier fix; before it, test mean 3.53 s and max 3.80 s). Delay ≈ time to reach DOWN (~0.45–0.6 s of the ~0.5–0.8 s descent) + `T_HOLD`
(3.0 s) + one frame. Live latency also adds the gap-repair buffer (≤5 frames,
~0.17 s) plus PipeGen inference.

**Synthetic data is easy.** Every class is separated perfectly, so these
numbers say the logic is wired correctly, not that it works on people. Real
recordings take priority.

### Tuning (`python -m logic.eval --data data/synth --tune`)
- Grid: V_DROP {0.8, 1.0, 1.2, 1.5, 1.8} × PHI_DOWN {50…70} × H_LOW {0.3, 0.4, 0.5, 0.7} × T_HOLD {2.5, 3.0, 3.5} = 300 combinations, ~26 s.
- 180/300 combinations have zero train false alarms (unchanged on the final code). **All 120 failures come from V_DROP ≤ 1.0**, which alerts on `lie_down_slow/brief` (a deliberate lie-down for 4–6 s): the kneel-to-floor stage is fast enough to start FALLING. Keep V_DROP ≥ 1.2 unless real data says otherwise.
- The winner equals the defaults (V_DROP 1.2, PHI_DOWN 60, H_LOW 0.5, T_HOLD 3.0); ties are broken toward the defaults. Test F1 1.000. `logic/config.tuned.json` therefore matches `config.json`.

## Final parameter values (`logic/config.json`)
- **Filters:** c_min 0.30; max_gap_frames 5; outlier_jump_Ls 0.5; L_ref window 2.0 s.
- **One Euro:** min_cutoff 1.0, beta 0.02, d_cutoff 1.0, reset after 1.0 s gap.
- **Features:** drop window 0.25 s (reference sample must be ≤ 0.75 s old); floor learned over 5 s upright (phi < 20°), provisional after 0.5 s, 90th percentile of the lower ankle; fallback hip + 1.2 L_ref.
- **Fall:** V_DROP 1.2 L/s; PHI_DOWN 60°; H_LOW 0.5 L; T_HOLD 3.0 s; PHI_RECOVER 30°; recover holds 2.0 s (DOWN) and 5.0 s (confirmed); FALLING timeout 1.5 s; cooldown 30 s; lost-person grace 2.0 s; down-condition break tolerance 0.3 s; **T_PROLONGED 10.0 s**.
- **Confidence:** 0.30·c_v + 0.35·c_phi + 0.35·c_t (as specified).

## Decisions and deviations (prompt vs PDF vs physics)
1. **One Euro filter** (prompt), not the PDF's EMA.
2. **Second path for slow falls (`PROLONGED_DOWN`):** fires when phi > PHI_DOWN and h < H_LOW hold for 10 s with no fast drop. It requires h, so with no floor estimate there is no prolonged alert, and a bed or sofa (hip ≈1 L up) never triggers it. It has a separate event kind and state, the same recovery and cooldown as FALL_CONFIRMED, and counts as a TP in eval for slow-lowering clips. It also backstops a fast fall that only reaches the floor pose after the FALLING timeout.
3. **Scale conflict in the brief:** a 200 px torso (0.5 m) means 400 px/m, but "3 m = 480 px" means 160 px/m. Synth uses a per-clip `px_per_m`: 400 for the benchmark, 160 (and others) for gait clips.
4. **Hip-speed reference sample:** the newest sample that is at least `drop_window_s` old. The prompt's "oldest" and the PDF skeleton (oldest in a 30-frame deque, ≈1 s) would smear the speed.
5. **DOWN continuity:** the T_HOLD timer needs the down condition continuously, allowing breaks ≤ 0.3 s. Otherwise "lie down and get up within 2 s" would confirm after the person stood up. A longer break restarts the timer; the machine stays in DOWN until recovery (phi < 30° for 2 s).
6. **Added guard:** STANDING→FALLING also needs the hip at the start of the drop window to be ≥ H_LOW (or unknown). Without it, jitter on someone already lying could start FALLING and alert 3 s later.
7. **Lost person:** in DOWN, FALL_CONFIRMED and PROLONGED_DOWN the timers *pause* while the torso is invisible, then reset to STANDING after 2 s. The FALLING timeout keeps running. "Visible" means both shoulders and both hips are valid.
8. **Gap repair is causal with a ≤5-frame delay** (it needs the next valid sample). Event times come from frame `t`, so measured delays are unaffected; live latency grows by ~0.17 s at 30 fps. Interpolation never bridges more than 1.0 s.
9. **Outlier re-anchoring:** a keypoint rejected for more than `max_gap_frames` frames in a row is accepted at its new position, so one bad anchor cannot lock it out. Rejections only compare against a sample from the last ~6 frames, and only once L_ref exists.
10. **Floor line = ankle line** (~0.04 m above the true floor in synth). The fallback `hip + 1.2 L_ref` follows the brief, but real anatomy is ≈1.8 L, so the fallback puts the floor too high and makes h too small. That is safe for falls but could help a deep bend reach DOWN. If h is unknown, the fast path uses phi alone.
11. **Blur:** the person box is expanded 15 % **on each side** (the conservative reading). With no locatable person the whole frame is blurred. `save_blurred` is the only frame writer, and its test spies on the encoder to prove the original is never written.
12. **Incident `ts`:** in replay, `start_wallclock + (t − t_first)` (the first frame maps to the start time) rather than `+ t`, because PipeGen's `t` may be an epoch. Live sources use the current local time. Records also carry `t_s` (frame time) for correlating with keypoint logs.
13. **Non-increasing timestamps** are dropped with a warning in `pipeline.py`.
14. **Tuner tie-break:** closest to the defaults (normalised by grid span), then lowest mean delay. This keeps T_HOLD at the 3 s product requirement unless a change actually helps.
15. **Eval false alarms** are counted strictly: any alert on a negative clip, or any extra alert outside `[t_fall_start, +20 s]` on a positive clip.
16. **Added `--record`** in `run.py` to capture live sessions in replay format for labelling and tuning.

## Synthetic generator (`logic/synth.py`)
- **Model:** 2D stick figure in metres (torso 0.5 m, thigh 0.43, shank 0.42, ankle 0.08; ±5 % body scale per clip).
- **Views:** side view for most clips; frontal view for sideways falls.
- **Timing:** 30 fps with ±3 ms jitter, 2 % dropped frames plus rare 2–3-frame bursts, and a random t offset (clips never start at 0).
- **Keypoint noise:** σ = 0.01 L, ~3 % dropouts in 1–3-frame bursts (written as `[0,0,0]`), confidences U(0.5, 0.95), keypoints outside the image removed, person dropped when fewer than 5 keypoints are visible. bbox is present in about half the clips; a smaller bystander appears in ~10 %.
- **Mix (n=200):** 90 fast falls (forward/backward/sideways × stay/recover, mean hip speed 2–3 L/s, phi 80–90°), 10 slow-lowering-stay (positive, PROLONGED_DOWN), and 100 negatives: standing, walk (+turn), sit_fast, bend, lie_down_slow/brief, lie_quick_recover, leave_frame.

## Known limitations
- Everything is validated only on synthetic stick figures. There is no occlusion model, no clothing or furniture, no camera perspective (one fixed px/m), no falls while walking, and no falls toward or away from the camera.
- **Strict torso validity:** phi and v need both shoulders and both hips at confidence ≥ 0.3. In a real side view the far-side shoulder or hip often scores low, which would starve the features. This is the first thing to check on real data (a single-side fallback is a small change, not built).
- **Ankles are often cropped** on a laptop camera at 2 m. The floor then falls back to the hips (see decision 10), and PROLONGED_DOWN stays silent if h is never available.
- One person is followed at a time. The tracker is nearest-mid-hip, which can swap identities when people cross closely.
- **Persistent DOWN:** a person who ends up kneeling or sitting on the floor (30° < phi < 60°) stays in DOWN without alerting until they stand.
- `frame_saved` is `null` unless `--video` is given. The live frame source from PipeGen is not wired (the camera is owned by the PipeGen app).

## PipeGen assumptions still to verify against real output
1. **Field names** and nesting (`t`, `persons`, `kp`, `bbox`, `id`). Change `_from_pipegen()` / `PIPEGEN_NAME_MAP` only.
2. **Keypoint order and names:** COCO-17 assumed.
3. **Coordinates:** pixels assumed. If normalised [0, 1], scale them in `_from_pipegen()`; One Euro `beta` and the blur box work in pixels.
4. **Timestamp units and origin:** seconds assumed. DeepStream PTS is often in nanoseconds, so convert.
5. **Missing-keypoint encoding:** absent, `[0,0,0]` and low confidence are all handled.
6. **Confidence scale:** 0–1 probabilities assumed (not logits).
7. **Granularity:** one object per frame assumed. If PipeGen emits one object per person, group them by frame in `_from_pipegen()`/`read_keypoints()`.
8. **Transport:** stdout, a growing file, a folder of per-frame files and MQTT are all supported; confirm which PipeGen uses and the topic name.
9. **Video/keypoint time alignment** for `--video` evidence frames.

## Tier 2 status: done

| Acceptance check | Result |
|---|---|
| Calibrated walking speed within 10 % | ✔ within ~0.2 % on synthetic walks at 0.6–1.3 m/s, 160 and 400 px/m |
| Cadence within 5 steps/min | ✔ within ~1 spm (85–115 spm) |
| Asymmetric walk (left steps 15 % slower, true SI 13.95) within 3 points | ✔ 11.9–15.6 over 8 seeds at 160 px/m and 12.5–15.0 at 400 px/m; tested on 6 seeds |
| Symmetric walk | SI ≤ 2.4, step CV ≤ 3.3 % (true 0) |
| TUG phases found on ≥ 90 % of synthetic TUG clips | ✔ 20/20; TUG time within 0.02 s and sit-to-stand within 0.02 s of truth |
| Risk handles missing metrics | ✔ renormalised weights; no metrics gives `score=None, band=None` |
| `pytest` | 132 passed, 0 skipped |

- **Gait (`gait.py`):**
  - Steady windows need: hip speed > 0.3 L_ref/s, rolling speed CV ≤ 0.25 over 1 s, phi < 25°, one direction, no data hole > 0.3 s, ≥ 1.5 s after trimming 0.3 s from each end.
  - Speed is distance over time across all windows.
  - Steps: `find_peaks` on |A_L,x − A_R,x| (distance 0.25 s × fps_est, prominence 0.1 L_ref), then a least-squares parabola over ±0.1 s around each peak.
  - Peaks whose fit window is cut by the window edge are dropped. Intervals are pooled across windows, and each interval is assigned to the side that leads at its closing peak.
- **Live gait in `run.py`:** recomputed every 2 s of frame time over a 60 s buffer and published to `safestride/risk` when a new steady window ends. At end of input, the session summary (the one with the most steps) updates the `--person` baseline in `logic/baselines.json`. Gait costs about 2.5× CPU: the runner drops from ~11k to ~4k frames/s, still far above 30 fps.
- **TUG (`tug.py`):**
  - y_n uses the 2nd/98th percentiles of hip y, and the hip must travel ≥ 0.4 L_ref.
  - The turn is the first reversal of hip vx (central difference ±0.25 s) with hysteresis. The threshold is max(0.2 L_ref/s, 0.5 × median walking speed), so the TURN phase spans the slow-down and re-acceleration, not just the instant of reversal.
- **Risk (`risk.py`):**
  - Formulae, ref/bad values and weights are as specified, with the guide's worked example reproduced in a test (S = 0.50, MODERATE).
  - Drift uses the literal formula, comparing v_t with 0.8 × b_t where b_t already includes v_t (slightly less sensitive than comparing with b_(t−1)).
  - z uses the previous ≥ 5 sessions (ddof = 1).
  - The values are illustrative design choices, not a validated clinical model.

### Changes to Tier 1 modules (each proven by a failing Tier 2 test)
1. **Bug: outlier limit across dropped frames** (`filters.py`). The fixed per-frame jump limit (0.5 L_ref) rejected genuinely fast ankles after 2–3 dropped frames, smearing step peaks. The limit now scales with the time since the last accepted sample, relative to the median of the last 15 frame intervals (estimated from timestamps, so no fps is assumed). The fall benchmark is unchanged (test mean delay now 3.51 s, max 3.60 s).
2. **`CleanFrame.raw`** (`filters.py`): the repaired but unsmoothed keypoints. Gait uses them for ankles because the One Euro filter's speed-dependent lag shifted left- and right-led step peaks differently. On a noise-free asymmetric walk, SI read 8.4 instead of 13.95. The fall path still uses the smoothed keypoints.
3. **Synthetic walking model** (`synth.py`): knee flexion now goes as cos³ with compensating thigh flexion, and the gait phase runs at the mean rate within ±0.6 rad of each step peak. The old model put spurious separation peaks at mid-swing and placed the left/right rate change exactly at the peaks, which made the "true" step times ill-defined.

## Tier 3 status: done
- **`overlay.py`:** the OpenCV view (skeleton coloured by fall state, torso angle, hip speed, risk band, last-alert banner) runs the full decision layer. It works on a video file plus JSONL, aligning the first keypoint frame with the video start, and on a live camera index with keypoints from any `read_keypoints` source in a background thread. `--save` writes the overlay video with every person blurred first; a test checks the saved video and the blurred evidence frame. The window itself is not tested here (headless runs only). On the demo laptop the PipeGen app probably owns the camera.
- **`compare_precision.py`:** aligns two runs by timestamp (nearest within half the median frame interval) and reports, as Markdown:
  - normalised keypoint error, overall and per keypoint;
  - the confidence-drop (> 0.2) fraction;
  - alerts and labelled falls caught per run.

  It is tested on a synthetic perturbation with known error (σ = 0.02 L gives a mean error of 0.025 L, within 15 %). It needs real FP16 and INT8 recordings of the same clips; it cannot use the same live run.
- **Other changes:** `privacy.blur_people` was factored out of `save_blurred` for reuse by the overlay (behaviour unchanged). `pytest` now gives 137 passed.

## Not done yet
- No real recordings evaluated yet. Everything above is synthetic.
- The live overlay window and a real MQTT broker were not exercised: no display test and no Mosquitto on this Windows dev machine. MQTT was tested with a fake client and an unreachable broker only.
