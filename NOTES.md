# SafeStride decision layer — engineering notes

All numbers are on **synthetic** data unless stated otherwise. SafeStride is a screening prototype, not a medical device. Measured results live in `docs/RESULTS.md`, which `python -m logic.make_results` generates.

## Final status (prompt 2)
1. **Built:**
   - Tier 1: config, io (+ `auto_adapt`), filters, features, fall (+ PROLONGED_DOWN), pipeline, incidents, mqtt_out, privacy, run, synth, eval.
   - Tier 2: gait, tug, risk.
   - Tier 3: overlay, compare_precision.
   - Prompt 2: demo (`--selftest`), make_results.
   - Docs: README, ARCHITECTURE, PROJECT_DESCRIPTION, RESULTS, DEMO_SCRIPT.
2. **Tests:** `python -m pytest logic/tests -q`. The final count is in the last commit message and the final report.
3. **Synthetic held-out test split:** every expected event raised, 0 false alarms, fast-fall delay mean 3.51 s (max 3.60). PROLONGED_DOWN fires on all `lie_slow_stay` clips and on no `lie_slow_recover` clip.
4. **Throughput:** ~11k frames/s for the fall path, ~3.5k with gait (one core).
5. **Not exercised here:** a real MQTT broker (only fakes and an unreachable broker; `--selftest` checks delivery when one exists) and the OpenCV window (headless tests only).
6. **No real PipeGen output seen yet.** All format handling is in `io.py` (`auto_adapt`).

## Status at 18:54 IST (start of prompt 2), kept for the record
Tiers 1–3 were already done and 137 tests passed. Gaps against prompt 2:
- `auto_adapt`, `--ts-unit`/`--coords`/`--kp-map`;
- `demo.py`;
- filename labels and the margin report;
- the docs set;
- PROLONGED_DOWN details;
- synth class names.

All were closed in this session.

## Acceptance checks

| Check | Result |
|---|---|
| Fast-fall recall ≥ 0.95, zero false alarms, mean delay 3.0–3.6 s (held-out split) | ✔ recall 1.000, FP 0, mean 3.51 s (`test_tier1_synthetic_benchmark`) |
| PROLONGED_DOWN on `lie_slow_stay`, not on `lie_slow_recover` | ✔ 10/10 and 0/10 (same test) |
| ≥ 2000 frames/s | ✔ ~11,000 (`test_detector_throughput_at_least_2000_fps`, RESULTS.md §3) |
| Byte-identical replay with fixed `--start-wallclock` | ✔ `test_cli_replay_is_byte_identical` |
| `python -m logic.demo --selftest` prints PASS | ✔ (MQTT check SKIPPED without a broker) |
| Each `auto_adapt` layout has a unit test | ✔ `test_adapt.py` (layouts a–e, bbox forms, units s/ms/us/ns, frame numbers, normalised coords, kp_map, truncation, folder, MQTT) |
| No wall clock in filters/features/fall/gait/tug/risk | ✔ `test_static.py` |
| Gait: speed ±10 %, cadence ±5 spm, SI ±3 pts at 160 and 400 px/m; TUG ≥ 90 % | ✔ `test_gait.py`, `test_tug.py` (TUG 20/20) |
| Fresh clone: install + tests | see final report |

**Synthetic data is easy.** Every class separates perfectly, which says the logic is wired correctly, not that it works on people.

## Final parameter values (`logic/config.json`)
- **Filters:**
  - c_min 0.30;
  - max_gap_frames 5;
  - outlier_jump_Ls 0.5 per typical frame interval, scaled by elapsed time (frame_dt_window 15);
  - L_ref window 2.0 s;
  - One Euro min_cutoff 1.0, beta 0.02, d_cutoff 1.0, reset after a 1.0 s gap.
- **Features:**
  - drop window 0.25 s (reference sample ≤ 0.75 s old);
  - floor learned over 5 s upright (phi < 20°), provisional after 0.5 s, from the 90th percentile of the lower ankle;
  - fallback floor = standing hip + 1.2 L_ref.
- **Fall:**
  - V_DROP 1.2 L/s, PHI_DOWN 60°, H_LOW 0.5 L, T_HOLD 3.0 s, PHI_RECOVER 30°;
  - recover_hold_s 2.0 (DOWN and PROLONGED_DOWN), confirmed_recover_hold_s 5.0;
  - falling_timeout 1.5 s, cooldown 30 s, lost-person grace 2.0 s, break tolerance 0.3 s;
  - **prolonged_hold_s 10.0**, prolonged_requires_h false;
  - confidence = 0.30·c_v + 0.35·c_phi + 0.35·c_t.
- **Gait / TUG / risk:** as specified (see `config.json`). The risk ref/bad values and weights are illustrative, not clinically validated.
- **Input adaptation:**
  - default_fps 30 (frame-number inputs only);
  - unit inference from the median delta of the first 10 frames, with a plausible frame interval of 0.005–0.5 s;
  - coordinates within [0, 1.5] count as normalised.

### Tuning
- **Grid:** 300 combinations (V_DROP {0.8…1.8} × PHI_DOWN {50…70} × H_LOW {0.3…0.7} × T_HOLD {2.5, 3, 3.5}).
- **Feasible:** 180 have zero train false alarms. **All failures use V_DROP ≤ 1.0**, which alerts on `lie_slow_recover` (a deliberate lie-down of about 4 s): its kneel-to-floor stage starts FALLING.
- **Winner:** the defaults (ties are broken toward the defaults), so `config.tuned.json` equals `config.json`.
- **Margins:**
  - `lie_quick_recover` reaches 1.4–1.95 × V_DROP and lies flat, but for only 0.9–1.3 s, so **T_HOLD, not V_DROP, separates it from a fall**;
  - `leave_frame` reaches up to 2.3 × V_DROP as the body is cut by the frame edge, but phi stays under 30°.

## Decisions and deviations (prompt vs PDF vs physics)
1. **One Euro filter** (prompt), not the PDF's EMA.
2. **PROLONGED_DOWN** (prompt 2 spec):
   - Fires when phi > PHI_DOWN and (h < H_LOW **or h unknown**) hold for 10 s, allowing breaks ≤ 0.3 s, with no fast drop and not while FALL_CONFIRMED.
   - Recovers after phi < 30° for 2 s, emitting RECOVERED only if the alert was emitted.
   - Shares the cooldown.
   - **Risk:** with h unknown, someone already lying on a bed when monitoring starts (never seen upright, so no floor estimate) alerts after 10 s. `fall.prolonged_requires_h: true` turns that off; it was the original design. A bed with a known floor (hip ≈ 1 L up) does not alert.
3. **Scale conflict in the brief** (200 px torso vs 3 m = 480 px): synth uses a per-clip `px_per_m` (400 for the benchmark; 160 and 400 for gait).
4. **Hip-speed reference sample** = the newest sample at least `drop_window_s` old (the "oldest" reading smears the speed).
5. **DOWN continuity:** the T_HOLD timer needs the down condition continuously, allowing breaks ≤ 0.3 s.
6. **Added guard:** FALLING also needs the hip above H_LOW at the start of the drop window, so jitter on someone already lying cannot start a fall.
7. **Lost person:** timers pause in down states and reset to STANDING after 2 s invisible; the FALLING timeout keeps running.
8. **Gap repair delays output by ≤ 5 frames** (~0.17 s at 30 fps); event times use frame `t`.
9. **Outlier re-anchoring** after `max_gap_frames` rejections.
10. **Floor line = ankle line.** The fallback `hip + 1.2 L_ref` follows the brief; real anatomy is ≈ 1.8 L, so h reads low. If h is unknown, both paths use phi alone.
11. **Blur:** the box is expanded 15 % per side; the whole frame is blurred if no person is found; `save_blurred` is the only frame writer.
12. **Incident `ts`:** `start_wallclock + (t − t_first)` in replay, local time when live; records also carry `t_s`.
13. **Non-increasing timestamps** are dropped with a warning.
14. **Tuner tie-break:** closest to the defaults, then lowest delay.
15. **Eval (prompt 2):**
    - A TP needs the expected event **kind**; any other alert, including the wrong kind, is a false alarm.
    - FALL_CONFIRMED and PROLONGED_DOWN are reported separately.
    - Real clips (`--labels-from-filenames`) have no start time: the event may occur anywhere and delay is not scored.
    - Real-data `--tune` uses a coarse grid without a split and refuses to write `config.tuned.json` from < 10 clips.
16. **Margin report "longest down"** = the longest continuous stretch with phi > PHI_DOWN and (h < H_LOW or unknown).
17. **`read_keypoints` normalises time to start at 0** (spec 2.3). Incident `t_s` and `--record` files therefore start at 0. `read_jsonl`, used by eval, keeps the original times so `labels.json` start times stay valid.
18. **The demo status line** prints once per second of **frame** time, which equals wall time live and with `--realtime`.
19. **Truncation detection** in `tail_jsonl` triggers when the file becomes shorter than the read position; a file replaced by a longer one is not detected.
20. **Prompt-1 changes kept:** `--record`, the `CleanFrame.raw` keypoints for gait, and the time-scaled outlier limit (a bug found by the gait test).

## Synthetic generator
- **Model:** 2D stick figure in metres, projected with `px_per_m`. Frontal view for sideways falls.
- **Timing:** 30 fps ± 3 ms with dropped frames.
- **Keypoint noise:** σ 0.01 L, ~3 % dropouts in bursts, confidence U(0.5, 0.95), off-image keypoints removed, optional bbox, a bystander in ~10 % of clips.
- **Mix (n=200):**
  - 90 fast falls (forward, backward, sideways × stay or recover);
  - 10 `lie_slow_stay`;
  - 100 negatives: standing, walk (+turn), sit_fast, bend, `lie_slow_recover`, lie_quick_recover, leave_frame.
- **Gait walks:** known speed, cadence and asymmetry; the step peaks sit at the true step times. TUG clips carry the true phase times.

## Known limitations
- **Validated only on synthetic stick figures.** There is no occlusion model, clothing, furniture or perspective, and no falls while walking or toward the camera.
- **Strict torso validity:** phi and v need both shoulders and both hips at confidence ≥ 0.3. A real side view often scores the far side low. **This is the first thing to check on real data.**
- **Ankles are often cropped at 2 m,** so the floor falls back to hip + 1.2 L_ref (see decision 10).
- **Tracking:** one person is followed (nearest mid-hip); identities can swap when people cross.
- **Floor sitting:** someone sitting or kneeling on the floor after a fall (30° < phi < 60°) stays in DOWN without alerting.
- **Evidence frames:** `frame_saved` stays null unless `--video` is given, because the PipeGen app owns the camera.
- **Synthetic gait:** the walking model is idealised (uniform hip speed). Real hip speed oscillates within each step, which the steady-window CV threshold of 0.25 should tolerate. Unverified.

## Vision-side (PipeGen) assumptions still to verify on real output
All of these are handled in `io.py`. Run one sample through `python -m logic.demo --source file:sample.jsonl --log-level INFO` and read the detected-layout line.
1. **Layout and field names:** which of the supported layouts PipeGen emits. If none fits, extend `auto_adapt`.
2. **Keypoint set and order:** COCO-17 assumed; otherwise pass `--kp-map`.
3. **Coordinates:** pixels or normalised. Normalised needs `--frame-size` unless the object carries width/height.
4. **Timestamp field and unit:** inferred from the median frame delta. Check the logged unit (DeepStream PTS is usually ns).
5. **Confidence scale:** 0–1 probabilities assumed, not logits.
6. **Granularity:** one object per frame with all persons assumed. If PipeGen emits one object per person, they must be grouped by frame in `io.py`.
7. **Transport:** file, folder, stdout or MQTT, and the topic name.
8. **Video/keypoint time alignment** for evidence frames.

## Real-data handoff log
No real sample received yet.
