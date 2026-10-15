# Two-minute demo script

## Before going on stage

Do these 10 minutes before.

Terminal 1, the MQTT broker (skip if the Mosquitto service is already running):

```bash
mosquitto -v
```

Terminal 2, check the broker and the whole stack:

```bash
python -m logic.demo --selftest
```

It must print `SELFTEST PASS`, including `[PASS] MQTT safestride/alerts delivered`.

Terminal 3, watch everything SafeStride publishes:

```bash
mosquitto_sub -h localhost -t "safestride/#" -v
```

Room setup:
- Put the crash mat in view, side-on to the camera, with a chair next to it. Make sure the full body, ideally including the feet, is visible when standing.
- Optional: put two tape markers 3 m apart on the floor in the walking line, note their pixel positions, and pass them as `--calib x1,y1,x2,y2,3.0`.

## Live run

Terminal 2: the PipeGen app writes keypoints, and the demo follows them.

```bash
python -m logic.demo --source tail:PATH/TO/pipegen_keypoints.jsonl --camera builtin0 --model pose-fp16 --person demo --overlay
```

Use `folder:DIR` or `mqtt:localhost:1883/TOPIC` instead of `tail:` if that is how PipeGen outputs. Check the startup log line that names the detected layout. If the coordinates are normalised, add `--frame-size 1280x720`.

## Script (about 2 minutes)

| Time | Action | What the audience sees |
|---|---|---|
| 0:00–0:10 | Show the PipeGen Pipeline IR graph and the TensorRT speed report. | PipeGen is the perception engine. |
| 0:10–0:20 | Stand in view. | The status line shows `STANDING`, torso about 0°, hip height about 1.8 torso lengths. |
| 0:20–0:45 | **Staged fall onto the mat** (spotter present) and stay down. | State goes `FALLING` → `DOWN`. After about 3 s a `*** FALL_CONFIRMED` line appears, the MQTT terminal shows the alert on `safestride/alerts`, and a new line appears in `incidents.jsonl`. |
| 0:45–0:55 | Stand up and stay up for 5 s. | `RECOVERED` is logged. |
| 0:55–1:15 | **Sit down fast on the chair**, then **bend to pick something up**. | The state may briefly show `FALLING`, then returns to `STANDING`. **No alert.** |
| 1:15–1:40 | **Walk side-on across the view**, turn and walk back once or twice. | The status line and `safestride/risk` show walking speed, symmetry and the risk band. Walk slowly with a shuffle on the next pass, and the band rises. |
| 1:40–2:00 | Close. | "SafeStride is a privacy-preserving screening aid running entirely on the edge. It uses one 2D camera, our thresholds are starting values, and it does not diagnose. It tells a carer to look closer." |

Do not lie on the mat for more than a few seconds unless you want to show the slow-fall alert: 10 s lying low raises `PROLONGED_DOWN`.

## Fallback if the live feed fails

Replay the recorded or synthetic clips at real speed. They exercise the same code path, MQTT and incident log:

```bash
python -m logic.demo --source file:docs/demo_fall.jsonl --realtime --overlay
python -m logic.demo --source file:docs/demo_walk.jsonl --realtime --overlay --person demo
```

If a real staged recording exists (saved earlier with `python -m logic.run --input ... --record staged_fall.jsonl`), replay it instead:

```bash
python -m logic.demo --source file:staged_fall.jsonl --realtime --overlay
```

If MQTT is the problem, add `--no-mqtt`; alerts still go to `incidents.jsonl` and the terminal.
