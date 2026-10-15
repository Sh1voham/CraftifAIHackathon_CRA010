"""One-command live demo of the SafeStride decision layer.

    python -m logic.demo --source tail:C:/pipegen/out/keypoints.jsonl
    python -m logic.demo --source folder:C:/pipegen/out/frames
    python -m logic.demo --source mqtt:localhost:1883/pipegen/keypoints
    python -m logic.demo --source file:docs/demo_fall.jsonl --realtime --overlay
    python -m logic.demo --selftest

Runs filters, features, the fall machine, gait and risk, the incident log and
MQTT publishing, and prints one status line per second of frame time. Per-frame
errors are logged with the frame time and skipped. ``--selftest`` replays a
built-in synthetic fall (and a fast sit) through the whole stack and prints
PASS or FAIL, checking MQTT delivery when a broker is reachable.
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from logic.config import Config, load_config
from logic.fall import ALERT_KINDS
from logic.gait import Calibration
from logic.incidents import IncidentLog, WallClock
from logic.io import AdaptOpts, Frame, add_adapter_args, opts_from_args, read_keypoints, write_jsonl
from logic.mqtt_out import MqttPublisher, make_client
from logic.risk import BaselineStore
from logic.run import GaitMonitor, Runner, VideoFrames

logger = logging.getLogger("safestride.demo")


def source_frames(source: str, cfg: Config, opts: AdaptOpts) -> Iterator[Frame]:
    """Frames for ``tail:PATH``, ``folder:DIR``, ``mqtt:HOST:PORT/TOPIC``, ``file:PATH`` or ``-``."""
    kind, _, rest = source.partition(":")
    io = cfg.io
    if source == "-" or kind == "stdin":
        return read_keypoints("-", opts=opts)
    if kind == "tail":
        return read_keypoints(rest, follow=True, poll_s=io.poll_s, opts=opts)
    if kind == "folder":
        return read_keypoints(rest, poll_s=io.poll_s, settle_s=io.file_settle_s, opts=opts)
    if kind == "mqtt":
        return read_keypoints(f"mqtt://{rest}", opts=opts)
    if kind == "file":
        return read_keypoints(rest, opts=opts)
    raise ValueError(f"unknown --source '{source}' (use tail:PATH, folder:DIR, mqtt:HOST:PORT/TOPIC or file:PATH)")


def status_line(runner: Runner, t: float) -> str:
    """One-line state summary for the terminal."""
    f = runner.det.last_features

    def fmt(v: float | None, spec: str, unit: str) -> str:
        return "-" if v is None else f"{v:{spec}}{unit}"

    risk = runner.risk_records[-1] if runner.risk_records else None
    band = "-" if risk is None else f"{risk['band']} ({risk['risk_score']:.2f})"
    speed = "" if risk is None or risk["walk_speed_mps"] is None else f" walk {risk['walk_speed_mps']:.2f} m/s"
    n_alerts = sum(1 for r in runner.records if r["event"] in ALERT_KINDS)
    return (f"[t={t:7.1f}s] {runner.det.state.value:<14} torso {fmt(f.phi if f else None, '3.0f', 'deg'):>6}  "
            f"hip speed {fmt(f.v if f else None, '+.2f', ' L/s'):>10}  hip h {fmt(f.h if f else None, '.2f', ' L'):>7}"
            f"  risk {band}{speed}  alerts {n_alerts}")


class Canvas:
    """Overlay window: video frames when given, otherwise a dark keypoint-only canvas."""

    def __init__(self, size: tuple[int, int], video: str | None) -> None:
        import cv2

        from logic.overlay import OverlayState

        self.cv2 = cv2
        self.size = size
        self.video = VideoFrames(video) if video else None
        self.state_cls = OverlayState
        self.state: Any = None

    def show(self, runner: Runner, frame: Frame, t_rel: float) -> bool:
        """Draw the latest state; False when the user closed the window."""
        from logic.overlay import draw

        img = self.video.at(t_rel) if self.video else None
        if img is None:
            img = np.full((self.size[1], self.size[0], 3), 30, np.uint8)
        f = runner.det.last_features
        persons = frame.persons
        kp = None
        if persons:
            from logic.io import KEYPOINT_NAMES
            c_min = runner.cfg.filters.c_min
            kp = [k if k is not None and k[2] >= c_min else None for k in (persons[0].kp.get(n) for n in KEYPOINT_NAMES)]
        alert = None
        if runner.records:
            last = runner.records[-1]
            alert = f"{last['event']}  {last['ts']}  confidence={last['confidence']:.2f}"
        band = runner.risk_records[-1]["band"] if runner.risk_records else None
        draw(img, kp, runner.det.state, f.phi if f else None, f.v if f else None, band, alert)
        self.cv2.imshow("SafeStride demo", img)
        return (self.cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)


def run_demo(args: argparse.Namespace, cfg: Config) -> int:
    """Main loop for a live or replayed source."""
    opts = opts_from_args(args, cfg.io)
    publisher = None
    if not args.no_mqtt:
        host, _, port = args.mqtt.partition(":")
        publisher = MqttPublisher(host, int(port or 1883), cfg.mqtt, client_id="safestride-demo")
    incidents = IncidentLog(args.incidents)
    calib = None
    if args.calib:
        x1, y1, x2, y2, metres = (float(v) for v in args.calib.split(","))
        calib = Calibration.from_points((x1, y1), (x2, y2), metres)
    baselines = BaselineStore(args.baselines, cfg.risk) if args.person else None
    video = VideoFrames(args.video) if args.video else None
    clock = WallClock(None)  # live wall clock for incident timestamps
    runner = Runner(cfg, clock, args.camera, args.model, incidents, publisher,
                    frame_source=video.at if video else None, blur_dir=args.blur_dir,
                    gait=GaitMonitor(cfg, calib), baselines=baselines, person=args.person)
    canvas = Canvas(opts.frame_size or (1280, 720), args.video) if args.overlay else None
    wall0 = time.monotonic()
    t0: float | None = None
    next_status: float | None = None
    next_beat = wall0
    try:
        for frame in source_frames(args.source, cfg, opts):
            t0 = frame.t if t0 is None else t0
            if args.realtime:
                lag = (frame.t - t0) - (time.monotonic() - wall0)
                if lag > 0:
                    time.sleep(lag)
            try:
                for ev in runner.process(frame):
                    rec = runner.records[-1] if runner.records else {}
                    print(f"*** {ev.kind} at t={ev.ts_s:.2f}s confidence={ev.confidence:.2f} "
                          f"-> {args.incidents}{' + MQTT ' + cfg.mqtt.topic_alerts if publisher else ''} "
                          f"{json.dumps(rec)}", flush=True)
            except Exception:  # noqa: BLE001 - never stop the demo on one bad frame
                logger.exception("t=%.3f: error while processing frame", frame.t)
                continue
            if next_status is None or frame.t >= next_status:
                print(status_line(runner, frame.t - t0), flush=True)
                next_status = frame.t + cfg.io.status_period_s
            if publisher is not None and time.monotonic() >= next_beat:
                publisher.publish(cfg.mqtt.topic_status, runner.status())
                next_beat = time.monotonic() + cfg.mqtt.heartbeat_s
            if canvas is not None and not canvas.show(runner, frame, frame.t - t0):
                break
    except KeyboardInterrupt:
        print("stopped", flush=True)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        runner.finish()
        if runner.risk_records:
            print(f"session gait/risk: {json.dumps(runner.risk_records[-1])}", flush=True)
        if publisher is not None:
            publisher.publish(cfg.mqtt.topic_status, runner.status())
            publisher.close()
        incidents.close()
        if canvas is not None:
            canvas.cv2.destroyAllWindows()
    return 0


# ---------------------------------------------------------------- self test

def _broker_reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def selftest(cfg: Config, mqtt_addr: str) -> bool:
    """Replay a synthetic fall and a fast sit through the full stack; print PASS/FAIL per check."""
    from logic.synth import make_clip

    checks: list[tuple[str, bool | None, str]] = []
    host, _, port_s = mqtt_addr.partition(":")
    port = int(port_s or 1883)
    broker = _broker_reachable(host, port)
    received: list[dict[str, Any]] = []
    sub = None
    if broker:
        sub = make_client("safestride-selftest-sub")
        sub.on_message = lambda _c, _u, msg: received.append(json.loads(msg.payload))
        sub.connect(host, port, cfg.mqtt.keepalive_s)
        sub.subscribe(cfg.mqtt.topic_alerts, qos=1)
        sub.loop_start()
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        fall = make_clip("fall", "forward_recover", 7, "selftest_fall")
        sit = make_clip("sit_fast", "chair", 8, "selftest_sit")
        write_jsonl(d / "fall.jsonl", fall.frames)
        write_jsonl(d / "sit.jsonl", sit.frames)
        publisher = MqttPublisher(host, port, cfg.mqtt, client_id="safestride-selftest") if broker else None
        if publisher is not None:
            for _ in range(20):  # give the publisher up to 2 s to connect
                if publisher.connected:
                    break
                time.sleep(0.1)
        results = {}
        for name, clip in (("fall", fall), ("sit", sit)):
            log = IncidentLog(d / f"{name}_incidents.jsonl")
            runner = Runner(cfg, WallClock(datetime.now().astimezone()), "selftest", "synthetic", log, publisher,
                            gait=GaitMonitor(cfg))
            frames = list(read_keypoints(d / f"{name}.jsonl"))
            for fr in frames:
                runner.process(fr)
            runner.finish()
            log.close()
            lines = (d / f"{name}_incidents.jsonl").read_text(encoding="utf-8").splitlines()
            results[name] = (runner, [json.loads(x) for x in lines], clip.label, clip.frames[0].t)
        fall_runner, fall_log, lab, t_first = results["fall"]
        kinds = [r["event"] for r in fall_log]
        checks.append(("fall clip raises exactly one FALL_CONFIRMED", kinds.count("FALL_CONFIRMED") == 1, str(kinds)))
        if "FALL_CONFIRMED" in kinds:
            delay = fall_log[kinds.index("FALL_CONFIRMED")]["t_s"] - (lab["t_fall_start"] - t_first)
            checks.append(("alert 3.0-4.0 s after the fall starts", 3.0 <= delay <= 4.0, f"{delay:.2f} s"))
        checks.append(("RECOVERED logged after getting up", "RECOVERED" in kinds, str(kinds)))
        keys = {"ts", "event", "confidence", "torso_angle_deg", "hip_drop_speed_Lps", "time_down_s", "camera",
                "model", "frame_saved"}
        checks.append(("incident log schema", bool(fall_log) and keys <= set(fall_log[0]), ""))
        sit_kinds = [r["event"] for r in results["sit"][1]]
        checks.append(("fast sit raises no alert", sit_kinds == [], str(sit_kinds)))
        if publisher is not None:
            for _ in range(30):
                if any(m.get("event") == "FALL_CONFIRMED" for m in received):
                    break
                time.sleep(0.1)
            got = any(m.get("event") == "FALL_CONFIRMED" for m in received)
            checks.append((f"MQTT {cfg.mqtt.topic_alerts} delivered via {host}:{port}", got,
                           f"{len(received)} message(s)"))
            publisher.close()
        else:
            checks.append((f"MQTT delivery (no broker at {host}:{port})", None, "SKIPPED"))
    if sub is not None:
        sub.loop_stop()
        sub.disconnect()
    ok = all(c[1] is not False for c in checks)
    for name, passed, detail in checks:
        mark = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
        print(f"  [{mark}] {name}{'  (' + detail + ')' if detail else ''}", flush=True)
    print("SELFTEST PASS" if ok else "SELFTEST FAIL", flush=True)
    return ok


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m logic.demo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", help="tail:PATH | folder:DIR | mqtt:HOST:PORT/TOPIC | file:CLIP.jsonl | -")
    ap.add_argument("--selftest", action="store_true", help="replay a built-in synthetic fall and print PASS/FAIL")
    ap.add_argument("--realtime", action="store_true", help="pace a file source at its original timestamps")
    ap.add_argument("--overlay", action="store_true", help="open an OpenCV window (keypoint canvas or --video)")
    ap.add_argument("--video", default=None, help="video aligned with the keypoints (overlay and blurred evidence)")
    ap.add_argument("--camera", default="builtin0", help="camera label written to incidents")
    ap.add_argument("--model", default="pose-fp16", help="model label written to incidents")
    ap.add_argument("--person", default=None, help="person label for the walking-speed baseline")
    ap.add_argument("--calib", default=None, help="floor calibration 'x1,y1,x2,y2,metres'")
    ap.add_argument("--mqtt", default="localhost:1883", help="broker host:port")
    ap.add_argument("--no-mqtt", action="store_true")
    ap.add_argument("--incidents", default="incidents.jsonl")
    ap.add_argument("--baselines", default="logic/baselines.json")
    ap.add_argument("--blur-dir", default="blurred")
    ap.add_argument("--config", default=None)
    ap.add_argument("--log-level", default="WARNING")
    add_adapter_args(ap)
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.selftest:
        return 0 if selftest(cfg, args.mqtt) else 1
    if not args.source:
        ap.error("--source is required (or use --selftest)")
    return run_demo(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
