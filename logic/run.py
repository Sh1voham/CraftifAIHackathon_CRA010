"""SafeStride decision layer CLI: keypoints in, alerts out.

    python -m logic.run --input clip.jsonl --mqtt localhost:1883 --camera builtin0 --model pose-fp16
    pipegen-app | python -m logic.run --input - --mqtt localhost:1883

Fall events go to incidents.jsonl, MQTT safestride/alerts and stdout. Gait /
risk summaries go to MQTT safestride/risk and the log.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from logic.config import Config, load_config
from logic.fall import ALERT_KINDS, Event, FallDetector
from logic.gait import Calibration, GaitSample, GaitSummary, analyze_gait, sample_from
from logic.incidents import IncidentLog, WallClock, event_record
from logic.io import Frame, add_adapter_args, opts_from_args, read_keypoints, write_frames
from logic.mqtt_out import MqttPublisher
from logic.pipeline import FramePipeline, Step
from logic.privacy import save_blurred
from logic.risk import BaselineStore, RiskResult, risk_score, summary_payload
from logic.tug import TugResult, segment_tug

logger = logging.getLogger("safestride")


class VideoFrames:
    """Random access to a video file by time since its first frame."""

    def __init__(self, path: str | Path) -> None:
        import cv2

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise OSError(f"cannot open video {path}")

    def at(self, t_rel: float) -> np.ndarray | None:
        """Frame at ``t_rel`` seconds, or None past the end."""
        self.cap.set(self._cv2.CAP_PROP_POS_MSEC, max(t_rel, 0.0) * 1000.0)
        ok, img = self.cap.read()
        return img if ok else None


class GaitMonitor:
    """Rolling gait, TUG and risk analysis over the last ``buffer_s`` seconds of frame time."""

    def __init__(self, cfg: Config, calib: Calibration | None = None) -> None:
        self.cfg = cfg
        self.calib = calib
        self.samples: deque[GaitSample] = deque()
        self.best: tuple[GaitSummary, TugResult, RiskResult] | None = None
        self._next_t: float | None = None
        self._last_key: tuple | None = None

    def add(self, step: Step) -> tuple[GaitSummary, TugResult, RiskResult] | None:
        """Add a frame; every ``update_period_s`` return a new summary if walking data changed."""
        s = sample_from(step.clean, step.features)
        self.samples.append(s)
        while self.samples and s.t - self.samples[0].t > self.cfg.gait.buffer_s:
            self.samples.popleft()
        if self._next_t is None:
            self._next_t = s.t + self.cfg.gait.update_period_s
        if s.t < self._next_t:
            return None
        self._next_t = s.t + self.cfg.gait.update_period_s
        return self.analyze()

    def analyze(self, force: bool = False) -> tuple[GaitSummary, TugResult, RiskResult] | None:
        """Analyse the buffer; None if there is no steady walking or nothing new (unless ``force``)."""
        samples = list(self.samples)
        gait = analyze_gait(samples, self.cfg.gait, self.calib)
        if not gait.valid:
            return None
        tug = segment_tug(samples, self.cfg.tug)
        risk = risk_score({"walk_speed": gait.walk_speed_mps, "symmetry": gait.symmetry_pct,
                           "step_cv": gait.step_cv_pct, "tug": tug.tug_s if tug.valid else None}, self.cfg.risk)
        key = (round(gait.windows[-1][1], 1), tug.valid)
        if key == self._last_key and not force:
            return None
        self._last_key = key
        if self.best is None or gait.n_steps >= self.best[0].n_steps:
            self.best = (gait, tug, risk)
        return gait, tug, risk


class Runner:
    """Drives pipeline + fall detector and fans events out to log, MQTT and stdout.

    ``frame_source(t_rel)`` (optional) returns the camera image at ``t_rel``
    seconds after the first keypoint frame; alert frames are then saved, blurred,
    under ``blur_dir``. Without it only keypoints are kept.
    """

    def __init__(self, cfg: Config, clock: WallClock, camera: str = "builtin0", model: str = "pose-fp16",
                 incidents: IncidentLog | None = None, publisher: MqttPublisher | None = None,
                 out: Any = None, frame_source: Callable[[float], np.ndarray | None] | None = None,
                 blur_dir: str | Path = "blurred", gait: GaitMonitor | None = None,
                 baselines: BaselineStore | None = None, person: str | None = None) -> None:
        self.cfg = cfg
        self.clock = clock
        self.camera, self.model = camera, model
        self.incidents = incidents
        self.publisher = publisher
        self.out = out
        self.frame_source = frame_source
        self.blur_dir = Path(blur_dir)
        self.pipe = FramePipeline(cfg)
        self.det = FallDetector(cfg.fall)
        self.frames = 0
        self.last_t: float | None = None
        self.records: list[dict[str, Any]] = []
        self.gait = gait
        self.baselines = baselines
        self.person = person
        self.risk_records: list[dict[str, Any]] = []
        self._t_first: float | None = None
        self._last_persons: list[Any] = []

    def process(self, frame: Frame) -> list[Event]:
        """Feed one input frame."""
        self.frames += 1
        self.last_t = frame.t
        if self._t_first is None:
            self._t_first = frame.t
        self._last_persons = list(frame.persons)
        self.clock.observe(frame.t)
        events: list[Event] = []
        for step in self.pipe.push(frame):
            events += self._step(step)
        return events

    def finish(self) -> list[Event]:
        """Flush the pipeline at end of input."""
        events: list[Event] = []
        for step in self.pipe.flush():
            events += self._step(step)
        if self.gait is not None and self.last_t is not None:
            self.gait.analyze(force=True)
            if self.gait.best is not None:
                g, tug, risk = self.gait.best
                drift = None
                if self.baselines is not None and self.person and g.walk_speed_mps is not None:
                    drift = self.baselines.update(self.person, g.walk_speed_mps)
                    if drift.drift_alert:
                        logger.warning("walking speed drift for %s: %s", self.person, "; ".join(drift.reasons))
                self._publish_risk(self.last_t, g, tug, risk, drift, final=True)
        return events

    def _step(self, step: Step) -> list[Event]:
        before = self.det.state
        events = self.det.update(step.features.t, step.features)
        if self.det.state is not before:
            logger.info("t=%.3f state %s -> %s", step.features.t, before.value, self.det.state.value)
        for ev in events:
            self._emit(ev, step)
        if self.gait is not None:
            res = self.gait.add(step)
            if res is not None:
                self._publish_risk(step.features.t, *res)
        return events

    def _publish_risk(self, t: float, gait: GaitSummary, tug: TugResult, risk: RiskResult,
                      drift: Any = None, final: bool = False) -> None:
        payload = summary_payload(self.clock.iso(t), gait, tug, risk, drift, camera=self.camera,
                                  model=self.model, person=self.person, final=final)
        self.risk_records.append(payload)
        if self.publisher is not None:
            self.publisher.publish(self.cfg.mqtt.topic_risk, payload)
        logger.info("t=%.3f gait%s: speed=%s m/s cadence=%s SI=%s CV=%s TUG=%s -> risk %s %s",
                    t, " (session)" if final else "", payload["walk_speed_mps"], payload["cadence_spm"],
                    payload["symmetry_pct"], payload["step_cv_pct"], payload["tug_s"], payload["band"],
                    payload["risk_score"])

    def _save_frame(self, ev: Event, ts: str, step: Step) -> str | None:
        if self.frame_source is None or ev.kind not in ALERT_KINDS:
            return None
        img = self.frame_source(ev.ts_s - (self._t_first or ev.ts_s))
        if img is None:
            logger.warning("t=%.3f: no video frame available to save", ev.ts_s)
            return None
        stamp = ts[:19].replace("-", "").replace(":", "").replace("T", "_")
        path = self.blur_dir / f"{stamp}_{ev.kind.lower()}.jpg"
        save_blurred(img, path, [step.clean.kp, *self._last_persons], self.cfg.privacy, self.cfg.filters.c_min)
        return path.as_posix()

    def _emit(self, ev: Event, step: Step) -> None:
        ts = self.clock.iso(ev.ts_s)
        try:
            saved = self._save_frame(ev, ts, step)
        except Exception:  # noqa: BLE001 - evidence saving must not block the alert
            logger.exception("t=%.3f: could not save blurred frame", ev.ts_s)
            saved = None
        rec = event_record(ev, ts, self.camera, self.model, saved)
        self.records.append(rec)
        if self.incidents is not None:
            self.incidents.append(rec)
        if self.publisher is not None:
            self.publisher.publish(self.cfg.mqtt.topic_alerts, rec)
        if self.out is not None:
            self.out.write(json.dumps(rec) + "\n")
            self.out.flush()
        level = logging.WARNING if ev.kind in ALERT_KINDS else logging.INFO
        logger.log(level, "t=%.3f %s confidence=%.2f", ev.ts_s, ev.kind, ev.confidence)

    def status(self) -> dict[str, Any]:
        """Heartbeat payload."""
        f = self.det.last_features
        return {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "state": self.det.state.value,
            "frames": self.frames,
            "last_t": self.last_t,
            "person_visible": bool(f is not None and f.phi is not None),
            "camera": self.camera,
            "model": self.model,
        }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m logic.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="JSONL file, '-' for stdin, a folder of per-frame JSON files, or mqtt://host:port/topic")
    ap.add_argument("--follow", action="store_true", help="tail a growing JSONL file")
    ap.add_argument("--idle-timeout", type=float, default=None,
                    help="stop a live source after this many seconds without data")
    ap.add_argument("--mqtt", default=None, help="broker host:port (omit to disable MQTT)")
    ap.add_argument("--camera", default="builtin0")
    ap.add_argument("--model", default="pose-fp16")
    ap.add_argument("--config", default=None, help="config JSON (default logic/config.json)")
    ap.add_argument("--incidents", default="incidents.jsonl")
    ap.add_argument("--video", default=None,
                    help="video file aligned with the keypoints; alert frames are saved blurred")
    ap.add_argument("--blur-dir", default="blurred")
    ap.add_argument("--record", default=None,
                    help="also save every received frame, in replay format, to this JSONL file")
    ap.add_argument("--realtime", action="store_true", help="replay at the original timestamps")
    ap.add_argument("--start-wallclock", default=None,
                    help="ISO time of the first frame; makes replay output reproducible")
    ap.add_argument("--no-gait", action="store_true", help="disable gait / risk analysis")
    ap.add_argument("--calib", default=None,
                    help="floor calibration 'x1,y1,x2,y2,metres': two pixel points a known distance apart")
    ap.add_argument("--person", default=None, help="person label for the walking-speed baseline")
    ap.add_argument("--baselines", default="logic/baselines.json", help="baseline store (JSON)")
    ap.add_argument("--log-level", default="INFO")
    add_adapter_args(ap)
    return ap.parse_args(argv)


def _is_live(args: argparse.Namespace) -> bool:
    src = args.input
    return src == "-" or src.startswith("mqtt://") or args.follow or Path(src).is_dir()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.start_wallclock:
        start = datetime.fromisoformat(args.start_wallclock)
        clock = WallClock(start if start.tzinfo else start.astimezone())
    else:
        clock = WallClock(None if _is_live(args) else datetime.now().astimezone())

    publisher = None
    if args.mqtt:
        host, _, port = args.mqtt.partition(":")
        publisher = MqttPublisher(host, int(port or 1883), cfg.mqtt)
    else:
        logger.info("MQTT disabled (no --mqtt)")
    incidents = IncidentLog(args.incidents)
    frame_source = VideoFrames(args.video).at if args.video else None
    calib = None
    if args.calib:
        x1, y1, x2, y2, metres = (float(v) for v in args.calib.split(","))
        calib = Calibration.from_points((x1, y1), (x2, y2), metres)
    gait = None if args.no_gait else GaitMonitor(cfg, calib)
    baselines = BaselineStore(args.baselines, cfg.risk) if args.person and gait is not None else None
    runner = Runner(cfg, clock, args.camera, args.model, incidents, publisher, out=sys.stdout,
                    frame_source=frame_source, blur_dir=args.blur_dir, gait=gait, baselines=baselines,
                    person=args.person)

    record = open(args.record, "w", encoding="utf-8", newline="\n") if args.record else None
    wall0 = time.monotonic()
    t0: float | None = None
    next_beat = wall0
    try:
        for frame in read_keypoints(args.input, follow=args.follow, poll_s=cfg.io.poll_s,
                                    idle_timeout_s=args.idle_timeout, settle_s=cfg.io.file_settle_s,
                                    opts=opts_from_args(args, cfg.io)):
            if args.realtime:
                t0 = frame.t if t0 is None else t0
                lag = (frame.t - t0) - (time.monotonic() - wall0)
                if lag > 0:
                    time.sleep(lag)
            if record is not None:
                write_frames(record, [frame])
                record.flush()
            try:
                runner.process(frame)
            except Exception:  # noqa: BLE001 - one bad frame must not stop monitoring
                logger.exception("t=%.3f: error while processing frame", frame.t)
            if publisher is not None and time.monotonic() >= next_beat:
                publisher.publish(cfg.mqtt.topic_status, runner.status())
                next_beat = time.monotonic() + cfg.mqtt.heartbeat_s
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        try:
            runner.finish()
        except Exception:  # noqa: BLE001
            logger.exception("t=%s: error while flushing", runner.last_t)
        if publisher is not None:
            publisher.publish(cfg.mqtt.topic_status, runner.status())
            publisher.close()
        incidents.close()
        if record is not None:
            record.close()
    logger.info("done: %d frames, %d events", runner.frames, len(runner.records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
