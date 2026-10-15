"""SafeStride decision layer CLI: keypoints in, alerts out.

    python -m logic.run --input clip.jsonl --mqtt localhost:1883 --camera builtin0 --model pose-fp16
    pipegen-app | python -m logic.run --input - --mqtt localhost:1883
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from logic.config import Config, load_config
from logic.fall import ALERT_KINDS, Event, FallDetector
from logic.incidents import IncidentLog, WallClock, event_record
from logic.io import Frame, read_keypoints, write_frames
from logic.mqtt_out import MqttPublisher
from logic.pipeline import FramePipeline, Step
from logic.privacy import save_blurred

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


class Runner:
    """Drives pipeline + fall detector and fans events out to log, MQTT and stdout.

    ``frame_source(t_rel)`` (optional) returns the camera image at ``t_rel``
    seconds after the first keypoint frame; alert frames are then saved, blurred,
    under ``blur_dir``. Without it only keypoints are kept.
    """

    def __init__(self, cfg: Config, clock: WallClock, camera: str = "builtin0", model: str = "pose-fp16",
                 incidents: IncidentLog | None = None, publisher: MqttPublisher | None = None,
                 out: Any = None, frame_source: Callable[[float], np.ndarray | None] | None = None,
                 blur_dir: str | Path = "blurred") -> None:
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
        return events

    def _step(self, step: Step) -> list[Event]:
        before = self.det.state
        events = self.det.update(step.features.t, step.features)
        if self.det.state is not before:
            logger.info("t=%.3f state %s -> %s", step.features.t, before.value, self.det.state.value)
        for ev in events:
            self._emit(ev, step)
        return events

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
    ap.add_argument("--log-level", default="INFO")
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
    runner = Runner(cfg, clock, args.camera, args.model, incidents, publisher, out=sys.stdout,
                    frame_source=frame_source, blur_dir=args.blur_dir)

    record = open(args.record, "w", encoding="utf-8", newline="\n") if args.record else None
    wall0 = time.monotonic()
    t0: float | None = None
    next_beat = wall0
    try:
        for frame in read_keypoints(args.input, follow=args.follow, poll_s=cfg.io.poll_s,
                                    idle_timeout_s=args.idle_timeout, settle_s=cfg.io.file_settle_s):
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
