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
from typing import Any

from logic.config import Config, load_config
from logic.fall import ALERT_KINDS, Event, FallDetector
from logic.incidents import IncidentLog, WallClock, event_record
from logic.io import Frame, read_keypoints
from logic.mqtt_out import MqttPublisher
from logic.pipeline import FramePipeline, Step

logger = logging.getLogger("safestride")


class Runner:
    """Drives pipeline + fall detector and fans events out to log, MQTT and stdout."""

    def __init__(self, cfg: Config, clock: WallClock, camera: str = "builtin0", model: str = "pose-fp16",
                 incidents: IncidentLog | None = None, publisher: MqttPublisher | None = None,
                 out: Any = None) -> None:
        self.cfg = cfg
        self.clock = clock
        self.camera, self.model = camera, model
        self.incidents = incidents
        self.publisher = publisher
        self.out = out
        self.pipe = FramePipeline(cfg)
        self.det = FallDetector(cfg.fall)
        self.frames = 0
        self.last_t: float | None = None
        self.records: list[dict[str, Any]] = []

    def process(self, frame: Frame) -> list[Event]:
        """Feed one input frame."""
        self.frames += 1
        self.last_t = frame.t
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
            self._emit(ev)
        return events

    def _emit(self, ev: Event) -> None:
        rec = event_record(ev, self.clock.iso(ev.ts_s), self.camera, self.model, None)
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
    runner = Runner(cfg, clock, args.camera, args.model, incidents, publisher, out=sys.stdout)

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
    logger.info("done: %d frames, %d events", runner.frames, len(runner.records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
