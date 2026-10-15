"""OpenCV overlay: skeleton, fall state, torso angle, hip speed, risk band, last alert.

    python -m logic.overlay --video clip.mp4 --input clip.jsonl            # replay
    pipegen-app | python -m logic.overlay --camera 0 --input -             # live
    python -m logic.overlay --video clip.mp4 --input clip.jsonl --headless --save demo.mp4

The overlay also runs the full decision layer (incident log, MQTT). Any video it
saves has every person blurred before the skeleton is drawn.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from logic.config import Config, load_config
from logic.fall import ALERT_KINDS, State
from logic.incidents import IncidentLog, WallClock
from logic.io import KEYPOINT_NAMES, Frame, add_adapter_args, opts_from_args, read_keypoints
from logic.mqtt_out import MqttPublisher
from logic.privacy import blur_people
from logic.run import GaitMonitor, Runner

logger = logging.getLogger("safestride.overlay")

EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16)]
STATE_COLOURS = {  # BGR
    State.STANDING: (80, 200, 80), State.FALLING: (0, 200, 255), State.DOWN: (0, 140, 255),
    State.FALL_CONFIRMED: (0, 0, 255), State.PROLONGED_DOWN: (0, 0, 255),
}
BAND_COLOURS = {"LOW": (80, 200, 80), "MODERATE": (0, 200, 255), "HIGH": (0, 0, 255)}


def draw(image: np.ndarray, kp: list[Any] | None, state: State, phi: float | None, v: float | None,
         band: str | None = None, alert: str | None = None) -> np.ndarray:
    """Draw skeleton and status onto ``image`` (in place) and return it.

    ``kp`` is a COCO-17 list of (x, y, ...) or None entries.
    """
    colour = STATE_COLOURS.get(state, (255, 255, 255))
    if kp is not None:
        for a, b in EDGES:
            if kp[a] is not None and kp[b] is not None:
                cv2.line(image, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])), colour, 3)
        for k in kp:
            if k is not None:
                cv2.circle(image, (int(k[0]), int(k[1])), 4, (255, 255, 255), -1)
    lines = [f"state: {state.value}",
             f"torso angle: {'-' if phi is None else f'{phi:.0f} deg'}",
             f"hip speed: {'-' if v is None else f'{v:+.2f} L/s'}"]
    if band is not None:
        lines.append(f"fall-risk band: {band}")
    for i, text in enumerate(lines):
        c = BAND_COLOURS.get(band, colour) if text.startswith("fall-risk") else colour
        cv2.putText(image, text, (12, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (12, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
    if alert is not None:
        h, w = image.shape[:2]
        bg = (0, 0, 200) if any(k in alert for k in ALERT_KINDS) else (0, 140, 0)
        cv2.rectangle(image, (0, h - 50), (w, h), bg, -1)
        cv2.putText(image, alert, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return image


class OverlayState:
    """Runs keypoint frames through the decision layer and keeps what to draw."""

    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self.kp: list[Any] | None = None
        self.persons: list[Any] = []
        self.alert: str | None = None
        self.lock = threading.Lock()

    def feed(self, frame: Frame) -> None:
        """Process one keypoint frame."""
        events = self.runner.process(frame)
        with self.lock:
            self.persons = list(frame.persons)
            anchor = self.runner.pipe.track.anchor
            # draw the followed person's latest raw keypoints (cleaned ones lag by a few frames)
            chosen = min(frame.persons, key=lambda p: _dist(p, anchor)) if frame.persons else None
            self.kp = [chosen.kp.get(n) for n in KEYPOINT_NAMES] if chosen else None
            if self.kp is not None:
                c_min = self.runner.cfg.filters.c_min
                self.kp = [k if k is not None and k[2] >= c_min else None for k in self.kp]
            for ev in events:
                self.alert = f"{ev.kind}  t={ev.ts_s:.1f}s  confidence={ev.confidence:.2f}"

    def render(self, image: np.ndarray, blur: bool) -> np.ndarray:
        """Overlay the latest state on a copy of ``image``, blurring people first if asked."""
        with self.lock:
            cfg = self.runner.cfg
            out = blur_people(image, self.persons, cfg.privacy, cfg.filters.c_min) if blur else image.copy()
            f = self.runner.det.last_features
            band = self.runner.risk_records[-1]["band"] if self.runner.risk_records else None
            return draw(out, self.kp, self.runner.det.state, f.phi if f else None, f.v if f else None, band,
                        self.alert)


def _dist(p: Any, anchor: tuple[float, float] | None) -> float:
    if anchor is None:
        return 0.0
    lh, rh = p.kp.get("l_hip"), p.kp.get("r_hip")
    if lh is not None and rh is not None:
        x, y = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
    elif p.bbox is not None:
        x, y = (p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2
    else:
        return float("inf")
    return float(np.hypot(x - anchor[0], y - anchor[1]))


def replay_video(cap: Any, frames: Iterator[Frame], state: OverlayState) -> Iterator[np.ndarray]:
    """Yield video frames, feeding keypoint frames up to each video timestamp first.

    The first keypoint frame is aligned with the start of the video.
    """
    pending: Frame | None = None
    t0: float | None = None
    while True:
        ok, img = cap.read()
        if not ok:
            return
        t_vid = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        while True:
            if pending is None:
                pending = next(frames, None)
                if pending is None:
                    break
                t0 = pending.t if t0 is None else t0
            if pending.t - t0 > t_vid + 1e-6:  # type: ignore[operator]
                break
            try:
                state.feed(pending)
            except Exception:  # noqa: BLE001
                logger.exception("t=%.3f: error while processing frame", pending.t)
            pending = None
        yield img


def live_camera(cap: Any, frames: Iterator[Frame], state: OverlayState) -> Iterator[np.ndarray]:
    """Yield camera frames while a background thread feeds keypoints."""
    def pump() -> None:
        for f in frames:
            try:
                state.feed(f)
            except Exception:  # noqa: BLE001
                logger.exception("t=%.3f: error while processing frame", f.t)

    threading.Thread(target=pump, daemon=True).start()
    while True:
        ok, img = cap.read()
        if not ok:
            return
        yield img


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m logic.overlay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="video file to replay")
    src.add_argument("--camera", type=int, help="live camera index")
    ap.add_argument("--input", required=True, help="keypoints: JSONL file, '-', folder or mqtt://host:port/topic")
    ap.add_argument("--config", default=None)
    ap.add_argument("--mqtt", default=None)
    ap.add_argument("--incidents", default="incidents.jsonl")
    ap.add_argument("--blur-dir", default="blurred")
    ap.add_argument("--camera-name", default="builtin0")
    ap.add_argument("--model", default="pose-fp16")
    ap.add_argument("--save", default=None, help="write the overlay video here (people are blurred)")
    ap.add_argument("--headless", action="store_true", help="do not open a window")
    add_adapter_args(ap)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")

    cfg: Config = load_config(args.config)
    publisher = None
    if args.mqtt:
        host, _, port = args.mqtt.partition(":")
        publisher = MqttPublisher(host, int(port or 1883), cfg.mqtt)
    clock = WallClock(datetime.now().astimezone() if args.video else None)
    incidents = IncidentLog(args.incidents)
    current: dict[str, np.ndarray] = {}
    runner = Runner(cfg, clock, args.camera_name, args.model, incidents, publisher,
                    frame_source=lambda _t: current.get("img"), blur_dir=args.blur_dir, gait=GaitMonitor(cfg))
    state = OverlayState(runner)
    frames = read_keypoints(args.input, poll_s=cfg.io.poll_s, settle_s=cfg.io.file_settle_s,
                            opts=opts_from_args(args, cfg.io))
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise OSError(f"cannot open {'video ' + args.video if args.video else f'camera {args.camera}'}")
    fps = cap.get(cv2.CAP_PROP_FPS) or cfg.io.video_fps_fallback
    images = replay_video(cap, frames, state) if args.video else live_camera(cap, frames, state)
    writer = None
    try:
        for img in images:
            current["img"] = img
            if args.save:
                if writer is None:
                    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (img.shape[1], img.shape[0]))
                writer.write(state.render(img, blur=True))
            if not args.headless:
                cv2.imshow("SafeStride", state.render(img, blur=False))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        runner.finish()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if publisher is not None:
            publisher.close()
        incidents.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
