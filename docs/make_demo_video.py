"""Build docs/demo_video.mp4 from the PipeGen screenshots and a decision-layer replay.

    python docs/make_demo_video.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logic.config import Config  # noqa: E402
from logic.fall import ALERT_KINDS  # noqa: E402
from logic.incidents import WallClock  # noqa: E402
from logic.io import KEYPOINT_NAMES, read_jsonl  # noqa: E402
from logic.overlay import draw  # noqa: E402
from logic.run import GaitMonitor, Runner  # noqa: E402

W, H, FPS = 1280, 720, 30
OUT = ROOT / "docs" / "demo_video.mp4"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(img: np.ndarray, s: str, y: int, scale: float = 0.9, colour=(255, 255, 255), x: int = 40) -> None:
    cv2.putText(img, s, (x, y), FONT, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, colour, 2, cv2.LINE_AA)


def fit(img: np.ndarray, top: int = 90) -> np.ndarray:
    canvas = np.full((H, W, 3), 18, np.uint8)
    h, w = img.shape[:2]
    s = min(W / w, (H - top - 20) / h)
    r = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    y0, x0 = top, (W - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return canvas


def card(lines: list[tuple[str, float]], seconds: float, vw: cv2.VideoWriter) -> None:
    img = np.full((H, W, 3), 18, np.uint8)
    y = 200
    for s, sc in lines:
        text(img, s, y, sc, (230, 230, 230) if sc < 1.2 else (90, 220, 200))
        y += int(70 * max(sc, 0.8))
    for _ in range(int(seconds * FPS)):
        vw.write(img)


def slide(path: Path, title: str, sub: str, seconds: float, vw: cv2.VideoWriter) -> None:
    img = cv2.imread(str(path))
    if img is None:
        return
    frame = fit(img)
    text(frame, title, 45, 1.0, (90, 220, 200))
    text(frame, sub, 80, 0.65)
    for _ in range(int(seconds * FPS)):
        vw.write(frame)


def replay(clip: Path, title: str, vw: cv2.VideoWriter, person: str | None = None) -> None:
    cfg = Config()
    frames = list(read_jsonl(clip))
    xs = [k[0] for f in frames for p in f.persons for k in p.kp.values() if k[2] > 0]
    ys = [k[1] for f in frames for p in f.persons for k in p.kp.values() if k[2] > 0]
    cw, ch = int(max(xs) + 100), int(max(ys) + 100)
    runner = Runner(cfg, WallClock(datetime(2026, 10, 15, 21, 30).astimezone()), gait=GaitMonitor(cfg), person=None)
    t0 = frames[0].t
    next_out = 0.0
    alert_line = None
    for f in frames:
        runner.process(f)
        if runner.records and runner.records[-1]["event"] in ALERT_KINDS:
            alert_line = runner.records[-1]
        while f.t - t0 >= next_out:
            img = np.full((ch, cw, 3), 30, np.uint8)
            kp = None
            if f.persons:
                kp = [f.persons[0].kp.get(n) for n in KEYPOINT_NAMES]
                kp = [k if k is not None and k[2] >= cfg.filters.c_min else None for k in kp]
            fe = runner.det.last_features
            band = runner.risk_records[-1]["band"] if runner.risk_records else None
            alert = None
            if runner.records:
                r = runner.records[-1]
                alert = f"{r['event']}  confidence {r['confidence']:.2f}  -> incidents.jsonl + MQTT safestride/alerts"
            draw(img, kp, runner.det.state, fe.phi if fe else None, fe.v if fe else None, band, alert)
            frame = fit(img, top=100)
            text(frame, title, 45, 1.0, (90, 220, 200))
            sub = f"SafeStride decision layer  |  t = {f.t - t0:5.1f} s"
            if runner.risk_records:
                rr = runner.risk_records[-1]
                sub += f"  |  walk {rr['walk_speed_mps']} m/s, symmetry {rr['symmetry_pct']} %, risk {rr['band']}"
            text(frame, sub, 82, 0.65)
            if alert_line is not None:
                text(frame, '{"event": "%s", "confidence": %.3f, "torso_angle_deg": %.1f, "time_down_s": %.2f}' % (
                    alert_line["event"], alert_line["confidence"], alert_line["torso_angle_deg"],
                    alert_line["time_down_s"]), H - 70, 0.55, (180, 255, 180))
            vw.write(frame)
            next_out += 1.0 / FPS


def main() -> None:
    vw = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    shots, images = ROOT / "screenshots", ROOT / "docs" / "images"
    card([("SafeStride", 2.0), ("Fall detection and gait-based fall-risk screening", 0.9),
          ("Built on CraftifAI PipeGen  |  Track A  |  Problem #2: Fall Detection and Elder Care", 0.7),
          ("Edge GPU: NVIDIA GeForce GTX 1650  |  TensorRT FP16  |  everything on-device", 0.7)], 4, vw)
    slide(shots / "08_requirement_prompt.jpeg", "1. Requirement in plain English",
          "Pose estimation -> 17 COCO keypoints -> fall rule -> JSON + live display, blur before saving", 4, vw)
    slide(images / "flow_draft_1.png", "2. PipeGen flow draft",
          "video input -> human pose estimation (17 COCO keypoints) -> person tracking -> fall detection", 4, vw)
    slide(images / "pipeline_ir_node_graph.png", "3. Pipeline IR node graph",
          "Generated by PipeGen, approved before compilation", 4, vw)
    slide(images / "tensorrt_fp16_compiled.png", "4. Model: yolo11s-pose -> TensorRT FP16 engine",
          "23.5 MB, 272 layers, built in 61.3 s on the GTX 1650", 4, vw)
    slide(images / "tensorrt_fp16_speed_report.png", "5. PipeGen speed report",
          "FP16: 19.4 ms/frame, ~52 FPS, 24.8 W  (camera needs 30 FPS)", 5, vw)
    slide(shots / "19_pipeline_running_pose_tracking_blur.jpeg", "6. Running live on our recorded clip",
          "DeepStream app: skeleton, person box, tracker ID 1, person blurred for privacy", 5, vw)
    slide(shots / "20_pipeline_running_pose_tracking_blur_2.jpeg", "6. Running live on our recorded clip",
          "~19-20 FPS on screen on a 4 GB laptop GPU", 4, vw)
    card([("7. SafeStride decision layer (logic/, Python, CPU)", 1.0),
          ("keypoints -> confidence gate, gap repair, One Euro smoothing, torso-length scale", 0.7),
          ("-> state machine STANDING > FALLING > DOWN > FALL_CONFIRMED (3 s hold)", 0.7),
          ("-> incidents.jsonl + MQTT safestride/alerts; gait + risk band on safestride/risk", 0.7),
          ("Next: a staged fall replayed through the decision layer", 0.7)], 4, vw)
    replay(ROOT / "docs" / "demo_fall.jsonl", "8. Fall: hips drop fast, person stays down -> alert after ~3 s", vw)
    replay(ROOT / "docs" / "demo_walk.jsonl", "9. Walking: speed, step symmetry and fall-risk band", vw)
    card([("Honest limits", 1.3), ("Single 2D camera, narrow view; thresholds are starting values", 0.75),
          ("Tested on synthetic clips + our recording; more real staged falls needed", 0.75),
          ("Screening aid, not a medical device", 0.75),
          ("github.com/Sh1voham/CraftifAIHackathon_SafeStride", 0.8)], 4, vw)
    vw.release()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
