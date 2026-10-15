"""Tests for logic.overlay (headless)."""
import json

import cv2
import numpy as np

from logic.fall import State
from logic.io import Frame, Person, write_jsonl
from logic.overlay import draw, main
from logic.tests.fixtures import scripted_fall


def _checker(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return (((yy // 4 + xx // 4) % 2) * 255).astype(np.uint8)[..., None].repeat(3, axis=2)


def test_draw_skeleton_status_and_banner() -> None:
    img = np.zeros((300, 400, 3), np.uint8)
    kp = [None] * 17
    kp[5], kp[11] = (200.0, 100.0, 0.9), (200.0, 200.0, 0.9)  # shoulder-hip edge
    out = draw(img, kp, State.FALL_CONFIRMED, 85.0, 2.4, band="HIGH", alert="FALL_CONFIRMED t=3.0s")
    assert out is img
    assert tuple(img[150, 200]) == (0, 0, 255)  # red skeleton line in FALL_CONFIRMED
    assert tuple(img[290, 390]) == (0, 0, 200)  # red alert banner
    assert img[:150, :300].any()  # status text drawn


def test_headless_replay_saves_blurred_overlay_and_evidence(tmp_path) -> None:
    scale = 0.5
    frames = [Frame(t=f.t, persons=[Person(kp={n: (k[0] * scale, k[1] * scale, k[2])
                                                for n, k in p.kp.items()}) for p in f.persons])
              for f in scripted_fall(stand_s=4.0, lie_s=4.5)]
    kp_path = tmp_path / "kp.jsonl"
    write_jsonl(kp_path, frames)
    video = str(tmp_path / "v.avi")
    h, w = 540, 480
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
    bg = _checker(h, w)
    for _ in frames:
        vw.write(bg)
    vw.release()
    out = str(tmp_path / "overlay.avi")
    inc = tmp_path / "inc.jsonl"
    assert main(["--video", video, "--input", str(kp_path), "--headless", "--save", out,
                 "--incidents", str(inc), "--blur-dir", str(tmp_path / "blurred")]) == 0
    rec = json.loads(inc.read_text().splitlines()[0])
    assert rec["event"] == "FALL_CONFIRMED" and rec["frame_saved"]
    evidence = cv2.imread(rec["frame_saved"])
    lying = (slice(460, 485), slice(100, 290))  # lying person (hips ~y 467, ankles ~y 487) at half scale
    assert evidence[lying].std() < 0.5 * bg[lying].std()
    cap = cv2.VideoCapture(out)
    n, last = 0, None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        n, last = n + 1, img
    assert n == len(frames)
    assert last[h - 10, w - 10, 2] > 150 and last[h - 10, w - 10, 0] < 80  # red alert banner (BGR)
    assert last[lying].std() < 0.6 * bg[lying].std()  # person blurred in the saved overlay
