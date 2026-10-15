"""PipeGen 0.5.1 video_json_export format (real sample from our yolo11s-pose pipeline)."""
import json
from datetime import datetime
from pathlib import Path

from logic.config import Config
from logic.incidents import WallClock
from logic.io import KEYPOINT_NAMES, read_keypoints
from logic.run import Runner
from logic.synth import make_clip

SAMPLE = Path(__file__).resolve().parents[2] / "docs" / "pipegen_metadata_sample.json"


def test_real_pipegen_sample_parses() -> None:
    (frame,) = list(read_keypoints(SAMPLE))
    p = frame.persons[0]
    assert frame.t == 0.0 and p.id == 1
    assert p.bbox == (340.5, 4.8, 340.5 + 244.0, 4.8 + 708.8)
    assert p.kp["l_shoulder"][:2] == (541.0, 164.5) and p.kp["r_ankle"][:2] == (448.5, 754.0)
    assert "nose" not in p.kp  # 16-point export: COCO indices 1..16
    assert p.kp["l_hip"][1] < p.kp["l_knee"][1] < p.kp["l_ankle"][1]  # body order is plausible


def _to_pipegen(frames) -> dict:
    out = []
    for i, f in enumerate(frames):
        shapes = []
        for p in f.persons:
            pts = [list(p.kp.get(n, (0.0, 0.0, 0.0))[:2]) for n in KEYPOINT_NAMES[1:]]
            shapes.append({"kind": "skeleton", "points": pts, "label": "person", "id": 1, "score": 0.9})
        out.append({"frame": i, "source_id": 0, "pts_ns": int(f.t * 1e9), "width": 1920, "height": 1080,
                    "objects": [], "shapes": shapes, "fields": {"timestamp_s": round(f.t, 3)}})
    return {"pipeline": "video_json_export", "frames": out}


def test_fall_detected_from_pipegen_format(tmp_path) -> None:
    clip = make_clip("fall", "forward_stay", 3, "f")
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(_to_pipegen(clip.frames)), encoding="utf-8")
    r = Runner(Config(), WallClock(datetime(2026, 10, 15).astimezone()))
    for f in read_keypoints(path):
        r.process(f)
    r.finish()
    assert [x["event"] for x in r.records] == ["FALL_CONFIRMED"]
