"""Tests for logic.demo (no broker, no window)."""
import json

from logic.config import Config
from logic.demo import main, selftest
from logic.io import write_jsonl
from logic.synth import make_clip


def test_selftest_passes_without_broker(capsys) -> None:
    assert selftest(Config(), "127.0.0.1:1")
    out = capsys.readouterr().out
    assert "SELFTEST PASS" in out and "[SKIP] MQTT delivery" in out and "[FAIL]" not in out


def test_demo_file_source_prints_status_each_second_and_events(tmp_path, capsys) -> None:
    clip = make_clip("fall", "forward_stay", 4, "c")
    path = tmp_path / "c.jsonl"
    write_jsonl(path, clip.frames)
    inc = tmp_path / "inc.jsonl"
    assert main(["--source", f"file:{path}", "--no-mqtt", "--incidents", str(inc)]) == 0
    out = capsys.readouterr().out.splitlines()
    status = [x for x in out if x.startswith("[t=")]
    duration = clip.frames[-1].t - clip.frames[0].t
    assert abs(len(status) - duration) <= 2  # about one line per second of frame time
    assert any(x.startswith("*** FALL_CONFIRMED") for x in out)
    assert json.loads(inc.read_text().splitlines()[0])["event"] == "FALL_CONFIRMED"
    assert "torso" in status[0] and "risk" in status[0]


def test_demo_reports_bad_source_and_missing_frame_size(tmp_path, capsys) -> None:
    assert main(["--source", "nope:x", "--no-mqtt", "--incidents", str(tmp_path / "i.jsonl")]) == 2
    norm = tmp_path / "norm.jsonl"
    norm.write_text("".join(json.dumps({"t": i / 30, "persons": [{"keypoints": [[0.5, 0.5, 0.9]] * 17}]}) + "\n"
                            for i in range(12)), encoding="utf-8")
    assert main(["--source", f"file:{norm}", "--no-mqtt", "--incidents", str(tmp_path / "i.jsonl")]) == 2
    err = capsys.readouterr().err
    assert "unknown --source" in err and "--frame-size" in err
    assert main(["--source", f"file:{norm}", "--no-mqtt", "--frame-size", "1280x720",
                 "--incidents", str(tmp_path / "i.jsonl")]) == 0
