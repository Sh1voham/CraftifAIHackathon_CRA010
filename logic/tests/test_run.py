"""End-to-end tests: pipeline, incident log, MQTT publisher and the run CLI."""
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from logic.config import Config, MqttCfg
from logic.fall import Event
from logic.incidents import IncidentLog, WallClock, event_record
from logic.io import read_jsonl, write_jsonl
from logic.mqtt_out import MqttPublisher
from logic.run import Runner, VideoFrames
from logic.tests.fixtures import scripted_fall, scripted_sit

ROOT = Path(__file__).resolve().parents[2]
START = "2026-10-15T20:41:00.000+05:30"
SCHEMA = ["ts", "event", "confidence", "torso_angle_deg", "hip_drop_speed_Lps", "time_down_s",
          "camera", "model", "frame_saved", "t_s"]


def _replay(frames, **kw) -> Runner:
    r = Runner(Config(), WallClock(datetime.fromisoformat(START)), **kw)
    for f in frames:
        r.process(f)
    r.finish()
    return r


def test_scripted_fall_alerts_once_about_t_hold_after_landing() -> None:
    r = _replay(scripted_fall())
    assert [rec["event"] for rec in r.records] == ["FALL_CONFIRMED"]
    rec = r.records[0]
    assert list(rec) == SCHEMA
    t_fall_start = 56.0
    assert 3.0 <= rec["t_s"] - t_fall_start <= 3.6
    assert rec["torso_angle_deg"] > 75 and rec["hip_drop_speed_Lps"] > 1.2 and rec["confidence"] > 0.8
    assert rec["ts"].startswith("2026-10-15T20:41:0") and rec["ts"].endswith("+05:30")


def test_fast_sit_does_not_alert() -> None:
    assert _replay(scripted_sit()).records == []


def test_wallclock_replay_mapping() -> None:
    c = WallClock(datetime.fromisoformat(START))
    c.observe(100.0)
    c.observe(101.0)
    assert c.iso(103.512) == "2026-10-15T20:41:03.512+05:30"
    assert WallClock(None).iso(1.0)[:2] == "20"  # live: current local time


def test_incident_log_appends_durably(tmp_path) -> None:
    ev = Event(1.0, "FALL_CONFIRMED", 0.8567, 78.44, 2.444, 3.1)
    log = IncidentLog(tmp_path / "sub" / "inc.jsonl")
    log.append(event_record(ev, START, "cam", "m", None))
    log.append(event_record(ev, START, "cam", "m", "blurred/x.jpg"))
    log.close()
    lines = (tmp_path / "sub" / "inc.jsonl").read_bytes().split(b"\n")
    assert lines[-1] == b"" and len(lines) == 3
    rec = json.loads(lines[0])
    assert rec["confidence"] == 0.857 and rec["torso_angle_deg"] == 78.4 and rec["frame_saved"] is None


class FakePub:
    def __init__(self) -> None:
        self.sent = []
        self.on_connect = self.on_disconnect = None

    def reconnect_delay_set(self, a, b): pass
    def max_queued_messages_set(self, n): pass
    def connect_async(self, host, port, keepalive): pass
    def loop_start(self): self.on_connect(self, None, {}, 0, None)
    def loop_stop(self): pass
    def disconnect(self): pass

    def publish(self, topic, payload, qos=0):
        self.sent.append((topic, json.loads(payload), qos))
        return SimpleNamespace(rc=0)


def test_runner_publishes_alerts_with_qos1() -> None:
    fake = FakePub()
    pub = MqttPublisher("h", 1, MqttCfg(), client=fake)
    _replay(scripted_fall(), publisher=pub)
    assert [(t, p["event"], q) for t, p, q in fake.sent] == [("safestride/alerts", "FALL_CONFIRMED", 1)]


def test_unreachable_broker_warns_once_and_never_raises(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="logic.mqtt_out"):
        pub = MqttPublisher("127.0.0.1", 1, MqttCfg(reconnect_min_s=1, reconnect_max_s=1))
        for _ in range(5):
            assert pub.publish("safestride/alerts", {"x": 1}) is False
        pub.close()
    assert sum("unreachable" in r.message or "refused" in r.message for r in caplog.records) == 1


def test_publisher_survives_client_exceptions(caplog) -> None:
    class Broken(FakePub):
        def publish(self, *a, **k):
            raise RuntimeError("socket gone")
    pub = MqttPublisher("h", 1, MqttCfg(), client=Broken())
    with caplog.at_level(logging.WARNING):
        assert pub.publish("t", {}) is False
    assert "socket gone" in caplog.text


def _cli(args, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "logic.run", *args], cwd=ROOT, capture_output=True,
                          timeout=60, **kw)


def test_cli_replay_is_byte_identical(tmp_path) -> None:
    clip = tmp_path / "fall.jsonl"
    write_jsonl(clip, scripted_fall())
    outs = []
    for i in range(2):
        inc = tmp_path / f"inc{i}.jsonl"
        res = _cli(["--input", str(clip), "--incidents", str(inc), "--start-wallclock", START])
        assert res.returncode == 0, res.stderr.decode()
        outs.append((inc.read_bytes(), res.stdout))
    assert outs[0] == outs[1]
    assert json.loads(outs[0][0].splitlines()[0])["event"] == "FALL_CONFIRMED"


def test_cli_stdin_mode_with_unreachable_mqtt(tmp_path) -> None:
    clip = tmp_path / "fall.jsonl"
    write_jsonl(clip, scripted_fall())
    inc = tmp_path / "inc.jsonl"
    res = _cli(["--input", "-", "--incidents", str(inc), "--mqtt", "127.0.0.1:1",
                "--camera", "cam9", "--model", "pose-int8"], input=clip.read_bytes())
    assert res.returncode == 0, res.stderr.decode()
    rec = json.loads(res.stdout.decode().splitlines()[0])
    assert rec["event"] == "FALL_CONFIRMED" and rec["camera"] == "cam9" and rec["model"] == "pose-int8"
    assert json.loads(inc.read_text())["event"] == "FALL_CONFIRMED"
    assert b"unreachable" in res.stderr or b"refused" in res.stderr


@pytest.mark.parametrize("bad", ["not json\n", '{"t": "x"}\n'])
def test_cli_survives_bad_input_lines(tmp_path, bad) -> None:
    clip = tmp_path / "fall.jsonl"
    write_jsonl(clip, scripted_fall())
    lines = clip.read_text().splitlines(keepends=True)
    clip.write_text("".join(lines[:10] + [bad] + lines[10:]))
    res = _cli(["--input", str(clip), "--incidents", str(tmp_path / "i.jsonl"), "--start-wallclock", START])
    assert res.returncode == 0 and b"FALL_CONFIRMED" in res.stdout


def test_alert_frame_is_saved_blurred(tmp_path) -> None:
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, size=(1080, 800, 3), dtype=np.uint8)
    asked = []

    def source(t_rel):
        asked.append(t_rel)
        return noise

    r = _replay(scripted_fall(), frame_source=source, blur_dir=tmp_path / "blurred")
    rec = r.records[0]
    assert rec["frame_saved"].endswith("_fall_confirmed.jpg") and asked == [pytest.approx(rec["t_s"] - 50.0, abs=1e-3)]  # t_s is rounded
    saved = cv2.imread(rec["frame_saved"])
    hip_region = (slice(900, 960), slice(300, 600))  # where the person lies
    assert saved[hip_region].std() < 0.3 * noise[hip_region].std()


def test_video_frames_random_access(tmp_path) -> None:
    path = str(tmp_path / "v.avi")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(20):
        vw.write(np.full((48, 64, 3), i * 10, dtype=np.uint8))
    vw.release()
    vf = VideoFrames(path)
    assert abs(int(vf.at(1.0).mean()) - 100) <= 3
    assert vf.at(100.0) is None


def test_cli_record_writes_replayable_frames(tmp_path) -> None:
    clip = tmp_path / "fall.jsonl"
    frames = scripted_fall()
    write_jsonl(clip, frames)
    rec = tmp_path / "rec.jsonl"
    res = _cli(["--input", "-", "--incidents", str(tmp_path / "i.jsonl"), "--record", str(rec)],
               input=clip.read_bytes())
    assert res.returncode == 0, res.stderr.decode()
    back = list(read_jsonl(rec))
    assert len(back) == len(frames)
    assert [round(f.t, 6) for f in back] == [round(f.t - frames[0].t, 6) for f in frames]  # normalised time
    assert back[5].persons == frames[5].persons


def _gait_runner(clip_frames, **kw):
    from logic.run import GaitMonitor

    fake = FakePub()
    r = Runner(Config(), WallClock(datetime.fromisoformat(START)), publisher=MqttPublisher("h", 1, MqttCfg(),
               client=fake), gait=GaitMonitor(Config(), kw.pop("calib", None)), **kw)
    for f in clip_frames:
        r.process(f)
    r.finish()
    return r, fake


def test_runner_publishes_gait_risk_summaries(tmp_path) -> None:
    from logic.risk import BaselineStore
    from logic.synth import make_walk

    clip = make_walk(2, speed=0.8, cadence=95, duration=12.0)
    store = BaselineStore(tmp_path / "base.json", Config().risk)
    r, fake = _gait_runner(clip.frames, baselines=store, person="alice")
    risk_msgs = [p for t, p, _ in fake.sent if t == "safestride/risk"]
    assert risk_msgs and risk_msgs == r.risk_records
    final = risk_msgs[-1]
    assert final["final"] and final["person"] == "alice" and final["band"] in ("LOW", "MODERATE", "HIGH")
    assert final["walk_speed_mps"] == pytest.approx(0.8, rel=0.1) and final["calibrated"] is False
    assert final["baseline_mps"] == final["walk_speed_mps"] and final["drift_alert"] is False
    assert json.loads((tmp_path / "base.json").read_text())["alice"]["history"]


def test_runner_gait_includes_tug_and_ignores_falls() -> None:
    from logic.synth import make_tug

    r, _ = _gait_runner(make_tug(3).frames)
    assert r.risk_records[-1]["tug_s"] is not None and r.risk_records[-1]["final"]
    r2, fake = _gait_runner(scripted_fall())
    assert r2.risk_records == [] and [p["event"] for t, p, _ in fake.sent if t == "safestride/alerts"] == \
           ["FALL_CONFIRMED"]


def test_cli_gait_with_calibration_and_baseline(tmp_path) -> None:
    from logic.synth import make_walk

    clip = make_walk(4, speed=1.0, cadence=100, duration=9.0)
    path = tmp_path / "walk.jsonl"
    write_jsonl(path, clip.frames)
    m = clip.label["marker_px"]
    base = tmp_path / "b.json"
    res = _cli(["--input", str(path), "--incidents", str(tmp_path / "i.jsonl"), "--person", "bob",
                "--baselines", str(base), "--calib", f"{m[0]},500,{m[1]},500,3.0"])
    assert res.returncode == 0, res.stderr.decode()
    assert b"(session)" in res.stderr and b"risk" in res.stderr
    hist = json.loads(base.read_text())["bob"]["history"]
    assert hist[0] == pytest.approx(1.0, rel=0.1)
