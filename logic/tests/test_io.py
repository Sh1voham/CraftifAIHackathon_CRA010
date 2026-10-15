"""Tests for logic.io: replay format, live transports and person selection."""
import io as stdio
import json
import logging
import threading
import time
from types import SimpleNamespace

from logic.io import (
    Frame, Person, TrackState, folder_objects, frame_from_dict, mqtt_objects, read_jsonl, read_keypoints,
    select_person, tail_objects, write_jsonl,
)


def _person(hx: float, hy: float, torso: float = 200.0, conf: float = 0.9, bbox=None, pid=None) -> Person:
    kp = {
        "l_shoulder": (hx - 5, hy - torso, conf), "r_shoulder": (hx + 5, hy - torso, conf),
        "l_hip": (hx - 5, hy, conf), "r_hip": (hx + 5, hy, conf),
    }
    return Person(kp=kp, id=pid, bbox=bbox)


def test_roundtrip_with_optional_fields(tmp_path) -> None:
    frames = [
        Frame(t=0.0, persons=[]),
        Frame(t=0.033, persons=[_person(100, 300, bbox=(1.0, 2.0, 3.0, 4.0), pid=7), _person(500, 300)]),
    ]
    path = tmp_path / "f.jsonl"
    write_jsonl(path, frames)
    back = list(read_jsonl(path))
    assert back == frames
    assert back[1].persons[1].id is None and back[1].persons[1].bbox is None


def test_keypoint_list_forms() -> None:
    triples = [[float(i), float(i) + 0.5, 0.9] for i in range(17)]
    flat = [v for tr in triples for v in tr]
    for kp in (triples, flat):
        f = frame_from_dict({"t": 1, "persons": [{"kp": kp}]})
        assert f.persons[0].kp["r_ankle"] == (16.0, 16.5, 0.9)
        assert f.persons[0].kp["nose"] == (0.0, 0.5, 0.9)


def test_malformed_lines_are_skipped_and_logged(tmp_path, caplog) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"t": 0, "persons": []}\nnot json\n{"persons": []}\n\n{"t": 1}\n', encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        frames = list(read_jsonl(path))
    assert [f.t for f in frames] == [0.0, 1.0]
    assert "malformed" in caplog.text and "unparseable" in caplog.text


def test_read_keypoints_file_and_stdin(tmp_path, monkeypatch) -> None:
    frames = [Frame(t=i * 0.03, persons=[_person(100 + i, 300)]) for i in range(5)]
    path = tmp_path / "k.jsonl"
    write_jsonl(path, frames)
    assert list(read_keypoints(path)) == frames
    monkeypatch.setattr("sys.stdin", stdio.StringIO(path.read_text(encoding="utf-8")))
    assert list(read_keypoints("-")) == frames


def test_tail_jsonl_follows_growing_file_with_partial_lines(tmp_path) -> None:
    path = tmp_path / "live.jsonl"
    lines = [json.dumps({"t": i * 0.1, "persons": []}) for i in range(6)]

    def writer() -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for line in lines:
                half = len(line) // 2
                fh.write(line[:half]); fh.flush(); time.sleep(0.02)
                fh.write(line[half:] + "\n"); fh.flush(); time.sleep(0.02)

    th = threading.Thread(target=writer)
    th.start()
    got = list(tail_objects(path, poll_s=0.005, idle_timeout_s=0.5))
    th.join()
    assert [o["t"] for o in got] == [i * 0.1 for i in range(6)]


def test_read_keypoints_follow_mode(tmp_path) -> None:
    path = tmp_path / "live.jsonl"
    write_jsonl(path, [Frame(t=1.0), Frame(t=2.0)])
    frames = list(read_keypoints(path, follow=True, poll_s=0.005, idle_timeout_s=0.2))
    assert [f.t for f in frames] == [0.0, 1.0]  # time is normalised to start at zero


def test_watch_folder_in_order_and_waits_for_unfinished_file(tmp_path) -> None:
    def writer() -> None:
        for i in range(5):
            text = json.dumps({"t": float(i), "persons": []})
            f = tmp_path / f"frame_{i:05d}.json"
            if i == 2:  # write in two parts to mimic a writer caught mid-file
                f.write_text(text[:5], encoding="utf-8"); time.sleep(0.03)
            f.write_text(text, encoding="utf-8")
            time.sleep(0.02)

    th = threading.Thread(target=writer)
    th.start()
    frames = list(read_keypoints(tmp_path, poll_s=0.005, idle_timeout_s=0.4))
    th.join()
    assert [f.t for f in frames] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_folder_skips_permanently_broken_file(tmp_path, caplog) -> None:
    (tmp_path / "a.json").write_text('{"t": 0}', encoding="utf-8")
    (tmp_path / "b.json").write_text('{"t": ', encoding="utf-8")
    (tmp_path / "c.json").write_text('[{"t": 2}, {"t": 3}]', encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        got = list(folder_objects(tmp_path, poll_s=0.005, idle_timeout_s=0.2, settle_s=0.03))
    assert [o["t"] for o in got] == [0, 2, 3]
    assert "b.json" in caplog.text


class FakeMqttClient:
    """Delivers canned payloads as soon as the network loop starts."""

    def __init__(self, payloads: list[bytes]) -> None:
        self.payloads = payloads
        self.subscribed: list[str] = []
        self.on_connect = self.on_message = None
        self.stopped = False

    def connect_async(self, host: str, port: int, keepalive: int) -> None:
        self.addr = (host, port)

    def subscribe(self, topic: str, qos: int = 0) -> None:
        self.subscribed.append(topic)

    def loop_start(self) -> None:
        self.on_connect(self, None, {}, 0, None)
        for p in self.payloads:
            self.on_message(self, None, SimpleNamespace(payload=p))

    def loop_stop(self) -> None:
        self.stopped = True

    def disconnect(self) -> None:
        pass


def test_mqtt_objects_with_fake_client(caplog) -> None:
    client = FakeMqttClient([b'{"t": 1, "persons": []}', b"garbage", b'[{"t": 2}, {"t": 3}]'])
    with caplog.at_level(logging.WARNING):
        got = list(mqtt_objects("h", 1, "pipegen/kp", client=client, poll_s=0.005, idle_timeout_s=0.05))
    assert [o["t"] for o in got] == [1, 2, 3]
    assert client.subscribed == ["pipegen/kp"] and client.stopped
    assert "malformed" in caplog.text


def test_select_person_first_pick_is_largest_torso() -> None:
    state = TrackState()
    small, big = _person(100, 300, torso=80), _person(600, 400, torso=200)
    assert select_person(Frame(t=0.0, persons=[small, big]), state) is big


def test_select_person_tracks_nearest_through_crossing() -> None:
    state = TrackState()
    # A walks right from x=100, B walks left from x=700; follow A the whole way.
    chosen_x = []
    for i in range(30):
        a = _person(100 + 20 * i, 400, torso=200)
        b = _person(700 - 20 * i, 410, torso=210 if i == 0 else 200)
        persons = [b, a] if i % 2 else [a, b]
        if i == 0:
            persons = [a]  # first frame only A is visible
        p = select_person(Frame(t=i / 30, persons=persons), state)
        chosen_x.append(p.kp["l_hip"][0] + 5)
    assert chosen_x == [100 + 20 * i for i in range(30)]


def test_select_person_repicks_after_grace_and_handles_empty() -> None:
    state = TrackState(lost_grace_s=2.0)
    a, b = _person(100, 400, torso=100), _person(600, 400, torso=200)
    assert select_person(Frame(t=0.0, persons=[a]), state) is a
    assert select_person(Frame(t=0.5, persons=[]), state) is None
    assert select_person(Frame(t=1.0, persons=[a, b]), state) is a  # within grace: nearest
    assert select_person(Frame(t=3.5, persons=[a, b]), state) is b  # lost > grace: biggest
    nohip = Person(kp={"nose": (5.0, 5.0, 0.1)})
    assert select_person(Frame(t=4.0, persons=[nohip]), TrackState()) is None
