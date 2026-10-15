"""Tests for io.auto_adapt / adapt_stream and the Frame-yielding transports."""
import json
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from logic.io import (
    KEYPOINT_NAMES, AdaptOpts, adapt_stream, auto_adapt, mqtt_frames, read_keypoints, tail_jsonl, watch_folder,
)

TRIPLES = [[100.0 + i, 200.0 + 2 * i, 0.5 + i / 100] for i in range(17)]


def _check(frame, x0: float = 100.0, y0: float = 200.0, scale: float = 1.0) -> None:
    p = frame.persons[0]
    assert set(p.kp) == set(KEYPOINT_NAMES)
    assert p.kp["nose"] == pytest.approx(((x0) * scale, y0 * scale, 0.5))
    assert p.kp["r_ankle"] == pytest.approx(((x0 + 16) * scale, (y0 + 32) * scale, 0.66))


def test_layout_a_replay_format() -> None:
    f = auto_adapt({"t": 1.5, "persons": [{"id": 3, "bbox": [1, 2, 300, 400],
                                           "kp": {n: TRIPLES[i] for i, n in enumerate(KEYPOINT_NAMES)}}]})
    _check(f)
    assert f.t == 1.5 and f.persons[0].id == 3 and f.persons[0].bbox == (1.0, 2.0, 300.0, 400.0)


@pytest.mark.parametrize("list_key", ["persons", "objects", "detections", "people"])
@pytest.mark.parametrize("kp_key", ["keypoints", "kp", "landmarks"])
def test_layout_b_dict_by_name_with_aliases(list_key, kp_key) -> None:
    aliases = {"l_shoulder": "left_shoulder", "r_hip": "RIGHT_HIP", "l_ankle": "leftAnkle"}
    kp = {aliases.get(n, n): {"x": TRIPLES[i][0], "y": TRIPLES[i][1], "score": TRIPLES[i][2]}
          for i, n in enumerate(KEYPOINT_NAMES)}
    f = auto_adapt({"ts": 2.0, list_key: [{"track_id": 7, kp_key: kp}]})
    _check(f)
    assert f.persons[0].id == 7


def test_layout_c_list_of_triplets() -> None:
    _check(auto_adapt({"timestamp": 0.0, "detections": [{"object_id": 1, "keypoints": TRIPLES}]}))


def test_layout_d_flat_list() -> None:
    flat = [v for tr in TRIPLES for v in tr]
    _check(auto_adapt({"t": 0.0, "people": [{"keypoints": flat}]}))


def test_layout_e_separate_arrays() -> None:
    xs, ys, cs = ([tr[k] for tr in TRIPLES] for k in range(3))
    _check(auto_adapt({"t": 0.0, "objects": [{"x": xs, "y": ys, "score": cs}]}))
    _check(auto_adapt({"t": 0.0, "objects": [{"keypoints": {"x": xs, "y": ys, "scores": cs}}]}))


def test_bbox_formats() -> None:
    def box(b):
        return auto_adapt({"t": 0, "persons": [{"bbox": b, "keypoints": TRIPLES}]}).persons[0].bbox
    assert box([90, 190, 130, 250]) == (90.0, 190.0, 130.0, 250.0)  # xyxy contains the keypoints
    assert box([90, 190, 40, 60]) == (90.0, 190.0, 130.0, 250.0)  # xywh (w < x)
    assert box({"left": 90, "top": 190, "width": 40, "height": 60}) == (90.0, 190.0, 130.0, 250.0)


def test_frame_number_timestamps_use_fps(caplog) -> None:
    objs = [{"frame_num": i, "persons": [{"keypoints": TRIPLES}]} for i in range(5)]
    with caplog.at_level(logging.WARNING):
        frames = list(adapt_stream(objs, AdaptOpts(fps=25.0)))
    assert [f.t for f in frames] == pytest.approx([0.0, 0.04, 0.08, 0.12, 0.16])
    assert "frame number" in caplog.text


@pytest.mark.parametrize("unit, scale", [("s", 1.0), ("ms", 1e3), ("us", 1e6), ("ns", 1e9)])
def test_timestamp_unit_inference_and_normalisation(unit, scale, caplog) -> None:
    base = 1_700_000_000.0 if unit == "s" else 1234.5  # epoch seconds or a large PTS offset
    objs = [{"pts": (base + i / 30) * scale, "persons": [{"keypoints": TRIPLES}]} for i in range(12)]
    with caplog.at_level(logging.INFO):
        frames = list(adapt_stream(objs, AdaptOpts()))
    assert frames[0].t == 0.0
    assert frames[-1].t == pytest.approx(11 / 30, abs=1e-6)
    assert f"in {unit}" in caplog.text and "layout" in caplog.text


def test_explicit_unit_overrides_inference() -> None:
    objs = [{"t": i * 33.0, "persons": []} for i in range(3)]
    frames = list(adapt_stream(objs, AdaptOpts(ts_unit="ms")))
    assert [f.t for f in frames] == pytest.approx([0.0, 0.033, 0.066])


def test_normalised_coordinates_scaled_by_frame_size() -> None:
    norm = [[(100.0 + i) / 1280, (200.0 + 2 * i) / 720, 0.5 + i / 100] for i in range(17)]
    objs = [{"t": i / 30, "persons": [{"keypoints": norm, "bbox": [0.05, 0.25, 0.1, 0.35]}]} for i in range(3)]
    frames = list(adapt_stream(objs, AdaptOpts(frame_size=(1280, 720))))
    _check(frames[0])
    assert frames[0].persons[0].bbox == pytest.approx((64.0, 180.0, 128.0, 252.0))
    in_obj = [{"t": i / 30, "width": 1280, "height": 720, "persons": [{"keypoints": norm}]} for i in range(3)]
    _check(next(adapt_stream(in_obj, AdaptOpts())))  # frame size taken from the object
    with pytest.raises(ValueError, match="--frame-size"):
        list(adapt_stream([{"t": 0.0, "persons": [{"keypoints": norm}]}], AdaptOpts()))
    px = next(adapt_stream([{"t": 0.0, "persons": [{"keypoints": norm}]}], AdaptOpts(coords="px")))
    assert px.persons[0].kp["nose"][0] == pytest.approx(100 / 1280)


def test_kp_map_for_non_coco_model() -> None:
    # An 18-point model (OpenPose-like order): index -> canonical name, unmapped indices dropped.
    kp_map = {"0": "nose", "2": "r_shoulder", "5": "l_shoulder", "8": "r_hip", "11": "l_hip"}
    pts = [[float(i), float(10 * i), 0.9] for i in range(18)]
    f = auto_adapt({"t": 0, "persons": [{"keypoints": pts}]}, AdaptOpts(kp_map=kp_map))
    assert set(f.persons[0].kp) == set(kp_map.values())
    assert f.persons[0].kp["l_hip"] == (11.0, 110.0, 0.9)


def test_bad_objects_are_skipped(caplog) -> None:
    objs = [{"t": 0.0, "persons": []}, "garbage", {"persons": []}, {"t": 0.1, "persons": []}]
    with caplog.at_level(logging.WARNING):
        frames = list(adapt_stream(objs, AdaptOpts(ts_unit="s")))
    assert [f.t for f in frames] == pytest.approx([0.0, 0.1])


def test_tail_jsonl_handles_truncation(tmp_path) -> None:
    path = tmp_path / "live.jsonl"
    path.write_text("".join(json.dumps({"t": i / 30, "persons": []}) + "\n" for i in range(3)), encoding="utf-8")

    def rewrite() -> None:
        time.sleep(0.15)
        path.write_text(json.dumps({"t": 5.0, "persons": []}) + "\n", encoding="utf-8")  # truncated and rewritten

    th = threading.Thread(target=rewrite)
    th.start()
    frames = list(tail_jsonl(path, poll_s=0.01, idle_timeout_s=0.5, opts=AdaptOpts(ts_unit="s")))
    th.join()
    assert [round(f.t, 3) for f in frames] == [0.0, 0.033, 0.067, 5.0]


def test_watch_folder_yields_frames(tmp_path) -> None:
    for i in range(4):
        (tmp_path / f"f_{i:04d}.json").write_text(json.dumps({"pts": 1e9 * (7 + i / 30), "objects": [
            {"keypoints": TRIPLES}]}), encoding="utf-8")
    frames = list(watch_folder(tmp_path, poll_s=0.01, idle_timeout_s=0.1))
    assert [round(f.t, 4) for f in frames] == [0.0, 0.0333, 0.0667, 0.1]
    assert list(read_keypoints(tmp_path, poll_s=0.01, idle_timeout_s=0.1))[1].t == pytest.approx(1 / 30)


def test_mqtt_frames_with_fake_client() -> None:
    class Fake:
        def __init__(self) -> None:
            self.on_connect = self.on_message = self.on_disconnect = None

        def reconnect_delay_set(self, a, b): pass
        def connect_async(self, host, port, keepalive): pass
        def subscribe(self, topic, qos=0): pass
        def loop_stop(self): pass
        def disconnect(self): pass

        def loop_start(self) -> None:
            self.on_connect(self, None, {}, 0, None)
            for i in range(3):
                msg = {"ts": 1000 * (5 + i / 30), "people": [{"kp": TRIPLES}]}
                self.on_message(self, None, SimpleNamespace(payload=json.dumps(msg).encode()))

    frames = list(mqtt_frames("h", 1, "pipegen/kp", timeout_s=0.05, client=Fake()))
    assert [round(f.t, 4) for f in frames] == [0.0, 0.0333, 0.0667]  # ms inferred
    _check(frames[0])
