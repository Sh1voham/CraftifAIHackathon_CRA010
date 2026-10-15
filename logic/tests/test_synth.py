"""Tests for logic.synth: determinism, noise model, labels and motion realism."""
import json
from collections import Counter

import numpy as np
import pytest

from logic.config import Config
from logic.io import frame_to_dict, read_jsonl
from logic.pipeline import FramePipeline
from logic.synth import dataset_plan, generate_dataset, make_clip, make_walk, save_dataset


def _features(clip):
    p = FramePipeline(Config())
    out = []
    for f in clip.frames:
        out += [s.features for s in p.push(f)]
    return out + [s.features for s in p.flush()]


def test_same_seed_same_clip_and_different_seed_differs() -> None:
    a, b = make_clip("fall", "forward_stay", 7, "a"), make_clip("fall", "forward_stay", 7, "b")
    assert [frame_to_dict(f) for f in a.frames] == [frame_to_dict(f) for f in b.frames]
    assert a.label == b.label
    c = make_clip("fall", "forward_stay", 8, "c")
    assert [frame_to_dict(f) for f in a.frames[:5]] != [frame_to_dict(f) for f in c.frames[:5]]


def test_timestamps_irregular_with_drops_and_offset() -> None:
    clip = make_clip("standing", "quiet", 3, "s")
    t = np.array([f.t for f in clip.frames])
    dt = np.diff(t)
    assert t[0] > 9.0  # clips do not start at 0
    assert (dt > 0).all()
    assert 0.001 < dt[dt < 0.05].std() < 0.004  # +/-3 ms jitter
    assert dt.max() > 1.5 / 30  # at least one dropped frame


def test_noise_model_dropouts_and_confidences() -> None:
    clip = make_clip("standing", "quiet", 11, "s")
    vals = [k for f in clip.frames for p in f.persons[:1] for k in p.kp.values()]
    zeros = sum(1 for k in vals if k[2] == 0.0)
    assert 0.015 < zeros / len(vals) < 0.05
    confs = [k[2] for k in vals if k[2] > 0]
    assert min(confs) >= 0.5 and max(confs) <= 0.95


def test_dataset_mix_and_labels() -> None:
    plan = dataset_plan(200, 0)
    c = Counter((k, v) for k, v, _ in plan)
    assert len(plan) == 200 and len({s for *_, s in plan}) == 200
    falls = sum(n for (k, _), n in c.items() if k == "fall")
    assert 85 <= falls <= 95
    assert {v for (k, v) in c if k == "fall"} == {f"{d}_{r}" for d in ("forward", "backward", "sideways")
                                                   for r in ("stay", "recover")}
    assert c[("lie_down_slow", "stay")] == 10 and c[("lie_down_slow", "brief")] == 10


def test_save_dataset_roundtrip(tmp_path) -> None:
    clips = generate_dataset(4, 5)
    save_dataset(clips, tmp_path)
    labels = json.loads((tmp_path / "labels.json").read_text())
    assert set(labels) == {c.name for c in clips}
    first = clips[0]
    assert [frame_to_dict(f) for f in read_jsonl(tmp_path / f"{first.name}.jsonl")] == \
           [frame_to_dict(f) for f in first.frames]


@pytest.mark.parametrize("variant", ["forward_stay", "backward_recover", "sideways_stay"])
def test_fall_kinematics(variant) -> None:
    clip = make_clip("fall", variant, 21, "f")
    lab = clip.label
    assert lab["is_fall"] and lab["expected_event"] == "FALL_CONFIRMED"
    assert 2.0 <= lab["drop_Ls"] / lab["descend_s"] <= 3.0  # mean hip drop speed in L/s
    assert clip.frames[0].t < lab["t_fall_start"] < clip.frames[-1].t - 5.0
    feats = _features(clip)
    after = [f for f in feats if f.t > lab["t_fall_start"] + 1.5 and f.t < lab["t_fall_start"] + 5.0]
    assert np.median([f.phi for f in after if f.phi is not None]) > 75
    assert np.median([f.h for f in after if f.h is not None]) < 0.4


def test_non_fall_kinematics_match_spec() -> None:
    sit = _features(make_clip("sit_fast", "chair", 4, "s"))
    assert max(f.phi for f in sit if f.phi is not None) < 20
    bend = _features(make_clip("bend", "pickup", 4, "b"))
    assert max(f.phi for f in bend if f.phi is not None) > 55
    assert min(f.h for f in bend if f.h is not None) > 0.7
    slow = _features(make_clip("lie_down_slow", "stay", 4, "l"))
    assert max(f.v for f in slow if f.v is not None) < 1.2


def test_walk_gait_clip_truth() -> None:
    clip = make_walk(1, speed=1.0, cadence=100.0, left_slower=0.15)
    lab = clip.label
    assert lab["cadence_spm"] == pytest.approx(100.0)
    assert lab["T_L"] / lab["T_R"] == pytest.approx(1.15)
    assert lab["si_pct"] == pytest.approx(100 * 0.15 / 1.075)
    assert all(f.persons for f in clip.frames)  # stays in view
