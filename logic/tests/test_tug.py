"""Tests for logic.tug, including the Tier 2 TUG acceptance check."""
import numpy as np
import pytest

from logic.config import Config
from logic.gait import GaitSample, samples_from_steps
from logic.pipeline import FramePipeline
from logic.synth import make_tug, make_walk
from logic.tug import segment_tug

CFG = Config()
PHASES = ["SIT", "STAND_UP", "WALK_OUT", "TURN", "WALK_BACK", "SIT_DOWN", "SIT"]


def _samples(clip):
    p = FramePipeline(CFG)
    steps = []
    for f in clip.frames:
        steps += p.push(f)
    return samples_from_steps(steps + p.flush())


@pytest.fixture(scope="module")
def tug_results():
    out = []
    for seed in range(20):
        clip = make_tug(seed, px_per_m=160.0 if seed % 2 else 400.0)
        out.append((clip.label, segment_tug(_samples(clip), CFG.tug)))
    return out


def test_tier2_tug_phases_found_on_90_percent(tug_results) -> None:
    valid = [r for _, r in tug_results if r.valid]
    assert len(valid) >= 0.9 * len(tug_results)
    for r in valid:
        assert [p[0] for p in r.phases] == PHASES
        for (_, a0, a1), (_, b0, _) in zip(r.phases, r.phases[1:]):
            assert a0 <= a1 == b0  # contiguous, ordered


def test_tug_timings_match_truth(tug_results) -> None:
    for lab, r in tug_results:
        if not r.valid:
            continue
        assert r.tug_s == pytest.approx(lab["tug_s"], abs=0.3)
        assert r.t_sts == pytest.approx(lab["t_sts"], abs=0.2)
        turn = dict((p[0], (p[1], p[2])) for p in r.phases[:5])["TURN"]
        assert turn[0] <= lab["t_turn_mid"] <= turn[1]


def _synthetic(ys, xs):
    return [GaitSample(t=i / 30, hip_x=x, hip_y=y, phi=5.0, L_ref=100.0, ankle_lx=None, ankle_rx=None)
            for i, (x, y) in enumerate(zip(xs, ys))]


def test_invalid_without_sit_stand_or_turn() -> None:
    walk = segment_tug(_samples(make_walk(1, duration=6.0)), CFG.tug)
    assert not walk.valid and "sit-to-stand" in walk.reason
    n = 300
    y = np.concatenate([np.full(100, 600.0), np.linspace(600, 450, 30), np.full(n - 160, 450.0),
                        np.linspace(450, 600, 30)])
    no_walk = segment_tug(_synthetic(list(y), [300.0] * n), CFG.tug)
    assert not no_walk.valid and "turn" in no_walk.reason
    stand_only = segment_tug(_synthetic(list(np.linspace(600, 450, 60)), [300.0] * 60), CFG.tug)
    assert not stand_only.valid and "sit-down" in stand_only.reason
    assert not segment_tug([], CFG.tug).valid
