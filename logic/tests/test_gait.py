"""Tests for logic.gait, including the Tier 2 gait acceptance checks."""
import math

import numpy as np
import pytest

from logic.config import Config, GaitCfg
from logic.gait import (
    Calibration, GaitSample, analyze_gait, detect_steps, samples_from_steps, steady_windows, symmetry_index,
)
from logic.pipeline import FramePipeline
from logic.synth import make_clip, make_walk

CFG = Config()


def _samples(clip):
    p = FramePipeline(CFG)
    steps = []
    for f in clip.frames:
        steps += p.push(f)
    return samples_from_steps(steps + p.flush())


def _calib(clip) -> Calibration:
    m = clip.label["marker_px"]
    return Calibration.from_points((m[0], 500.0), (m[1], 500.0), clip.label["marker_m"])


def test_calibration() -> None:
    c = Calibration.from_points((100.0, 200.0), (580.0, 200.0), 3.0)
    assert c.m_per_px == pytest.approx(0.00625) and c.calibrated
    f = Calibration.fallback(200.0, 0.5)
    assert f.m_per_px == pytest.approx(0.0025) and not f.calibrated


def test_symmetry_index_known_values() -> None:
    assert symmetry_index(0.62, 0.54) == pytest.approx(13.79, abs=0.01)  # guide's worked example
    assert symmetry_index(0.6, 0.6) == 0.0


def test_step_detection_on_pure_sinusoid() -> None:
    rng = np.random.default_rng(0)
    t = np.cumsum(np.full(240, 1 / 30) + rng.uniform(-0.003, 0.003, 240))  # 8 s, jittered
    step_T = 0.55
    d = 96.0 * np.abs(np.sin(np.pi * t / step_T))  # one peak per step
    steps = detect_steps(t, d, L_ref=80.0, cfg=GaitCfg())
    true = np.array([step_T * (k + 0.5) for k in range(20) if 0.1 <= step_T * (k + 0.5) <= t[-1] - 0.1])
    assert len(steps) == len(true)
    assert np.abs(steps - true).max() < 0.003  # sub-frame precision
    assert 60 / np.diff(steps).mean() == pytest.approx(60 / step_T, abs=0.5)


def test_steady_windows_split_at_turn_and_skip_standing() -> None:
    walk = make_walk(3, speed=1.0, cadence=100, turn_at=4.5, duration=10.0)
    s = _samples(walk)
    wins = steady_windows(s, CFG.gait)
    assert len(wins) == 2
    t_turn = 100.0 + 4.5
    for a, b in wins:
        assert not (s[a].t - 0.3 < t_turn + 0.6 and s[b - 1].t + 0.3 > t_turn)  # turn excluded
    assert steady_windows(_samples(make_clip("standing", "quiet", 1, "s")), CFG.gait) == []


def test_steady_windows_reject_leaning_torso() -> None:
    s = [GaitSample(t=i / 30, hip_x=100 + 5.0 * i, hip_y=500.0, phi=40.0, L_ref=80.0, ankle_lx=None,
                    ankle_rx=None) for i in range(120)]
    assert steady_windows(s, CFG.gait) == []
    upright = [GaitSample(**{**x.__dict__, "phi": 5.0}) for x in s]
    assert steady_windows(upright, CFG.gait) != []


@pytest.mark.parametrize("seed, speed, cadence, ppm", [
    (1, 1.0, 100, 160), (2, 0.8, 90, 160), (3, 1.3, 115, 400), (4, 0.6, 85, 160), (5, 1.1, 105, 400),
])
def test_tier2_speed_and_cadence(seed, speed, cadence, ppm) -> None:
    clip = make_walk(seed, speed=speed, cadence=cadence, px_per_m=ppm, duration=10.0 if speed < 0.8 else 8.0)
    g = analyze_gait(_samples(clip), CFG.gait, _calib(clip))
    assert g.valid and g.calibrated
    assert g.walk_speed_mps == pytest.approx(speed, rel=0.10)
    assert abs(g.cadence_spm - cadence) <= 5.0


@pytest.mark.parametrize("seed, ppm", [(1, 160), (2, 160), (3, 400), (4, 400), (5, 160), (6, 400)])
def test_tier2_asymmetric_walk_symmetry_index(seed, ppm) -> None:
    clip = make_walk(seed, speed=1.0, cadence=100, left_slower=0.15, px_per_m=ppm)
    g = analyze_gait(_samples(clip), CFG.gait, _calib(clip))
    assert g.n_left >= 4 and g.n_right >= 4
    assert abs(g.symmetry_pct - clip.label["si_pct"]) <= 3.0


def test_symmetric_walk_is_symmetric_and_regular() -> None:
    clip = make_walk(7, speed=1.0, cadence=100, px_per_m=160)
    g = analyze_gait(_samples(clip), CFG.gait, _calib(clip))
    assert g.symmetry_pct < 3.0 and g.step_cv_pct < 4.0


def test_fallback_scale_and_pooling_across_turn() -> None:
    clip = make_walk(8, speed=1.0, cadence=100, turn_at=4.5, duration=10.0)
    g = analyze_gait(_samples(clip), CFG.gait)  # no calibration
    assert not g.calibrated and any("calibrated=false" in r for r in g.reasons)
    assert g.walk_speed_mps == pytest.approx(1.0, rel=0.10)  # synthetic torso is exactly 0.5 m
    assert len(g.windows) == 2 and g.n_steps >= 8


def test_short_walk_reports_missing_metrics() -> None:
    clip = make_walk(9, speed=1.0, cadence=100, duration=3.0)
    g = analyze_gait(_samples(clip), CFG.gait, _calib(clip))
    assert g.valid and g.walk_speed_mps is not None
    assert g.symmetry_pct is None and any("steps per side" in r for r in g.reasons)
    none = analyze_gait(_samples(make_clip("sit_fast", "chair", 2, "s")), CFG.gait)
    assert not none.valid and none.walk_speed_mps is None and none.reasons
