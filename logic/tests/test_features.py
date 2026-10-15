"""Tests for logic.features."""
import random

import pytest

from logic.config import FeatureCfg
from logic.features import (
    FeatureTracker, FloorEstimator, aspect_ratio, hip_height, hip_speed, mid, percentile, torso_angle,
)
from logic.filters import N_KP, CleanFrame
from logic.io import KP_INDEX


@pytest.mark.parametrize("S, expected", [
    ((100.0, 300.0), 0.0),     # shoulder straight above the hip
    ((300.0, 300.0), 45.0),    # leaning right
    ((-100.0, 300.0), 45.0),   # leaning left: same angle
    ((300.0, 500.0), 90.0),    # horizontal
    ((100.0, 700.0), 0.0),     # upside down still reads as vertical torso
])
def test_torso_angle_known_vectors(S, expected) -> None:
    assert torso_angle(S, (100.0, 500.0)) == pytest.approx(expected, abs=1e-9)


def test_mid_and_helpers() -> None:
    assert mid((0.0, 0.0, 0.9), (2.0, 4.0, 0.5)) == (1.0, 2.0)
    assert mid(None, (1.0, 1.0)) is None
    assert hip_height(1000.0, 600.0, 200.0) == 2.0
    assert aspect_ratio((0.0, 0.0, 300.0, 100.0)) == 3.0
    assert aspect_ratio((0.0, 0.0, 10.0, 0.0)) is None and aspect_ratio(None) is None
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 90) == pytest.approx(4.6)


def test_hip_speed_constant_velocity_drop_with_irregular_timestamps() -> None:
    rng = random.Random(1)
    t, hist = 0.0, []
    for _ in range(40):
        t += 1 / 30 + rng.uniform(-0.003, 0.003)
        hist.append((t, 600.0 + 400.0 * t))  # 400 px/s downward
    t_now = t + 0.03
    v, ref_y = hip_speed(hist, t_now, 600.0 + 400.0 * t_now, L_ref=200.0, window_s=0.25, window_max_s=0.75)
    assert v == pytest.approx(2.0)  # 400 / 200 torso lengths per second, positive = down
    ages = [t_now - ts for ts, _ in hist]
    newest_old_enough = min(a for a in ages if a >= 0.25)
    assert ref_y == pytest.approx(600.0 + 400.0 * (t_now - newest_old_enough))


def test_hip_speed_upward_is_negative_and_none_without_reference() -> None:
    hist = [(0.0, 600.0), (0.1, 590.0), (0.2, 580.0), (0.3, 570.0)]
    v, _ = hip_speed(hist, 0.3 + 1e-9, 570.0, 100.0, 0.25, 0.75)
    assert v == pytest.approx(-1.0)
    assert hip_speed(hist[-2:], 0.35, 570.0, 100.0, 0.25, 0.75) is None  # nothing old enough
    assert hip_speed([(0.0, 600.0)], 2.0, 570.0, 100.0, 0.25, 0.75) is None  # too old


def _upright_frame(t: float, ankle_y: float | None, hip_y: float = 600.0, phi_lean: float = 0.0,
                   L_ref: float = 200.0, bbox=None) -> CleanFrame:
    kp = [None] * N_KP
    def put(n, x, y):
        kp[KP_INDEX[n]] = (x, y, 0.9, False)
    put("l_shoulder", 290.0 + phi_lean, hip_y - 200.0); put("r_shoulder", 310.0 + phi_lean, hip_y - 200.0)
    put("l_hip", 290.0, hip_y); put("r_hip", 310.0, hip_y)
    if ankle_y is not None:
        put("l_ankle", 290.0, ankle_y); put("r_ankle", 330.0, ankle_y - 20.0)
    return CleanFrame(t=t, kp=kp, L_ref=L_ref, bbox=bbox, visible=True)


def test_floor_learned_from_ankles_then_frozen() -> None:
    tr = FeatureTracker(FeatureCfg())
    rng = random.Random(0)
    f = None
    for i in range(200):  # 6.7 s upright at 30 fps
        f = tr.update(_upright_frame(i / 30, 960.0 + rng.uniform(-2, 2)))
        if i == 5:
            assert f.y_floor is None  # less than floor_min_learn_s of upright time
    assert tr.floor.method == "ankles" and tr.floor.frozen is not None
    assert f.y_floor == pytest.approx(960.0, abs=2.5)
    assert f.h == pytest.approx((f.y_floor - 600.0) / 200.0)
    frozen = tr.floor.frozen
    tr.update(_upright_frame(7.0, 1500.0))  # later samples do not move a frozen floor
    assert tr.floor.frozen == frozen


def test_floor_falls_back_to_hips_without_ankles() -> None:
    tr = FeatureTracker(FeatureCfg())
    f = None
    for i in range(200):
        f = tr.update(_upright_frame(i / 30, None, hip_y=600.0))
    assert tr.floor.method == "hips"
    assert f.y_floor == pytest.approx(600.0 + 1.2 * 200.0)


def test_floor_override_and_no_learning_while_not_upright() -> None:
    tr = FeatureTracker(FeatureCfg(y_floor_override=900.0))
    assert tr.update(_upright_frame(0.0, 960.0)).y_floor == 900.0
    tr2 = FeatureTracker(FeatureCfg())
    for i in range(100):
        f = tr2.update(_upright_frame(i / 30, 960.0, phi_lean=200.0))  # torso at 45 degrees
    assert f.y_floor is None and tr2.floor.upright_s == 0.0


def test_tracker_never_raises_on_missing_data() -> None:
    tr = FeatureTracker(FeatureCfg())
    f = tr.update(CleanFrame(t=0.0, kp=[None] * N_KP, L_ref=None, bbox=None, visible=False))
    assert (f.phi, f.v, f.h, f.L, f.hip, f.ar) == (None,) * 6
    half = _upright_frame(0.1, 960.0, bbox=(0.0, 0.0, 100.0, 200.0))
    half.kp[KP_INDEX["r_hip"]] = None
    f = tr.update(half)
    assert f.phi is None and f.hip is None and f.ar == 0.5


def test_tracker_hip_speed_and_h_prev_during_drop() -> None:
    tr = FeatureTracker(FeatureCfg(y_floor_override=1000.0))
    out = []
    for i in range(30):
        t = i / 30
        hip_y = 600.0 + (400.0 * (t - 0.5) if t > 0.5 else 0.0)
        out.append(tr.update(_upright_frame(t, 960.0, hip_y=hip_y)))
    last = out[-1]
    assert last.v == pytest.approx(2.0, abs=1e-6)
    assert last.h_prev > last.h
