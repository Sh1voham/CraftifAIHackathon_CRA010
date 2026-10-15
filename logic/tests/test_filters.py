"""Tests for logic.filters."""
import pytest

from logic.config import FilterCfg, OneEuroCfg
from logic.filters import KeypointCleaner, OneEuro, RefScale
from logic.io import Person

PASS = OneEuroCfg(min_cutoff=1e9, beta=0.0, d_cutoff=1.0, reset_gap_s=1.0)  # ~no smoothing


def _run(cleaner: KeypointCleaner, items):
    out = []
    for t, p in items:
        out += cleaner.push(t, p)
    return out + cleaner.flush()


def _body(t: float, x0: float = 300.0, dx: float = 0.0, conf: float = 0.9, drop: tuple = ()) -> Person:
    """Upright torso of length 200 px at hip (x0 + dx*t, 600); ``drop`` names keypoints to hide."""
    hx = x0 + dx * t
    kp = {
        "l_shoulder": (hx - 10, 400.0, conf), "r_shoulder": (hx + 10, 400.0, conf),
        "l_hip": (hx - 10, 600.0, conf), "r_hip": (hx + 10, 600.0, conf),
        "l_ankle": (hx - 10, 960.0, conf),
    }
    for n in drop:
        kp.pop(n)
    return Person(kp=kp)


# ------------------------------------------------------------- One Euro

def _step(cfg: OneEuroCfg, dt: float, until: float) -> float:
    f = OneEuro(cfg)
    f(0.0, 0.0)
    t, y = 0.0, 0.0
    while t < until - 1e-9:
        t += dt
        y = f(t, 100.0)
    return y


def test_one_euro_step_response_converges() -> None:
    cfg = OneEuroCfg(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    y_early, y_late = _step(cfg, 1 / 30, 0.1), _step(cfg, 1 / 30, 1.0)
    assert 0 < y_early < 60  # lags a step
    assert y_late > 98  # time constant 1/(2*pi) s


def test_one_euro_uses_real_dt_not_sample_count() -> None:
    cfg = OneEuroCfg(min_cutoff=1.0, beta=0.0, d_cutoff=1.0)
    y30, y60 = _step(cfg, 1 / 30, 0.5), _step(cfg, 1 / 60, 0.5)
    assert abs(y30 - y60) < 2.0  # same elapsed time -> nearly same response
    # Irregular sampling still tracks the time-based response.
    f = OneEuro(cfg)
    f(0.0, 0.0)
    ts = [0.02, 0.07, 0.09, 0.16, 0.2, 0.31, 0.33, 0.41, 0.5]
    y = 0.0
    for t in ts:
        y = f(t, 100.0)
    assert abs(y - y30) < 3.0


def test_one_euro_beta_reduces_lag_on_fast_motion() -> None:
    slow = OneEuro(OneEuroCfg(min_cutoff=1.0, beta=0.0))
    fast = OneEuro(OneEuroCfg(min_cutoff=1.0, beta=0.05))
    for i in range(31):
        t = i / 30
        ys, yf = slow(t, 500 * t), fast(t, 500 * t)
    assert abs(500 - yf) < abs(500 - ys)


def test_one_euro_guards_bad_dt_and_resets_after_gap() -> None:
    f = OneEuro(OneEuroCfg(min_cutoff=1.0, beta=0.0, reset_gap_s=1.0))
    f(0.0, 0.0)
    y1 = f(0.1, 100.0)
    assert f(0.1, 500.0) == y1  # dt == 0
    assert f(0.05, 500.0) == y1  # dt < 0
    assert f(1.2, 42.0) == 42.0  # gap > 1 s resets to the raw value


# ------------------------------------------------------------- RefScale

def test_refscale_median_window_and_hold() -> None:
    r = RefScale(window_s=2.0)
    assert r.update(0.0, None) is None
    for t, L in [(0.0, 200.0), (0.5, 210.0), (1.0, 190.0), (1.5, None), (1.9, 1000.0)]:
        r.update(t, L)
    assert r.value == pytest.approx(205.0)  # median of 190, 200, 210, 1000
    assert r.update(2.6, None) == pytest.approx(595.0)  # 0.0 and 0.5 left the window -> 190, 1000
    assert r.update(10.0, None) == pytest.approx(595.0)  # window empty -> hold


# ------------------------------------------------------------- cleaner

def test_cleaner_delays_by_max_gap_and_preserves_order() -> None:
    c = KeypointCleaner(FilterCfg(one_euro=PASS))
    emitted = [len(c.push(i / 30, _body(i / 30))) for i in range(10)]
    assert emitted == [0] * 5 + [1] * 5
    rest = c.flush()
    assert len(rest) == 5 and [f.t for f in rest] == [i / 30 for i in range(5, 10)]


def test_confidence_gating() -> None:
    c = KeypointCleaner(FilterCfg(max_gap_frames=0, one_euro=PASS))
    f = _run(c, [(0.0, _body(0.0, conf=0.2))])[0]
    assert all(k is None for k in f.kp)
    assert f.visible


def test_short_gap_is_interpolated_in_time() -> None:
    c = KeypointCleaner(FilterCfg(one_euro=PASS))
    ts = [0.0, 0.03, 0.07, 0.10, 0.12, 0.2, 0.23]  # irregular timestamps
    items = [(t, _body(t, dx=100.0, drop=("l_hip",) if 1 <= i <= 3 else ())) for i, t in enumerate(ts)]
    out = _run(c, items)
    for i, f in enumerate(out):
        k = f.get("l_hip")
        assert k is not None
        assert k[0] == pytest.approx(300 - 10 + 100 * ts[i])
        assert k[3] == (1 <= i <= 3)


def test_long_gap_stays_missing() -> None:
    c = KeypointCleaner(FilterCfg(max_gap_frames=5, one_euro=PASS))
    items = [(i / 30, _body(i / 30, drop=("l_hip",) if 2 <= i <= 7 else ())) for i in range(12)]
    out = _run(c, items)
    assert [f.get("l_hip") is None for f in out] == [2 <= i <= 7 for i in range(12)]


def test_gap_at_stream_end_and_missing_person() -> None:
    c = KeypointCleaner(FilterCfg(one_euro=PASS))
    items = [(0.0, _body(0.0)), (0.033, None), (0.066, _body(0.066)), (0.1, None)]
    out = _run(c, items)
    assert out[1].get("l_hip") is not None and out[1].get("l_hip")[3] and not out[1].visible
    assert out[3].get("l_hip") is None  # no later sample to interpolate towards


def test_outlier_is_rejected_and_repaired() -> None:
    c = KeypointCleaner(FilterCfg(one_euro=PASS, outlier_jump_Ls=0.5))
    items = []
    for i in range(20):
        p = _body(i / 30)
        if i == 10:
            x, y, cf = p.kp["l_ankle"]
            p.kp["l_ankle"] = (x + 150.0, y, cf)  # 0.75 L jump for one frame
        if i == 12:
            x, y, cf = p.kp["r_hip"]
            p.kp["r_hip"] = (x + 60.0, y, cf)  # 0.3 L: plausible, kept
        items.append((i / 30, p))
    out = _run(c, items)
    k10 = out[10].get("l_ankle")
    assert k10[0] == pytest.approx(290.0) and k10[3]  # rejected, then interpolated
    assert out[12].get("r_hip")[0] == pytest.approx(370.0) and not out[12].get("r_hip")[3]


def test_outlier_reanchors_after_persistent_move() -> None:
    c = KeypointCleaner(FilterCfg(one_euro=PASS, max_gap_frames=3))
    items = []
    for i in range(16):
        p = _body(i / 30)
        if i >= 5:
            x, y, cf = p.kp["l_ankle"]
            p.kp["l_ankle"] = (x + 200.0, y, cf)  # keypoint genuinely moved elsewhere
        items.append((i / 30, p))
    out = _run(c, items)
    xs = [f.get("l_ankle")[0] for f in out]
    flags = [f.get("l_ankle")[3] for f in out]
    assert xs[:5] == pytest.approx([290.0] * 5)
    assert xs[8:] == pytest.approx([490.0] * 8)  # accepted after max_gap_frames rejections
    assert flags[5:8] == [True] * 3 and xs[4] < xs[5] < xs[6] < xs[7] < xs[8]  # bridged smoothly


def test_smoothing_is_applied_after_repair() -> None:
    c = KeypointCleaner(FilterCfg(one_euro=OneEuroCfg(min_cutoff=0.5, beta=0.0)))
    out = _run(c, [(i / 30, _body(i / 30, x0=300.0 + (50.0 if i >= 10 else 0.0))) for i in range(15)])
    x10 = out[10].get("l_hip")[0]
    assert 290.0 < x10 < 340.0  # lagging the 50 px step
