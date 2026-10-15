"""Automatic Timed Up and Go (TUG) segmentation from hip trajectory.

Normalised hip height y_n = (y_max - H_y) / (y_max - y_min) over the session
(0 seated, 1 standing; image y points down, so seated is the largest y). The
extremes are robust percentiles. Phases: SIT, STAND_UP (y_n rises 0.1 -> 0.9),
WALK_OUT, TURN (hip vx reverses; the turn spans the time hip speed is below
``turn_speed_frac`` of the walking speed), WALK_BACK, SIT_DOWN
(y_n falls 0.9 -> 0.1), SIT.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from logic.config import TugCfg
from logic.gait import GaitSample


@dataclass
class TugResult:
    """TUG timings; ``valid`` is False with a ``reason`` if a phase is missing."""

    valid: bool
    reason: str | None = None
    t_sts: float | None = None  # sit-to-stand time, s
    tug_s: float | None = None  # stand-up start to sit-down end, s
    phases: list[tuple[str, float, float]] = field(default_factory=list)


def _cross(t: np.ndarray, y: np.ndarray, i: int, level: float) -> float:
    """Time where y crosses ``level`` between samples i and i + 1."""
    dy = y[i + 1] - y[i]
    return float(t[i] if dy == 0 else t[i] + (level - y[i]) / dy * (t[i + 1] - t[i]))


def _vx(t: np.ndarray, x: np.ndarray, half: float) -> np.ndarray:
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right") - 1
    dt = t[hi] - t[lo]
    v = np.zeros_like(x)
    ok = dt > 0
    v[ok] = (x[hi][ok] - x[lo][ok]) / dt[ok]
    return v


def _find_turn(t: np.ndarray, vx: np.ndarray, thr: float) -> tuple[float, float] | None:
    """First direction reversal (hysteresis ``thr``): (turn start, turn end)."""
    direction = 0
    last_fast = None  # index of the last sample moving fast in the current direction
    for i, v in enumerate(vx):
        if direction == 0:
            if abs(v) > thr:
                direction, last_fast = (1 if v > 0 else -1), i
            continue
        if v * direction > thr:
            last_fast = i
        elif -v * direction > thr:
            return float(t[last_fast]), float(t[i])  # type: ignore[index]
    return None


def segment_tug(samples: list[GaitSample], cfg: TugCfg) -> TugResult:
    """Segment one TUG trial from a session's samples."""
    s = [x for x in samples if x.hip_x is not None and x.hip_y is not None]
    if len(s) < 10:
        return TugResult(False, "too few frames with a visible hip")
    t = np.array([x.t for x in s])
    hx = np.array([x.hip_x for x in s], dtype=float)
    hy = np.array([x.hip_y for x in s], dtype=float)
    Ls = [x.L_ref for x in s if x.L_ref]
    if not Ls:
        return TugResult(False, "no body scale")
    L = float(np.median(Ls))
    y_max, y_min = np.percentile(hy, cfg.y_percentile_hi), np.percentile(hy, cfg.y_percentile_lo)
    if y_max - y_min < cfg.min_range_Ls * L:
        return TugResult(False, "no sit-to-stand transition (hip height barely changes)")
    yn = (y_max - hy) / (y_max - y_min)
    lo, hi = cfg.yn_low, cfg.yn_high

    i_hi = next((i for i in range(len(yn)) if yn[i] >= hi and (yn[:i] <= lo).any()), None)
    if i_hi is None:
        return TugResult(False, "stand-up not found")
    j = int(np.nonzero(yn[:i_hi] <= lo)[0][-1])
    t_a = _cross(t, yn, j, lo)
    t_b = _cross(t, yn, i_hi - 1, hi)

    above = np.nonzero(yn >= hi)[0]
    k_hi = int(above[-1])
    m = next((i for i in range(k_hi + 1, len(yn)) if yn[i] <= lo), None)
    if m is None or k_hi <= i_hi:
        return TugResult(False, "sit-down not found")
    t_c = _cross(t, yn, k_hi, hi)
    t_d = _cross(t, yn, m - 1, lo)

    walk = (t >= t_b) & (t <= t_c)
    vx = _vx(t, hx, cfg.vx_window_s / 2)[walk]
    moving = np.abs(vx)[np.abs(vx) > cfg.turn_speed_Ls * L]
    thr = max(cfg.turn_speed_Ls * L, cfg.turn_speed_frac * float(np.median(moving))) if len(moving) else 0.0
    turn = _find_turn(t[walk], vx, thr) if thr > 0 else None
    if turn is None:
        return TugResult(False, "turn not found (no reversal of walking direction)")
    phases = [("SIT", float(t[0]), t_a), ("STAND_UP", t_a, t_b), ("WALK_OUT", t_b, turn[0]),
              ("TURN", turn[0], turn[1]), ("WALK_BACK", turn[1], t_c), ("SIT_DOWN", t_c, t_d),
              ("SIT", t_d, float(t[-1]))]
    return TugResult(True, None, t_b - t_a, t_d - t_a, phases)
