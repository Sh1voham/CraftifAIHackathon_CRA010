"""Per-frame fall features from cleaned keypoints.

Torso angle ``phi``, normalised hip drop speed ``v``, hip height above the
floor line ``h`` and bbox aspect ratio ``AR``. Every feature is ``None`` when
its inputs are missing; nothing here raises on missing data.

Floor line ``y_floor``: the 90th percentile of the lower ankle's y over the
person's first ``floor_learn_s`` seconds of being upright (phi under
``floor_upright_phi_deg``), then frozen. If ankles are rarely visible it falls
back to ``median standing hip y + floor_fallback_Ls * L_ref``. A config value
``y_floor_override`` wins over both. Note the "floor" learned from ankles is
really the ankle line, a little above the true floor.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

from logic.config import FeatureCfg
from logic.filters import CleanFrame, I_LH, I_LS, I_RH, I_RS
from logic.io import KP_INDEX, BBox

I_LA, I_RA = KP_INDEX["l_ankle"], KP_INDEX["r_ankle"]

Point = tuple[float, float]


@dataclass
class Features:
    """Fall features for one frame (all optional)."""

    t: float
    visible: bool = False
    phi: float | None = None  # torso angle from vertical, degrees
    v: float | None = None  # hip drop speed, torso lengths per second (down = positive)
    h: float | None = None  # hip height above the floor line, torso lengths
    h_prev: float | None = None  # hip height at the start of the drop window
    ar: float | None = None  # bbox width / height
    L: float | None = None  # torso length this frame, px
    L_ref: float | None = None
    hip: Point | None = None
    y_floor: float | None = None


# ------------------------------------------------------------- pure functions

def mid(a: Sequence[float] | None, b: Sequence[float] | None) -> Point | None:
    """Midpoint of two points given as (x, y, ...) (None if either is missing)."""
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def torso_angle(S: Point, H: Point) -> float:
    """Angle of the hip->shoulder vector from vertical: 0 upright, 90 horizontal."""
    return math.degrees(math.atan2(abs(S[0] - H[0]), abs(S[1] - H[1])))


def hip_speed(history: Iterable[tuple[float, float]], t: float, hip_y: float, L_ref: float,
              window_s: float, window_max_s: float) -> tuple[float, float] | None:
    """Hip drop speed in torso lengths per second, and the reference hip y used.

    The reference is the newest sample at least ``window_s`` old; if none is
    within ``window_max_s`` the speed is unknown.
    """
    ref: tuple[float, float] | None = None
    for ts, ys in history:
        age = t - ts
        if window_s <= age <= window_max_s:
            ref = (ts, ys)  # history is oldest-first, so the last match is the newest
    if ref is None or L_ref <= 0.0:
        return None
    return ((hip_y - ref[1]) / (t - ref[0]) / L_ref, ref[1])


def hip_height(y_floor: float, hip_y: float, L_ref: float) -> float:
    """Hip height above the floor line, in torso lengths."""
    return (y_floor - hip_y) / L_ref


def aspect_ratio(bbox: BBox | None) -> float | None:
    """Box width / height, or None."""
    if bbox is None:
        return None
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return w / h if h > 0 else None


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default method)."""
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


# ------------------------------------------------------------- stateful parts

class FloorEstimator:
    """Learns ``y_floor`` from the first seconds of upright posture, then freezes."""

    def __init__(self, cfg: FeatureCfg) -> None:
        self.cfg = cfg
        self.upright_s = 0.0
        self.ankles: list[float] = []
        self.hips: list[float] = []
        self.frozen: float | None = None
        self.method: str | None = None  # "override" | "ankles" | "hips"
        self._t_prev: float | None = None

    def _estimate(self, L_ref: float | None) -> float | None:
        cfg = self.cfg
        if self.ankles and len(self.ankles) >= cfg.floor_min_ankle_fraction * len(self.hips):
            self.method = "ankles"
            return percentile(self.ankles, cfg.floor_percentile)
        if self.hips and L_ref is not None:
            self.method = "hips"
            return percentile(self.hips, 50.0) + cfg.floor_fallback_Ls * L_ref
        return None

    def update(self, t: float, phi: float | None, hip_y: float | None, ankle_y: float | None,
               L_ref: float | None) -> float | None:
        """Feed one frame; return the current floor estimate (or None)."""
        cfg = self.cfg
        if cfg.y_floor_override is not None:
            self.method = "override"
            return cfg.y_floor_override
        if self.frozen is not None:
            return self.frozen
        dt = 0.0 if self._t_prev is None else min(max(t - self._t_prev, 0.0), cfg.upright_dt_cap_s)
        self._t_prev = t
        if phi is not None and phi < cfg.floor_upright_phi_deg and hip_y is not None:
            self.upright_s += dt
            self.hips.append(hip_y)
            if ankle_y is not None:
                self.ankles.append(ankle_y)
        if self.upright_s < cfg.floor_min_learn_s:
            return None
        est = self._estimate(L_ref)
        if est is not None and self.upright_s >= cfg.floor_learn_s:
            self.frozen = est
        return est


class FeatureTracker:
    """Turns a stream of CleanFrames for one person into Features."""

    def __init__(self, cfg: FeatureCfg) -> None:
        self.cfg = cfg
        self.floor = FloorEstimator(cfg)
        self._hist: deque[tuple[float, float]] = deque()

    def update(self, cf: CleanFrame) -> Features:
        """Compute features for one cleaned frame."""
        cfg = self.cfg
        f = Features(t=cf.t, visible=cf.visible, L_ref=cf.L_ref, ar=aspect_ratio(cf.bbox))
        kp = cf.kp
        S, H = mid(kp[I_LS], kp[I_RS]), mid(kp[I_LH], kp[I_RH])
        if S is not None and H is not None:
            f.phi = torso_angle(S, H)
            f.L = math.hypot(S[0] - H[0], S[1] - H[1])
        f.hip = H
        ankles = [k[1] for k in (kp[I_LA], kp[I_RA]) if k is not None]
        f.y_floor = self.floor.update(cf.t, f.phi, H[1] if H else None, max(ankles) if ankles else None,
                                      cf.L_ref)
        while self._hist and cf.t - self._hist[0][0] > cfg.drop_window_max_s:
            self._hist.popleft()
        if H is not None and cf.L_ref is not None:
            sp = hip_speed(self._hist, cf.t, H[1], cf.L_ref, cfg.drop_window_s, cfg.drop_window_max_s)
            if sp is not None:
                f.v = sp[0]
                if f.y_floor is not None:
                    f.h_prev = hip_height(f.y_floor, sp[1], cf.L_ref)
            if f.y_floor is not None:
                f.h = hip_height(f.y_floor, H[1], cf.L_ref)
        if H is not None:
            self._hist.append((cf.t, H[1]))
        return f
