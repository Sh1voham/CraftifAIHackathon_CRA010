"""Keypoint cleaning: confidence gating, outlier rejection, gap repair, smoothing.

All timing comes from the frame timestamp ``t``; nothing assumes a frame rate.
The hot path uses plain floats and tuples (no numpy) to stay fast per frame.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from logic.config import FilterCfg, OneEuroCfg
from logic.io import KEYPOINT_NAMES, KP_INDEX, BBox, Person

N_KP = len(KEYPOINT_NAMES)
I_LS, I_RS, I_LH, I_RH = (KP_INDEX[n] for n in ("l_shoulder", "r_shoulder", "l_hip", "r_hip"))

# (x, y, confidence, interpolated) after cleaning; None when missing.
CleanKP = tuple[float, float, float, bool]


@dataclass
class CleanFrame:
    """One cleaned, smoothed frame for the followed person."""

    t: float
    kp: list[CleanKP | None]  # COCO-17 order
    L_ref: float | None
    bbox: BBox | None
    visible: bool  # a person was selected in this frame

    def get(self, name: str) -> CleanKP | None:
        """Keypoint by canonical name."""
        return self.kp[KP_INDEX[name]]


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """One Euro filter for a single scalar, using the real (variable) dt."""

    __slots__ = ("cfg", "x_prev", "dx_prev", "t_prev")

    def __init__(self, cfg: OneEuroCfg) -> None:
        self.cfg = cfg
        self.x_prev: float | None = None
        self.dx_prev = 0.0
        self.t_prev: float | None = None

    def reset(self) -> None:
        """Forget all history."""
        self.x_prev, self.dx_prev, self.t_prev = None, 0.0, None

    def __call__(self, t: float, x: float) -> float:
        if self.t_prev is None or self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, 0.0, t
            return x
        dt = t - self.t_prev
        if dt <= 0.0:  # duplicate or out-of-order timestamp: keep the last estimate
            return self.x_prev
        if dt > self.cfg.reset_gap_s:
            self.x_prev, self.dx_prev, self.t_prev = x, 0.0, t
            return x
        a_d = _alpha(self.cfg.d_cutoff, dt)
        dx_hat = a_d * (x - self.x_prev) / dt + (1.0 - a_d) * self.dx_prev
        a = _alpha(self.cfg.min_cutoff + self.cfg.beta * abs(dx_hat), dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class RefScale:
    """L_ref: median torso length over the last ``window_s`` seconds.

    Frames without a visible torso are ignored; if none remain in the window
    the last value is held.
    """

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self.samples: deque[tuple[float, float]] = deque()
        self.value: float | None = None

    def update(self, t: float, L: float | None) -> float | None:
        """Add a torso length (or None) observed at ``t``; return L_ref."""
        if L is not None and L > 0.0:
            self.samples.append((t, L))
        while self.samples and self.samples[0][0] < t - self.window_s:
            self.samples.popleft()
        if self.samples:
            vals = sorted(v for _, v in self.samples)
            n = len(vals)
            self.value = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        return self.value


def torso_length(pts: list[tuple[float, float, float] | None]) -> float | None:
    """||mid_shoulder - mid_hip|| if all four torso points are present."""
    ls, rs, lh, rh = pts[I_LS], pts[I_RS], pts[I_LH], pts[I_RH]
    if ls is None or rs is None or lh is None or rh is None:
        return None
    return math.hypot((ls[0] + rs[0] - lh[0] - rh[0]) / 2.0, (ls[1] + rs[1] - lh[1] - rh[1]) / 2.0)


@dataclass
class _Pending:
    idx: int
    t: float
    pts: list[tuple[float, float, float] | None]
    L_ref: float | None
    bbox: BBox | None
    visible: bool


class KeypointCleaner:
    """Streaming cleaner for one followed person.

    ``push`` gates by confidence, rejects outliers, then holds frames for up to
    ``max_gap_frames`` so short gaps can be interpolated using the next valid
    sample, and finally applies a One Euro filter per coordinate. Frames come
    out in order, delayed by at most ``max_gap_frames`` frames; call ``flush``
    at the end of a stream.
    """

    def __init__(self, cfg: FilterCfg) -> None:
        self.cfg = cfg
        self.ref = RefScale(cfg.Lref_window_s)
        self._idx = 0
        self._buf: deque[_Pending] = deque()
        # outlier state: last accepted raw sample per keypoint and its frame index
        self._acc: list[tuple[float, float] | None] = [None] * N_KP
        self._acc_idx: list[int] = [-1] * N_KP
        self._rejects: list[int] = [0] * N_KP
        # gap-repair state: last real (non-interpolated) emitted sample per keypoint
        self._prev: list[tuple[int, float, float, float, float] | None] = [None] * N_KP  # idx, t, x, y, c
        self._euro = [(OneEuro(cfg.one_euro), OneEuro(cfg.one_euro)) for _ in range(N_KP)]

    def push(self, t: float, person: Person | None) -> list[CleanFrame]:
        """Add the followed person's raw keypoints at ``t`` (None if not seen)."""
        cfg = self.cfg
        pts: list[tuple[float, float, float] | None] = [None] * N_KP
        if person is not None:
            for name, k in person.kp.items():
                if k[2] >= cfg.c_min:
                    pts[KP_INDEX[name]] = k
        L_ref = self.ref.update(t, torso_length(pts))
        if L_ref is not None:
            limit = cfg.outlier_jump_Ls * L_ref
            for i in range(N_KP):
                k = pts[i]
                if k is None:
                    continue
                acc = self._acc[i]
                recent = acc is not None and self._idx - self._acc_idx[i] <= cfg.max_gap_frames + 1
                if recent and math.hypot(k[0] - acc[0], k[1] - acc[1]) > limit:  # type: ignore[index]
                    self._rejects[i] += 1
                    if self._rejects[i] <= cfg.max_gap_frames:
                        pts[i] = None
                        continue
                self._acc[i], self._acc_idx[i], self._rejects[i] = (k[0], k[1]), self._idx, 0
        else:
            for i in range(N_KP):
                k = pts[i]
                if k is not None:
                    self._acc[i], self._acc_idx[i] = (k[0], k[1]), self._idx
        bbox = person.bbox if person is not None else None
        self._buf.append(_Pending(self._idx, t, pts, L_ref, bbox, person is not None))
        self._idx += 1
        out: list[CleanFrame] = []
        while len(self._buf) > cfg.max_gap_frames:
            out.append(self._emit(self._buf.popleft()))
        return out

    def flush(self) -> list[CleanFrame]:
        """Emit every buffered frame (end of stream)."""
        out = []
        while self._buf:
            out.append(self._emit(self._buf.popleft()))
        return out

    def _repair(self, i: int, e: _Pending) -> tuple[float, float, float] | None:
        prev = self._prev[i]
        if prev is None:
            return None
        p_idx, p_t, p_x, p_y, p_c = prev
        for nxt in self._buf:
            k = nxt.pts[i]
            if k is None:
                continue
            if nxt.idx - p_idx - 1 > self.cfg.max_gap_frames:
                return None
            span = nxt.t - p_t
            if span <= 0.0 or span > self.cfg.one_euro.reset_gap_s:
                return None
            w = (e.t - p_t) / span
            return (p_x + (k[0] - p_x) * w, p_y + (k[1] - p_y) * w, min(p_c, k[2]))
        return None

    def _emit(self, e: _Pending) -> CleanFrame:
        kp: list[CleanKP | None] = [None] * N_KP
        for i in range(N_KP):
            k = e.pts[i]
            interp = False
            if k is None:
                k = self._repair(i, e)
                if k is None:
                    continue
                interp = True
            else:
                self._prev[i] = (e.idx, e.t, k[0], k[1], k[2])
            fx, fy = self._euro[i]
            kp[i] = (fx(e.t, k[0]), fy(e.t, k[1]), k[2], interp)
        return CleanFrame(t=e.t, kp=kp, L_ref=e.L_ref, bbox=e.bbox, visible=e.visible)
