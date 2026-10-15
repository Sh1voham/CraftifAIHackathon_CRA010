"""Gait metrics from side-on walking: speed, cadence, step-time CV, symmetry.

Works on a window of cleaned frames for one person. Only steady, straight
walking segments are used. A laptop camera sees about 2 m, so step intervals
are pooled across all steady windows (never across a turn). Step peaks get
parabolic sub-sample refinement because frame quantisation alone would add
several points of symmetry-index error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.signal import find_peaks

from logic.config import GaitCfg
from logic.features import Features
from logic.filters import CleanFrame
from logic.io import KP_INDEX

I_LA, I_RA = KP_INDEX["l_ankle"], KP_INDEX["r_ankle"]


@dataclass(frozen=True)
class GaitSample:
    """Per-frame inputs to gait analysis (pixels)."""

    t: float
    hip_x: float | None
    hip_y: float | None
    phi: float | None
    L_ref: float | None
    ankle_lx: float | None
    ankle_rx: float | None


def sample_from(clean: CleanFrame, feats: Features) -> GaitSample:
    """Build a GaitSample from a cleaned frame and its features.

    Ankles come from the repaired but unsmoothed keypoints: the One Euro
    filter's speed-dependent lag shifts left and right step peaks by different
    amounts. ``detect_steps`` refines peaks with a zero-phase least-squares fit.
    """
    src = clean.raw if clean.raw is not None else clean.kp
    la, ra = src[I_LA], src[I_RA]
    hip = feats.hip
    return GaitSample(t=clean.t, hip_x=hip[0] if hip else None, hip_y=hip[1] if hip else None, phi=feats.phi,
                      L_ref=clean.L_ref, ankle_lx=la[0] if la else None, ankle_rx=ra[0] if ra else None)


@dataclass(frozen=True)
class Calibration:
    """Metres per pixel in the walking plane."""

    m_per_px: float
    calibrated: bool

    @classmethod
    def from_points(cls, p1: tuple[float, float], p2: tuple[float, float], d_real_m: float) -> "Calibration":
        """Two floor markers ``d_real_m`` apart, given in pixels."""
        return cls(d_real_m / math.hypot(p2[0] - p1[0], p2[1] - p1[1]), True)

    @classmethod
    def fallback(cls, L_ref_px: float, torso_m: float) -> "Calibration":
        """Rough scale assuming the torso is ``torso_m`` long (marked uncalibrated)."""
        return cls(torso_m / L_ref_px, False)


@dataclass
class GaitSummary:
    """Gait metrics; any metric may be None, with the reason in ``reasons``."""

    valid: bool
    reasons: list[str] = field(default_factory=list)
    calibrated: bool = False
    walk_speed_mps: float | None = None
    cadence_spm: float | None = None
    step_cv_pct: float | None = None
    symmetry_pct: float | None = None
    n_steps: int = 0
    n_left: int = 0
    n_right: int = 0
    walk_time_s: float = 0.0
    windows: list[tuple[float, float]] = field(default_factory=list)


# ---------------------------------------------------------------- helpers

def _central_vx(t: np.ndarray, x: np.ndarray, half: float) -> np.ndarray:
    """Hip horizontal speed by central difference over +/- ``half`` seconds."""
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right") - 1
    dt = t[hi] - t[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (x[hi] - x[lo]) / dt
    v[dt <= 0] = np.nan
    return v


def _rolling_cv(t: np.ndarray, s: np.ndarray, half: float) -> np.ndarray:
    out = np.full(len(t), np.inf)
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    for i in range(len(t)):
        w = s[lo[i]:hi[i]]
        w = w[np.isfinite(w)]
        if len(w) >= 3 and w.mean() > 0:
            out[i] = w.std() / w.mean()
    return out


def steady_windows(samples: list[GaitSample], cfg: GaitCfg) -> list[tuple[int, int]]:
    """Index ranges [i0, i1) of steady, straight walking (turns discarded).

    Steady: hip speed above ``steady_min_speed_Ls * L_ref``, low rolling
    coefficient of variation, torso angle under ``steady_max_phi_deg``, one
    direction of travel, no data hole longer than ``steady_max_gap_s``.
    """
    idx = [i for i, s in enumerate(samples) if s.hip_x is not None and s.L_ref and s.phi is not None]
    if len(idx) < 3:
        return []
    t = np.array([samples[i].t for i in idx])
    x = np.array([samples[i].hip_x for i in idx], dtype=float)
    L = np.array([samples[i].L_ref for i in idx], dtype=float)
    phi = np.array([samples[i].phi for i in idx], dtype=float)
    vx = _central_vx(t, x, cfg.vx_window_s / 2)
    speed = np.abs(vx)
    cv = _rolling_cv(t, speed, cfg.steady_roll_s / 2)
    ok = (speed > cfg.steady_min_speed_Ls * L) & (cv <= cfg.steady_speed_cv_max) & (phi < cfg.steady_max_phi_deg)
    sign = np.sign(np.nan_to_num(vx))

    runs: list[tuple[int, int]] = []
    start = None
    for k in range(len(idx) + 1):
        cont = (k < len(idx) and ok[k] and start is not None and sign[k] == sign[start]
                and t[k] - t[k - 1] <= cfg.steady_max_gap_s)
        if start is not None and not cont:
            runs.append((start, k))
            start = None
        if k < len(idx) and ok[k] and start is None:
            start = k
    out = []
    for a, b in runs:
        ta, tb = t[a] + cfg.turn_trim_s, t[b - 1] - cfg.turn_trim_s
        if tb - ta < cfg.steady_min_len_s:
            continue
        a2 = a + int(np.searchsorted(t[a:b], ta, side="left"))
        b2 = a + int(np.searchsorted(t[a:b], tb, side="right"))
        out.append((idx[a2], idx[b2 - 1] + 1))
    return out


def walking_speed(samples: list[GaitSample], window: tuple[int, int], calib: Calibration) -> float:
    """Metres per second of hip travel over a window."""
    s1 = next(s for s in samples[window[0]:window[1]] if s.hip_x is not None)
    s2 = next(s for s in reversed(samples[window[0]:window[1]]) if s.hip_x is not None)
    return calib.m_per_px * abs(s2.hip_x - s1.hip_x) / (s2.t - s1.t)  # type: ignore[operator]


def _refine_peak(t: np.ndarray, d: np.ndarray, i: int, half: float) -> float:
    """Vertex of a least-squares parabola fitted to d within +/- ``half`` s of sample i."""
    m = np.abs(t - t[i]) <= half
    if m.sum() < 4:
        return float(t[i])
    c2, c1, _ = np.polyfit(t[m] - t[i], d[m], 2)
    if c2 >= 0:
        return float(t[i])
    off = -c1 / (2 * c2)
    return float(t[i] + off) if abs(off) <= half else float(t[i])


def detect_steps(t: np.ndarray, d: np.ndarray, L_ref: float, cfg: GaitCfg) -> np.ndarray:
    """Step times: peaks of ankle separation ``d(t)``, refined to sub-frame precision."""
    if len(t) < 3:
        return np.array([])
    fps_est = float(np.median(1.0 / np.diff(t)))
    peaks, _ = find_peaks(d, distance=max(1, int(cfg.peak_min_distance_s * fps_est)),
                          prominence=cfg.peak_prominence_Ls * L_ref)
    half = cfg.peak_fit_half_s
    inner = [int(i) for i in peaks if t[i] - t[0] >= half and t[-1] - t[i] >= half]  # cut-off peaks lie
    return np.array([_refine_peak(t, d, i, half) for i in inner])


def symmetry_index(T_L: float, T_R: float) -> float:
    """SI = 100 |T_L - T_R| / (0.5 (T_L + T_R))."""
    return 100.0 * abs(T_L - T_R) / (0.5 * (T_L + T_R))


def window_steps(samples: list[GaitSample], window: tuple[int, int], cfg: GaitCfg
                 ) -> tuple[list[float], list[str]]:
    """Step intervals in one window and the leading side ('L'/'R') of the step each ends with."""
    ws = [s for s in samples[window[0]:window[1]] if s.ankle_lx is not None and s.ankle_rx is not None]
    if len(ws) < 3:
        return [], []
    t = np.array([s.t for s in ws])
    diff = np.array([s.ankle_lx - s.ankle_rx for s in ws])  # type: ignore[operator]
    L_ref = float(np.median([s.L_ref for s in ws if s.L_ref]))
    hx = [s.hip_x for s in ws if s.hip_x is not None]
    direction = math.copysign(1.0, hx[-1] - hx[0])
    times = detect_steps(t, np.abs(diff), L_ref, cfg)
    sides = []
    for tp in times:
        sep = float(np.interp(tp, t, diff))
        sides.append("L" if sep * direction > 0 else "R")
    intervals = list(np.diff(times))
    return intervals, sides[1:]


def analyze_gait(samples: list[GaitSample], cfg: GaitCfg, calib: Calibration | None = None) -> GaitSummary:
    """Gait summary over ``samples`` (one person, time-ordered)."""
    windows = steady_windows(samples, cfg)
    if not windows:
        return GaitSummary(valid=False, reasons=["no steady straight walking of at least "
                                                 f"{cfg.steady_min_len_s:g} s"])
    if calib is None:
        L = np.median([s.L_ref for i0, i1 in windows for s in samples[i0:i1] if s.L_ref])
        calib = Calibration.fallback(float(L), cfg.fallback_torso_m)
    out = GaitSummary(valid=True, calibrated=calib.calibrated)
    dist = dur = 0.0
    intervals: list[float] = []
    sides: list[str] = []
    for w in windows:
        t0, t1 = samples[w[0]].t, samples[w[1] - 1].t
        dist += walking_speed(samples, w, calib) * (t1 - t0)
        dur += t1 - t0
        out.windows.append((t0, t1))
        iv, sd = window_steps(samples, w, cfg)
        intervals += iv
        sides += sd
    out.walk_speed_mps = dist / dur
    out.walk_time_s = dur
    if not calib.calibrated:
        out.reasons.append("speed uses the fallback scale (torso = "
                           f"{cfg.fallback_torso_m:g} m); calibrated=false")
    out.n_steps = len(intervals)
    if len(intervals) >= 2:
        T = np.array(intervals)
        out.cadence_spm = 60.0 / float(T.mean())
        out.step_cv_pct = 100.0 * float(T.std(ddof=1)) / float(T.mean())
    else:
        out.reasons.append(f"only {len(intervals)} step intervals: no cadence or CV")
    tl = [iv for iv, s in zip(intervals, sides) if s == "L"]
    tr = [iv for iv, s in zip(intervals, sides) if s == "R"]
    out.n_left, out.n_right = len(tl), len(tr)
    if len(tl) >= cfg.min_steps_per_side and len(tr) >= cfg.min_steps_per_side:
        out.symmetry_pct = symmetry_index(float(np.mean(tl)), float(np.mean(tr)))
    else:
        out.reasons.append(f"symmetry needs {cfg.min_steps_per_side} steps per side "
                           f"(have L={len(tl)}, R={len(tr)})")
    return out


def samples_from_steps(steps: Iterable) -> list[GaitSample]:
    """GaitSamples from pipeline Steps (objects with ``clean`` and ``features``)."""
    return [sample_from(s.clean, s.features) for s in steps]
