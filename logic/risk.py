"""Combined fall-risk band and per-person walking-speed baseline.

The ref/bad values and weights in ``config.json`` are illustrative design
choices, not a validated clinical model. This is a screening aid that tells a
carer to look closer; it does not diagnose.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from logic.config import RiskCfg, RiskMetricCfg

METRICS = ("walk_speed", "symmetry", "step_cv", "tug")


def sub_score(value: float, m: RiskMetricCfg) -> float:
    """0 (good) .. 1 (poor) for one metric."""
    s = (value - m.ref) / (m.bad - m.ref) if m.higher_is_worse else (m.ref - value) / (m.ref - m.bad)
    return float(min(max(s, 0.0), 1.0))


def band_for(score: float, cfg: RiskCfg) -> str:
    """LOW / MODERATE / HIGH."""
    if score >= cfg.band_high:
        return "HIGH"
    return "MODERATE" if score >= cfg.band_moderate else "LOW"


@dataclass
class RiskResult:
    """Weighted score over the metrics that are available."""

    score: float | None
    band: str | None
    sub_scores: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


def risk_score(values: dict[str, float | None], cfg: RiskCfg) -> RiskResult:
    """S = sum(w_i s_i) / sum(w_i) over available metrics; None if none are available."""
    subs, missing = {}, []
    num = den = 0.0
    for name in METRICS:
        v = values.get(name)
        if v is None:
            missing.append(name)
            continue
        m: RiskMetricCfg = getattr(cfg, name)
        subs[name] = sub_score(v, m)
        num += m.weight * subs[name]
        den += m.weight
    if den == 0:
        return RiskResult(None, None, subs, missing)
    score = num / den
    return RiskResult(score, band_for(score, cfg), subs, missing)


@dataclass
class DriftResult:
    """Outcome of adding one session's walking speed to a person's baseline."""

    speed: float
    baseline: float
    baseline_before: float | None
    z: float | None
    n_history: int
    drift_alert: bool
    reasons: list[str] = field(default_factory=list)


class BaselineStore:
    """Per-person walking-speed baseline kept in a small JSON file.

    Baseline b_t = (1 - beta) b_(t-1) + beta v_t. Drift alert when
    v_t < drift_ratio * b_t, or when z = (v_t - mu) / sigma < drift_z over the
    person's last sessions (needs at least ``drift_min_sessions``).
    """

    def __init__(self, path: str | Path, cfg: RiskCfg) -> None:
        self.path = Path(path)
        self.cfg = cfg
        self.data: dict[str, Any] = {}
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def update(self, person: str, speed: float) -> DriftResult:
        """Add one session for ``person`` and persist the store."""
        cfg = self.cfg
        rec = self.data.get(person, {"baseline": None, "history": []})
        before = rec["baseline"]
        history = list(rec["history"])
        b = speed if before is None else (1 - cfg.baseline_beta) * before + cfg.baseline_beta * speed
        reasons = []
        alert = False
        if before is not None and speed < cfg.drift_ratio * b:
            alert = True
            reasons.append(f"speed {speed:.2f} m/s below {cfg.drift_ratio:g} x baseline {b:.2f} m/s")
        z = None
        if len(history) >= cfg.drift_min_sessions:
            recent = np.array(history[-cfg.history_max:])
            sigma = float(recent.std(ddof=1))
            if sigma > 0:
                z = (speed - float(recent.mean())) / sigma
                if z < cfg.drift_z:
                    alert = True
                    reasons.append(f"z = {z:.2f} over the last {len(recent)} sessions")
        history = (history + [speed])[-cfg.history_max:]
        self.data[person] = {"baseline": b, "history": history}
        self._save()
        return DriftResult(speed, b, before, z, len(history) - 1, alert, reasons)

    def _save(self) -> None:
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)


def summary_payload(ts: str, gait: Any, tug: Any, risk: RiskResult, drift: DriftResult | None = None,
                    **extra: Any) -> dict[str, Any]:
    """Gait summary for MQTT ``safestride/risk`` (documented schema plus details)."""
    def r(v: float | None, nd: int) -> float | None:
        return None if v is None else round(float(v), nd)

    out: dict[str, Any] = {
        "ts": ts,
        "walk_speed_mps": r(gait.walk_speed_mps, 2),
        "cadence_spm": r(gait.cadence_spm, 1),
        "step_cv_pct": r(gait.step_cv_pct, 1),
        "symmetry_pct": r(gait.symmetry_pct, 1),
        "tug_s": r(tug.tug_s, 1) if tug is not None and tug.valid else None,
        "risk_score": r(risk.score, 2),
        "band": risk.band,
        "calibrated": gait.calibrated,
        "drift_alert": None if drift is None else drift.drift_alert,
        "n_steps": gait.n_steps,
        "missing": risk.missing,
        "notes": list(gait.reasons) + ([tug.reason] if tug is not None and not tug.valid and tug.reason else []),
    }
    if drift is not None:
        out.update(baseline_mps=r(drift.baseline, 2), drift_reasons=drift.reasons)
    out.update(extra)
    return out
