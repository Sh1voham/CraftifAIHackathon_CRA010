"""Tests for logic.risk."""
import json

import pytest

from logic.config import RiskCfg
from logic.gait import GaitSummary
from logic.risk import BaselineStore, band_for, risk_score, sub_score, summary_payload
from logic.tug import TugResult

CFG = RiskCfg()


def test_guide_worked_example() -> None:
    r = risk_score({"walk_speed": 0.8, "symmetry": 13.8, "step_cv": 6.0, "tug": 11.0}, CFG)
    assert r.sub_scores["walk_speed"] == pytest.approx(0.667, abs=0.001)
    assert r.sub_scores["symmetry"] == pytest.approx(0.44)
    assert r.sub_scores["step_cv"] == pytest.approx(0.4286, abs=1e-3)
    assert r.sub_scores["tug"] == pytest.approx(0.375)
    assert r.score == pytest.approx(0.50, abs=0.005) and r.band == "MODERATE"


def test_sub_scores_clip_and_direction() -> None:
    assert sub_score(1.5, CFG.walk_speed) == 0.0 and sub_score(0.3, CFG.walk_speed) == 1.0
    assert sub_score(2.0, CFG.symmetry) == 0.0 and sub_score(40.0, CFG.symmetry) == 1.0


def test_bands() -> None:
    assert [band_for(s, CFG) for s in (0.0, 0.329, 0.33, 0.659, 0.66, 1.0)] == \
           ["LOW", "LOW", "MODERATE", "MODERATE", "HIGH", "HIGH"]


def test_missing_metrics_renormalise_and_none_available() -> None:
    r = risk_score({"walk_speed": 0.6, "symmetry": None, "step_cv": None}, CFG)
    assert r.score == 1.0 and r.band == "HIGH" and set(r.missing) == {"symmetry", "step_cv", "tug"}
    r2 = risk_score({"walk_speed": 1.2, "symmetry": 25.0}, CFG)
    assert r2.score == pytest.approx(0.25 / 0.60)
    empty = risk_score({}, CFG)
    assert empty.score is None and empty.band is None and len(empty.missing) == 4


def test_baseline_ewma_ratio_alert_and_persistence(tmp_path) -> None:
    path = tmp_path / "b.json"
    store = BaselineStore(path, CFG)
    first = store.update("alice", 1.0)
    assert first.baseline == 1.0 and first.baseline_before is None and not first.drift_alert
    d = store.update("alice", 0.7)
    assert d.baseline == pytest.approx(0.94) and d.drift_alert  # 0.7 < 0.8 * 0.94
    assert not BaselineStore(path, CFG).update("bob", 0.5).drift_alert  # separate person
    reloaded = json.loads(path.read_text())
    assert reloaded["alice"]["history"] == [1.0, 0.7] and "bob" in reloaded


def test_z_score_drift_needs_five_sessions(tmp_path) -> None:
    store = BaselineStore(tmp_path / "b.json", CFG)
    for v in (1.00, 1.02, 0.98, 1.01, 0.99):
        assert store.update("p", v).z is None  # fewer than drift_min_sessions before this one
    d = store.update("p", 0.95)  # 5 prior sessions, small drop: z << -2 but ratio not triggered
    assert d.z is not None and d.z < -2 and d.drift_alert
    assert any("z =" in r for r in d.reasons)


def test_summary_payload_schema() -> None:
    g = GaitSummary(valid=True, calibrated=False, walk_speed_mps=0.8, cadence_spm=96.04, step_cv_pct=6.0,
                    symmetry_pct=13.8, n_steps=10, reasons=["fallback scale"])
    tug = TugResult(True, None, 1.2, 11.04)
    r = risk_score({"walk_speed": 0.8, "symmetry": 13.8, "step_cv": 6.0, "tug": 11.04}, CFG)
    p = summary_payload("2026-10-15T20:41:07.512+05:30", g, tug, r, person="alice")
    for k in ("ts", "walk_speed_mps", "cadence_spm", "step_cv_pct", "symmetry_pct", "tug_s", "risk_score", "band"):
        assert k in p
    assert p["tug_s"] == 11.0 and p["band"] == "MODERATE" and p["person"] == "alice" and p["calibrated"] is False
    bad_tug = summary_payload("x", g, TugResult(False, "turn not found"), r)
    assert bad_tug["tug_s"] is None and "turn not found" in bad_tug["notes"]
