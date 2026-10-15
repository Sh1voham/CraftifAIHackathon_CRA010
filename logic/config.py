"""Tunable parameters for the SafeStride decision layer.

Every number used by the algorithms lives here. Defaults below mirror
``logic/config.json``; ``load_config`` overlays a JSON file on top of them and
rejects unknown keys so typos fail loudly.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


@dataclass(frozen=True)
class OneEuroCfg:
    """One Euro filter parameters (pixel units)."""

    min_cutoff: float = 1.0
    beta: float = 0.02
    d_cutoff: float = 1.0
    reset_gap_s: float = 1.0  # reset the filter after a gap longer than this


@dataclass(frozen=True)
class FilterCfg:
    """Keypoint cleaning: gating, outliers, gap repair, smoothing, body scale."""

    c_min: float = 0.30
    max_gap_frames: int = 5
    outlier_jump_Ls: float = 0.5  # per typical frame interval; scales up across dropped frames
    frame_dt_window: int = 15  # recent frame intervals used to estimate the typical interval
    Lref_window_s: float = 2.0
    one_euro: OneEuroCfg = field(default_factory=OneEuroCfg)


@dataclass(frozen=True)
class FeatureCfg:
    """Per-frame feature extraction and floor-line estimation."""

    drop_window_s: float = 0.25
    drop_window_max_s: float = 0.75  # no reference sample this old or younger -> v is None
    floor_upright_phi_deg: float = 20.0
    floor_learn_s: float = 5.0  # upright time used to learn the floor, then frozen
    floor_min_learn_s: float = 0.5  # upright time needed before a provisional floor exists
    floor_percentile: float = 90.0
    floor_min_ankle_fraction: float = 0.3  # share of upright samples with an ankle
    floor_fallback_Ls: float = 1.2  # y_floor = median standing hip y + this * L_ref
    upright_dt_cap_s: float = 0.2  # cap per-frame dt when accumulating upright time
    y_floor_override: float | None = None


@dataclass(frozen=True)
class FallCfg:
    """Fall state machine thresholds and timers."""

    V_DROP: float = 1.2
    PHI_DOWN: float = 60.0
    H_LOW: float = 0.5
    T_HOLD: float = 3.0
    PHI_RECOVER: float = 30.0
    T_PROLONGED: float = 10.0
    recover_hold_s: float = 2.0
    confirmed_recover_hold_s: float = 5.0
    falling_timeout_s: float = 1.5
    alert_cooldown_s: float = 30.0
    lost_person_grace_s: float = 2.0
    down_break_tolerance_s: float = 0.3
    conf_w_v: float = 0.30
    conf_w_phi: float = 0.35
    conf_w_t: float = 0.35
    conf_v_scale: float = 2.0  # c_v = v_peak / (conf_v_scale * V_DROP)
    conf_phi_lo_deg: float = 30.0
    conf_phi_span_deg: float = 60.0


@dataclass(frozen=True)
class GaitCfg:
    """Gait metrics from steady side-on walking."""

    vx_window_s: float = 0.5  # central-difference window for hip horizontal speed
    steady_min_speed_Ls: float = 0.3  # torso lengths per second
    steady_speed_cv_max: float = 0.25  # rolling std / mean of hip speed
    steady_roll_s: float = 1.0
    steady_max_phi_deg: float = 25.0
    steady_max_gap_s: float = 0.3  # a longer hole in the data ends a window
    steady_min_len_s: float = 1.5
    turn_trim_s: float = 0.3  # trimmed from both ends of each window
    peak_min_distance_s: float = 0.25
    peak_prominence_Ls: float = 0.1
    peak_fit_half_s: float = 0.1  # least-squares parabola around each step peak
    min_steps_per_side: int = 4
    fallback_torso_m: float = 0.50  # rough scale when no calibration is given
    buffer_s: float = 60.0  # live: frames kept for gait analysis
    update_period_s: float = 2.0  # live: recompute every this many seconds of frame time


@dataclass(frozen=True)
class TugCfg:
    """Timed Up and Go segmentation."""

    yn_low: float = 0.1
    yn_high: float = 0.9
    y_percentile_lo: float = 2.0  # robust standing (min y) and seated (max y) levels
    y_percentile_hi: float = 98.0
    min_range_Ls: float = 0.4  # hip must travel at least this far between sitting and standing
    vx_window_s: float = 0.5
    turn_speed_Ls: float = 0.2  # minimum hysteresis threshold on hip speed, torso lengths/s
    turn_speed_frac: float = 0.5  # turn = hip speed below this fraction of the median walking speed


@dataclass(frozen=True)
class RiskMetricCfg:
    """Sub-score mapping for one metric (illustrative, not clinically validated)."""

    ref: float
    bad: float
    weight: float
    higher_is_worse: bool = True


@dataclass(frozen=True)
class RiskCfg:
    """Combined fall-risk score and per-person baseline (illustrative design choices)."""

    walk_speed: RiskMetricCfg = field(default_factory=lambda: RiskMetricCfg(1.2, 0.6, 0.35, False))
    symmetry: RiskMetricCfg = field(default_factory=lambda: RiskMetricCfg(5.0, 25.0, 0.25))
    step_cv: RiskMetricCfg = field(default_factory=lambda: RiskMetricCfg(3.0, 10.0, 0.20))
    tug: RiskMetricCfg = field(default_factory=lambda: RiskMetricCfg(8.0, 16.0, 0.20))
    band_moderate: float = 0.33
    band_high: float = 0.66
    baseline_beta: float = 0.2
    drift_ratio: float = 0.8
    drift_z: float = -2.0
    drift_min_sessions: int = 5
    history_max: int = 20


@dataclass(frozen=True)
class IoCfg:
    """Live transport settings."""

    poll_s: float = 0.05
    file_settle_s: float = 1.0  # give up on an unparseable per-frame file after this


@dataclass(frozen=True)
class MqttCfg:
    """MQTT output settings."""

    topic_alerts: str = "safestride/alerts"
    topic_risk: str = "safestride/risk"
    topic_status: str = "safestride/status"
    qos: int = 1
    heartbeat_s: float = 5.0
    keepalive_s: int = 30
    reconnect_min_s: int = 1
    reconnect_max_s: int = 30
    max_queued: int = 1000


@dataclass(frozen=True)
class PrivacyCfg:
    """Blur applied before any frame is written."""

    expand_frac: float = 0.15  # added on each side of the person box
    kernel: int = 51
    sigma: float = 20.0


@dataclass(frozen=True)
class EvalCfg:
    """Evaluation and threshold tuning."""

    match_window_s: float = 20.0
    test_fraction: float = 0.3
    split_seed: int = 0
    grid_V_DROP: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5, 1.8)
    grid_PHI_DOWN: tuple[float, ...] = (50.0, 55.0, 60.0, 65.0, 70.0)
    grid_H_LOW: tuple[float, ...] = (0.3, 0.4, 0.5, 0.7)
    grid_T_HOLD: tuple[float, ...] = (2.5, 3.0, 3.5)


@dataclass(frozen=True)
class Config:
    """Top-level configuration."""

    filters: FilterCfg = field(default_factory=FilterCfg)
    features: FeatureCfg = field(default_factory=FeatureCfg)
    fall: FallCfg = field(default_factory=FallCfg)
    gait: GaitCfg = field(default_factory=GaitCfg)
    tug: TugCfg = field(default_factory=TugCfg)
    risk: RiskCfg = field(default_factory=RiskCfg)
    io: IoCfg = field(default_factory=IoCfg)
    mqtt: MqttCfg = field(default_factory=MqttCfg)
    privacy: PrivacyCfg = field(default_factory=PrivacyCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)


def _merge(base: Any, data: dict[str, Any], where: str) -> Any:
    """Return a copy of dataclass ``base`` with values from ``data``."""
    names = {f.name for f in dataclasses.fields(base)}
    changes: dict[str, Any] = {}
    for key, value in data.items():
        if key not in names:
            raise ValueError(f"unknown config key '{where}{key}'")
        current = getattr(base, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"config key '{where}{key}' must be an object")
            changes[key] = _merge(current, value, f"{where}{key}.")
        elif isinstance(current, tuple):
            changes[key] = tuple(value)
        else:
            changes[key] = value
    return dataclasses.replace(base, **changes)


def config_from_dict(data: dict[str, Any]) -> Config:
    """Build a Config from a (possibly partial) nested dict."""
    return _merge(Config(), data, "")


def load_config(path: str | Path | None = None) -> Config:
    """Load ``path`` (default ``logic/config.json``) over the built-in defaults."""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(p, encoding="utf-8") as fh:
        return config_from_dict(json.load(fh))


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Nested plain-dict view of a Config (tuples become lists)."""
    return json.loads(json.dumps(dataclasses.asdict(cfg)))


def save_config(cfg: Config, path: str | Path) -> None:
    """Write ``cfg`` as pretty JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config_to_dict(cfg), fh, indent=2)
        fh.write("\n")
