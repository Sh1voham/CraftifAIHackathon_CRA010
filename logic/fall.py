"""Fall state machine.

Fast path (spec):
    STANDING --v > V_DROP (hip was above H_LOW)--> FALLING
    FALLING  --phi > PHI_DOWN and h < H_LOW within falling_timeout_s--> DOWN
    FALLING  --timeout--> STANDING                (a sit or a bend)
    DOWN     --phi < PHI_RECOVER for recover_hold_s--> STANDING
    DOWN     --down condition held for T_HOLD--> FALL_CONFIRMED (event)
    FALL_CONFIRMED --phi < PHI_RECOVER for confirmed_recover_hold_s--> STANDING (RECOVERED)

Slow path (prolonged lying without a fast drop):
    STANDING --phi > PHI_DOWN and h < H_LOW held for T_PROLONGED--> PROLONGED_DOWN (event)
    PROLONGED_DOWN --same recovery rule as FALL_CONFIRMED--> STANDING (RECOVERED)

The down conditions may break for up to ``down_break_tolerance_s`` without
restarting their timers. While the torso is not visible in a down state the
timers pause; after ``lost_person_grace_s`` the machine resets to STANDING.
If ``h`` is unknown the fast path uses ``phi`` alone; the slow path needs ``h``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from logic.config import FallCfg
from logic.features import Features


class State(str, Enum):
    """Fall detector states."""

    STANDING = "STANDING"
    FALLING = "FALLING"
    DOWN = "DOWN"
    FALL_CONFIRMED = "FALL_CONFIRMED"
    PROLONGED_DOWN = "PROLONGED_DOWN"


ALERT_KINDS = ("FALL_CONFIRMED", "PROLONGED_DOWN")


@dataclass(frozen=True)
class Event:
    """A detector output; ``ts_s`` is the frame timestamp."""

    ts_s: float
    kind: str  # "FALL_CONFIRMED" | "PROLONGED_DOWN" | "RECOVERED"
    confidence: float
    torso_angle_deg: float
    hip_drop_speed_Lps: float
    time_down_s: float


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class FallDetector:
    """Consumes per-frame Features and emits fall Events."""

    def __init__(self, cfg: FallCfg) -> None:
        self.cfg = cfg
        self._state = State.STANDING
        self._last: Features | None = None
        self._last_alert_t: float | None = None
        self._lost_since: float | None = None
        self._v_since_upright = 0.0
        self._reset()

    @property
    def state(self) -> State:
        """Current state (read-only)."""
        return self._state

    @property
    def last_features(self) -> Features | None:
        """Features from the latest update (read-only)."""
        return self._last

    def _reset(self) -> None:
        self._state = State.STANDING
        self._t_fall = 0.0
        self._v_peak = 0.0
        self._down_start: float | None = None
        self._down_ok = 0.0
        self._upright_since: float | None = None
        self._slow_start: float | None = None
        self._slow_ok = 0.0
        self._alert_emitted = False
        self._alert_conf = 0.0
        self._alert_start = 0.0

    # ------------------------------------------------------------------ helpers

    def _fast_down(self, f: Features) -> bool:
        assert f.phi is not None
        return f.phi > self.cfg.PHI_DOWN and (f.h is None or f.h < self.cfg.H_LOW)

    def _slow_down(self, f: Features) -> bool:
        assert f.phi is not None
        return f.phi > self.cfg.PHI_DOWN and f.h is not None and f.h < self.cfg.H_LOW

    def _confidence(self, v_peak: float, phi: float, elapsed: float, hold: float) -> float:
        c = self.cfg
        c_v = _clip01(v_peak / (c.conf_v_scale * c.V_DROP))
        c_phi = _clip01((phi - c.conf_phi_lo_deg) / c.conf_phi_span_deg)
        c_t = _clip01(elapsed / hold)
        return c.conf_w_v * c_v + c.conf_w_phi * c_phi + c.conf_w_t * c_t

    def _alert(self, t: float, kind: str, phi: float, v_peak: float, start: float, hold: float) -> list[Event]:
        conf = self._confidence(v_peak, phi, t - start, hold)
        self._alert_conf, self._alert_start = conf, start
        if self._last_alert_t is not None and t - self._last_alert_t < self.cfg.alert_cooldown_s:
            self._alert_emitted = False
            return []
        self._last_alert_t = t
        self._alert_emitted = True
        return [Event(t, kind, conf, phi, v_peak, t - start)]

    def _recovered(self, t: float, phi: float, hold: float) -> list[Event]:
        """Track upright time; on recovery reset and return any RECOVERED event."""
        if phi >= self.cfg.PHI_RECOVER:
            self._upright_since = None
            return []
        if self._upright_since is None:
            self._upright_since = t
        if t - self._upright_since < hold:
            return []
        events = []
        if self._state in (State.FALL_CONFIRMED, State.PROLONGED_DOWN) and self._alert_emitted:
            events.append(Event(t, "RECOVERED", self._alert_conf, phi, self._v_peak,
                                self._upright_since - self._alert_start))
        self._reset()
        return events

    def _shift_timers(self, gap: float) -> None:
        if self._down_start is not None:
            self._down_start += gap
        self._down_ok += gap
        self._slow_ok += gap
        if self._slow_start is not None:
            self._slow_start += gap
        if self._upright_since is not None:
            self._upright_since += gap
        self._alert_start += gap

    # ------------------------------------------------------------------ update

    def update(self, t: float, f: Features | None) -> list[Event]:
        """Advance the machine with the features of the frame at ``t``."""
        self._last = f
        c = self.cfg
        if f is None or f.phi is None:
            return self._update_lost(t)
        if self._lost_since is not None:
            if self._state is not State.FALLING:
                self._shift_timers(t - self._lost_since)
            self._lost_since = None
        phi = f.phi
        if phi < c.PHI_RECOVER:
            self._v_since_upright = 0.0
        if f.v is not None and f.v > self._v_since_upright:
            self._v_since_upright = f.v

        if self._state is State.STANDING:
            if f.v is not None and f.v > c.V_DROP and (f.h_prev is None or f.h_prev >= c.H_LOW):
                self._state, self._t_fall, self._v_peak = State.FALLING, t, f.v
                self._slow_start = None
            else:
                return self._update_slow(t, f)

        if self._state is State.FALLING:
            if f.v is not None and f.v > self._v_peak:
                self._v_peak = f.v
            if self._fast_down(f):
                self._state, self._down_start, self._down_ok = State.DOWN, t, t
                self._upright_since = None
            elif t - self._t_fall > c.falling_timeout_s:
                self._reset()
            return []

        if self._state is State.DOWN:
            if self._fast_down(f):
                self._down_ok = t
                if self._down_start is None:
                    self._down_start = t
                if t - self._down_start >= c.T_HOLD:
                    self._state = State.FALL_CONFIRMED
                    self._upright_since = None
                    return self._alert(t, "FALL_CONFIRMED", phi, self._v_peak, self._down_start, c.T_HOLD)
            elif self._down_start is not None and t - self._down_ok > c.down_break_tolerance_s:
                self._down_start = None  # condition broken: the hold timer restarts
            return self._recovered(t, phi, c.recover_hold_s)

        # FALL_CONFIRMED or PROLONGED_DOWN
        return self._recovered(t, phi, c.confirmed_recover_hold_s)

    def _update_slow(self, t: float, f: Features) -> list[Event]:
        c = self.cfg
        if self._slow_down(f):
            self._slow_ok = t
            if self._slow_start is None:
                self._slow_start = t
            if t - self._slow_start >= c.T_PROLONGED:
                self._state = State.PROLONGED_DOWN
                self._v_peak = self._v_since_upright
                self._upright_since = None
                assert f.phi is not None
                return self._alert(t, "PROLONGED_DOWN", f.phi, self._v_peak, self._slow_start, c.T_PROLONGED)
        elif self._slow_start is not None and t - self._slow_ok > c.down_break_tolerance_s:
            self._slow_start = None
        return []

    def _update_lost(self, t: float) -> list[Event]:
        c = self.cfg
        if self._lost_since is None:
            self._lost_since = t
        lost_for = t - self._lost_since
        if self._state is State.FALLING:
            if t - self._t_fall > c.falling_timeout_s:
                self._reset()
        elif lost_for > c.lost_person_grace_s:
            self._reset()
        return []
