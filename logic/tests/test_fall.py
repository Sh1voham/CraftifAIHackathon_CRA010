"""Tests for logic.fall: every transition of the fall state machine."""
import pytest

from logic.config import FallCfg
from logic.fall import FallDetector, State
from logic.features import Features

DT = 1 / 30
CFG = FallCfg()


def F(t: float, phi: float | None, v: float | None = 0.0, h: float | None = 1.8,
      h_prev: float | None = 1.8) -> Features:
    return Features(t=t, visible=phi is not None, phi=phi, v=v, h=h, h_prev=h_prev)


class Script:
    """Feeds a detector segment by segment at ~30 fps and records events and states."""

    def __init__(self, cfg: FallCfg = CFG) -> None:
        self.det = FallDetector(cfg)
        self.t = 0.0
        self.events = []
        self.states = []

    def run(self, seconds: float, **kw) -> "Script":
        end = self.t + seconds
        while self.t < end - 1e-9:
            self.t = round(self.t + DT, 9)
            f = None if kw.get("lost") else F(self.t, kw.get("phi", 5.0), kw.get("v", 0.0),
                                                kw.get("h", 1.8), kw.get("h_prev", 1.8))
            self.events += self.det.update(self.t, f)
            self.states.append(self.det.state)
        return self

    def stand(self, s: float) -> "Script":
        return self.run(s, phi=5.0)

    def drop(self) -> "Script":
        return self.run(0.2, phi=40.0, v=2.5, h=1.0, h_prev=1.8)

    def lie(self, s: float, h: float | None = 0.2) -> "Script":
        return self.run(s, phi=85.0, v=0.0, h=h, h_prev=h)

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]


def test_fast_fall_confirms_after_t_hold_with_confidence() -> None:
    s = Script().stand(1.0).drop()
    assert s.det.state is State.FALLING
    t_down = s.t + DT
    s.lie(2.9)
    assert s.det.state is State.DOWN and s.events == []
    s.lie(0.5)
    assert s.det.state is State.FALL_CONFIRMED
    (ev,) = s.events
    assert ev.kind == "FALL_CONFIRMED"
    assert ev.ts_s - t_down == pytest.approx(3.0, abs=DT + 1e-6)
    assert ev.time_down_s >= 3.0 and ev.hip_drop_speed_Lps == 2.5 and ev.torso_angle_deg == 85.0
    assert ev.confidence == pytest.approx(0.30 * 1.0 + 0.35 * (55 / 60) + 0.35 * 1.0)


def test_falling_times_out_back_to_standing_for_a_fast_sit() -> None:
    s = Script().stand(1.0).run(0.3, phi=10.0, v=1.6, h=1.3)
    assert s.det.state is State.FALLING
    s.run(1.0, phi=10.0, v=0.0, h=1.0)
    assert s.det.state is State.FALLING
    s.run(0.6, phi=10.0, v=0.0, h=1.0)
    assert s.det.state is State.STANDING and s.events == []


def test_falling_timeout_still_runs_while_person_lost() -> None:
    s = Script().stand(1.0).drop().run(2.0, lost=True)
    assert s.det.state is State.STANDING


def test_down_recovers_to_standing_before_hold() -> None:
    s = Script().stand(1.0).drop().lie(1.0).stand(1.9)
    assert s.det.state is State.DOWN
    s.stand(0.2)
    assert s.det.state is State.STANDING
    s.stand(5.0)
    assert s.events == []


def test_down_break_tolerance() -> None:
    short = Script().stand(1.0).drop().lie(2.0).run(0.2, phi=50.0, h=0.4).lie(0.9)
    assert short.kinds() == ["FALL_CONFIRMED"]  # 0.2 s blip did not restart the timer
    long = Script().stand(1.0).drop().lie(2.0).run(0.5, phi=50.0, h=0.4).lie(2.5)
    assert long.events == []  # timer restarted after the 0.5 s break
    long.lie(0.6)
    assert long.kinds() == ["FALL_CONFIRMED"]


def test_fast_path_uses_phi_alone_when_h_unknown() -> None:
    s = Script().stand(1.0).run(0.2, phi=40.0, v=2.5, h=None, h_prev=None).lie(3.2, h=None)
    assert s.kinds() == ["FALL_CONFIRMED"]


def test_recovered_after_confirmed_recover_hold() -> None:
    s = Script().stand(1.0).drop().lie(3.5).stand(4.9)
    assert s.det.state is State.FALL_CONFIRMED and s.kinds() == ["FALL_CONFIRMED"]
    s.stand(0.2)
    assert s.det.state is State.STANDING and s.kinds() == ["FALL_CONFIRMED", "RECOVERED"]
    assert s.events[1].time_down_s == pytest.approx(3.5, abs=0.1)


def test_recovery_hold_resets_if_person_slumps_again() -> None:
    s = Script().stand(1.0).drop().lie(3.5).stand(3.0).lie(1.0).stand(3.0)
    assert s.det.state is State.FALL_CONFIRMED and s.kinds() == ["FALL_CONFIRMED"]


def test_lost_person_pauses_timers_within_grace() -> None:
    s = Script().stand(1.0).drop().lie(2.0).run(1.5, lost=True).lie(0.9)
    assert s.events == []  # 2.0 + 0.9 s visibly down < T_HOLD; lost time does not count
    s.lie(0.2)
    assert s.kinds() == ["FALL_CONFIRMED"]


def test_lost_person_beyond_grace_resets() -> None:
    s = Script().stand(1.0).drop().lie(2.0).run(2.2, lost=True)
    assert s.det.state is State.STANDING
    s.lie(5.0)
    assert s.events == []  # back in STANDING: no fast drop, slow path needs T_PROLONGED
    confirmed = Script().stand(1.0).drop().lie(3.5).run(2.2, lost=True)
    assert confirmed.det.state is State.STANDING  # also resets FALL_CONFIRMED


def test_cooldown_suppresses_duplicate_alerts() -> None:
    s = Script().stand(1.0).drop().lie(3.5).stand(5.5)  # alert 1 + RECOVERED at ~t=10
    s.stand(1.0).drop().lie(3.5)  # second fall within 30 s of the first alert
    assert s.det.state is State.FALL_CONFIRMED
    assert s.kinds() == ["FALL_CONFIRMED", "RECOVERED"]
    s.stand(5.5)  # suppressed alert -> no RECOVERED either
    assert s.kinds() == ["FALL_CONFIRMED", "RECOVERED"]
    s.stand(15.0).drop().lie(3.5)  # > 30 s after the first alert
    assert s.kinds() == ["FALL_CONFIRMED", "RECOVERED", "FALL_CONFIRMED"]


def test_no_fast_drop_trigger_when_hip_already_low() -> None:
    s = Script().run(1.0, phi=85.0, h=0.2, h_prev=0.2, v=0.0)
    s.run(0.2, phi=85.0, v=2.0, h=0.1, h_prev=0.2)  # jitter while lying
    assert s.det.state is State.STANDING


def test_prolonged_down_after_slow_lowering() -> None:
    s = Script().stand(1.0).run(4.0, phi=40.0, v=0.4, h=1.0, h_prev=1.1)
    t_lying = s.t + DT
    s.lie(9.9)
    assert s.events == [] and s.det.state is State.STANDING
    s.lie(0.2)
    assert s.det.state is State.PROLONGED_DOWN
    (ev,) = s.events
    assert ev.kind == "PROLONGED_DOWN" and ev.ts_s - t_lying == pytest.approx(10.0, abs=DT + 1e-6)
    assert ev.hip_drop_speed_Lps == pytest.approx(0.4)
    s.stand(5.1)
    assert s.kinds() == ["PROLONGED_DOWN", "RECOVERED"] and s.det.state is State.STANDING


def test_prolonged_down_not_for_brief_lying_or_unknown_h() -> None:
    brief = Script().stand(1.0).run(3.0, phi=40.0, v=0.4, h=1.0).lie(5.0).stand(3.0)
    assert brief.events == []
    no_floor = Script().stand(1.0).run(3.0, phi=40.0, v=0.4, h=None, h_prev=None).lie(20.0, h=None)
    assert no_floor.events == []
    bed = Script().stand(1.0).run(3.0, phi=40.0, v=0.4, h=1.2).lie(20.0, h=1.0)  # hip on a bed
    assert bed.events == []


def test_fast_fall_backstop_via_prolonged_when_down_reached_late() -> None:
    # Fast drop, but the person only ends up flat after the FALLING timeout.
    s = Script().stand(1.0).drop().run(2.0, phi=45.0, h=0.4).lie(10.2)
    assert s.kinds() == ["PROLONGED_DOWN"]
