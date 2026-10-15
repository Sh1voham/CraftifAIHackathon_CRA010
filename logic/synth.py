"""Synthetic keypoint clips with ground truth, for testing and tuning.

A 2D stick figure is posed in metres (x right, y up from the floor) and
projected to pixels with a per-clip ``px_per_m``. Side-view clips move in the
image plane; sideways falls use a frontal view (person facing the camera).
Timestamps are ~30 fps with +/-3 ms jitter and dropped frames; keypoints get
Gaussian noise (0.01 torso lengths), ~3 % dropouts in 1-3 frame bursts,
random confidences in [0.5, 0.95], and are removed when outside the image.

    python -m logic.synth --out data/synth --n 200 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from logic.io import KEYPOINT_NAMES, Frame, Person, write_jsonl

# ---------------------------------------------------------------- generator constants
FPS_NOMINAL = 30.0
TS_JITTER_S = 0.003
P_FRAME_DROP = 0.02
P_BURST_DROP = 0.003
NOISE_L = 0.01  # keypoint noise sigma in torso lengths
P_KP_DROPOUT_START = 0.015  # x mean burst length 2 frames ~= 3 % of keypoints missing
CONF_RANGE = (0.5, 0.95)
MIN_VISIBLE_KP = 5  # fewer keypoints inside the image -> person not detected
BYSTANDER_SCALE = 0.6


@dataclass(frozen=True)
class Body:
    """Segment lengths in metres."""

    torso: float = 0.50
    thigh: float = 0.43
    shank: float = 0.42
    ankle_h: float = 0.08
    upper_arm: float = 0.30
    forearm: float = 0.27
    shoulder_w: float = 0.19  # half widths, used in the frontal view
    hip_w: float = 0.13

    @property
    def stand_hip(self) -> float:
        return self.thigh + self.shank + self.ankle_h

    def scaled(self, k: float) -> "Body":
        return Body(*(getattr(self, f) * k for f in self.__dataclass_fields__))


@dataclass(frozen=True)
class Pose:
    """Pose parameters. Angles in degrees.

    Side view: ``theta`` trunk angle from vertical (+ towards ``facing``), leg
    angles ``aL/aR`` from vertical (+ forward), knee flexion ``kL/kR``, shoulder
    flexion ``armL/armR``. Frontal view: whole-body roll ``psi`` (+ towards
    image +x), ``kL/kR`` shorten the legs, ``armL/armR`` abduct the arms.
    """

    hx: float
    hy: float
    theta: float = 3.0
    aL: float = 0.0
    kL: float = 0.0
    aR: float = 0.0
    kR: float = 0.0
    armL: float = 0.0
    armR: float = 0.0
    psi: float = 0.0
    torso_scale: float = 1.0
    facing: int = 1
    view: str = "side"


_NUM = ("hx", "hy", "theta", "aL", "kL", "aR", "kR", "armL", "armR", "psi", "torso_scale")


def smooth(s: float) -> float:
    """Smoothstep on [0, 1]."""
    s = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s
    return s * s * (3.0 - 2.0 * s)


def lerp_pose(a: Pose, b: Pose, s: float) -> Pose:
    """Interpolate numeric fields; view and facing come from ``b``."""
    return replace(b, **{f: getattr(a, f) + (getattr(b, f) - getattr(a, f)) * s for f in _NUM})


def keyframed(keys: list[tuple[float, Pose]]) -> Callable[[float], Pose]:
    """Pose function holding the ends and smoothstep-easing between keyframes."""
    def fn(t: float) -> Pose:
        if t <= keys[0][0]:
            return keys[0][1]
        for (t0, p0), (t1, p1) in zip(keys, keys[1:]):
            if t <= t1:
                return lerp_pose(p0, p1, smooth((t - t0) / (t1 - t0)) if t1 > t0 else 1.0)
        return keys[-1][1]
    return fn


def world_keypoints(p: Pose, b: Body) -> np.ndarray:
    """(17, 2) keypoints in metres for pose ``p``."""
    out = np.zeros((len(KEYPOINT_NAMES), 2))
    idx = {n: i for i, n in enumerate(KEYPOINT_NAMES)}
    H = np.array([p.hx, p.hy])
    torso = b.torso * p.torso_scale
    if p.view == "front":
        k = math.radians(p.psi)
        rot = np.array([[math.cos(k), math.sin(k)], [-math.sin(k), math.cos(k)]])
        legL, legR = math.cos(math.radians(p.kL)), math.cos(math.radians(p.kR))
        abL, abR = math.radians(p.armL), math.radians(p.armR)
        ua, fa = b.upper_arm, b.forearm
        local = {
            "nose": (0.0, torso + 0.25), "l_eye": (0.035, torso + 0.28), "r_eye": (-0.035, torso + 0.28),
            "l_ear": (0.075, torso + 0.25), "r_ear": (-0.075, torso + 0.25),
            "l_shoulder": (b.shoulder_w, torso), "r_shoulder": (-b.shoulder_w, torso),
            "l_elbow": (b.shoulder_w + ua * math.sin(abL), torso - ua * math.cos(abL)),
            "r_elbow": (-b.shoulder_w - ua * math.sin(abR), torso - ua * math.cos(abR)),
            "l_wrist": (b.shoulder_w + (ua + fa) * math.sin(abL), torso - (ua + fa) * math.cos(abL)),
            "r_wrist": (-b.shoulder_w - (ua + fa) * math.sin(abR), torso - (ua + fa) * math.cos(abR)),
            "l_hip": (b.hip_w, 0.0), "r_hip": (-b.hip_w, 0.0),
            "l_knee": (b.hip_w - 0.01, -b.thigh * legL), "r_knee": (-b.hip_w + 0.01, -b.thigh * legR),
            "l_ankle": (b.hip_w - 0.02, -(b.thigh + b.shank) * legL - b.ankle_h * 0.5),
            "r_ankle": (-b.hip_w + 0.02, -(b.thigh + b.shank) * legR - b.ankle_h * 0.5),
        }
        for n, (u, v) in local.items():
            out[idx[n]] = H + rot @ np.array([u, v])
        return out

    d = p.facing
    th = math.radians(p.theta)
    tv = np.array([d * math.sin(th), math.cos(th)])  # hip -> shoulder
    fwd = np.array([d * math.cos(th), -math.sin(th)])  # face direction, perpendicular to the trunk
    S = H + torso * tv

    def limb(a_deg: float) -> np.ndarray:
        a = math.radians(a_deg)
        return np.array([d * math.sin(a), -math.cos(a)])

    def arm(g_deg: float) -> np.ndarray:
        g = math.radians(g_deg)
        return -tv * math.cos(g) + fwd * math.sin(g)

    pts = {
        "nose": S + 0.25 * tv + 0.09 * fwd, "l_eye": S + 0.28 * tv + 0.07 * fwd,
        "r_eye": S + 0.28 * tv + 0.07 * fwd, "l_ear": S + 0.25 * tv - 0.02 * fwd,
        "r_ear": S + 0.25 * tv - 0.02 * fwd, "l_shoulder": S, "r_shoulder": S, "l_hip": H, "r_hip": H,
    }
    for side, a, kf, g in (("l", p.aL, p.kL, p.armL), ("r", p.aR, p.kR, p.armR)):
        knee = H + b.thigh * limb(a)
        pts[f"{side}_knee"] = knee
        pts[f"{side}_ankle"] = knee + (b.shank + b.ankle_h * 0.5) * limb(a - kf)
        elbow = S + b.upper_arm * arm(g)
        pts[f"{side}_elbow"] = elbow
        pts[f"{side}_wrist"] = elbow + b.forearm * arm(g + 15.0)
    for n, v in pts.items():
        out[idx[n]] = v
    return out


# ---------------------------------------------------------------- motions

def standing(b: Body, x: float, facing: int = 1, view: str = "side", rng: np.random.Generator | None = None
             ) -> Callable[[float], Pose]:
    """Quiet standing with slow sway."""
    ph = rng.uniform(0, 2 * math.pi) if rng is not None else 0.0
    base = Pose(hx=x, hy=b.stand_hip, theta=3.0, facing=facing, view=view)

    def fn(t: float) -> Pose:
        sway = math.sin(2 * math.pi * 0.3 * t + ph)
        if view == "front":
            return replace(base, hx=x + 0.01 * sway, psi=1.0 * sway, theta=0.0)
        return replace(base, hx=x + 0.01 * sway, theta=3.0 + 1.5 * sway)
    return fn


@dataclass
class Gait:
    """Walking parameters; ``T_L``/``T_R`` are the step times ending in a left/right-led peak."""

    speed: float = 1.0
    T_L: float = 0.6
    T_R: float = 0.6
    phase0: float = 0.0

    @property
    def cadence(self) -> float:
        return 120.0 / (self.T_L + self.T_R)

    @property
    def step_len(self) -> float:
        return self.speed * (self.T_L + self.T_R) / 2.0


PEAK_SYM_RAD = 0.6  # phase kept at the mean rate on each side of a step peak


def _half_phase(tau: float, T_half: float, T_mean: float) -> float:
    """Phase advance (0..pi) after ``tau`` s of a half-cycle lasting ``T_half``.

    The first and last PEAK_SYM_RAD radians run at the mean rate, so the
    ankle-separation curve is time-symmetric around every step peak; the
    extra or missing time of an asymmetric step goes into mid-swing.
    """
    d = PEAK_SYM_RAD
    edge = d * T_mean / math.pi
    mid = T_half - 2 * edge
    if tau < edge:
        return math.pi * tau / T_mean
    if tau < edge + mid:
        return d + (math.pi - 2 * d) * (tau - edge) / mid
    return math.pi - math.pi * (T_half - tau) / T_mean


def gait_phase(g: Gait, t: float) -> float:
    """Phase phi(t): left leads at phi = pi/2 (mod 2pi), right at 3pi/2."""
    cycle = g.T_L + g.T_R
    T_mean = cycle / 2
    tau = (t + g.phase0) % cycle
    k = math.floor((t + g.phase0) / cycle)
    if tau < g.T_R:
        return 2 * math.pi * k + math.pi / 2 + _half_phase(tau, g.T_R, T_mean)
    return 2 * math.pi * k + 3 * math.pi / 2 + _half_phase(tau - g.T_R, g.T_L, T_mean)


def walk_pose(b: Body, g: Gait, x: float, facing: int, phi: float, amp_scale: float = 1.0) -> Pose:
    """Side-view walking pose at hip x for gait phase ``phi``.

    Knee flexion goes as cos^3 of the phase so it vanishes smoothly at the
    steps (maximum ankle separation) and does not shift the step peaks. The
    swing thigh flexes forward to compensate, so knee flexion lifts the foot
    instead of dragging the ankle behind the hip.
    """
    leg = b.thigh + b.shank + b.ankle_h * 0.5
    A = math.degrees(math.asin(min(g.step_len / (2 * leg), 0.9))) * amp_scale
    s, c = math.sin(phi), math.cos(phi)
    a = A * s
    kL, kR = 45.0 * max(0.0, c) ** 3 * amp_scale, 45.0 * max(0.0, -c) ** 3 * amp_scale
    comp = (b.shank + b.ankle_h * 0.5) / leg
    return Pose(hx=x, hy=b.stand_hip - 0.03 * abs(s) * amp_scale, theta=4.0, aL=a + comp * kL, aR=-a + comp * kR,
                kL=kL, kR=kR, armL=-15.0 * s * amp_scale, armR=15.0 * s * amp_scale, facing=facing)


def walking(b: Body, g: Gait, x0: float, facing: int, turn_at: float | None = None, turn_s: float = 1.2
            ) -> Callable[[float], Pose]:
    """Straight walk, optionally reversing direction at ``turn_at`` over ``turn_s`` seconds."""
    def fn(t: float) -> Pose:
        phi = gait_phase(g, t)
        if turn_at is None or t <= turn_at:
            return walk_pose(b, g, x0 + facing * g.speed * t, facing, phi)
        x_turn = x0 + facing * g.speed * turn_at
        tau = t - turn_at
        if tau < turn_s:
            vel = math.cos(math.pi * tau / turn_s)
            x = x_turn + facing * g.speed * turn_s / math.pi * math.sin(math.pi * tau / turn_s)
            f = facing if vel >= 0 else -facing
            return walk_pose(b, g, x, f, phi, amp_scale=max(abs(vel), 0.15))
        return walk_pose(b, g, x_turn - facing * g.speed * (tau - turn_s), -facing, phi)
    return fn


def then(first: Callable[[float], Pose], t_switch: float, second: Callable[[float], Pose]) -> Callable[[float], Pose]:
    """Use ``first`` before ``t_switch`` and ``second(t - t_switch)`` after."""
    return lambda t: first(t) if t < t_switch else second(t - t_switch)


# ---------------------------------------------------------------- clips

@dataclass
class Scene:
    """Camera / image model for one clip."""

    px_per_m: float = 400.0
    width: int = 1920
    height: int = 1080
    floor_px: float = 1040.0
    x_origin_px: float = 960.0  # pixel x of world x = 0


@dataclass
class Clip:
    """A generated clip and its ground truth."""

    name: str
    frames: list[Frame]
    label: dict[str, Any] = field(default_factory=dict)


def timestamps(duration: float, rng: np.random.Generator) -> list[float]:
    """Jittered ~30 fps timestamps (relative), with occasional dropped frames."""
    ts, n = [], int(duration * FPS_NOMINAL)
    i = 0
    while i < n:
        if rng.random() < P_BURST_DROP:
            i += int(rng.integers(2, 4))
            continue
        if rng.random() >= P_FRAME_DROP:
            ts.append(i / FPS_NOMINAL + rng.uniform(-TS_JITTER_S, TS_JITTER_S))
        i += 1
    return ts


def render(pose_fn: Callable[[float], Pose], body: Body, scene: Scene, duration: float, rng: np.random.Generator,
           t0: float, with_bbox: bool, bystander: tuple[Callable[[float], Pose], Body] | None = None
           ) -> list[Frame]:
    """Sample a pose function into noisy, partially missing keypoint frames."""
    actors = [(pose_fn, body)] + ([bystander] if bystander is not None else [])
    countdown = np.zeros((len(actors), len(KEYPOINT_NAMES)), dtype=int)
    frames = []
    for t in timestamps(duration, rng):
        persons = []
        for ai, (fn, b) in enumerate(actors):
            w = world_keypoints(fn(t), b)
            px = np.empty_like(w)
            px[:, 0] = scene.x_origin_px + w[:, 0] * scene.px_per_m
            px[:, 1] = scene.floor_px - w[:, 1] * scene.px_per_m
            torso_px = b.torso * scene.px_per_m
            noisy = px + rng.normal(0.0, NOISE_L * torso_px, size=px.shape)
            conf = rng.uniform(*CONF_RANGE, size=len(KEYPOINT_NAMES))
            inside = (px[:, 0] >= 0) & (px[:, 0] < scene.width) & (px[:, 1] >= 0) & (px[:, 1] < scene.height)
            kp: dict[str, tuple[float, float, float]] = {}
            for i, name in enumerate(KEYPOINT_NAMES):
                if countdown[ai, i] > 0:
                    countdown[ai, i] -= 1
                    dropped = True
                elif rng.random() < P_KP_DROPOUT_START:
                    countdown[ai, i] = int(rng.integers(1, 4)) - 1
                    dropped = True
                else:
                    dropped = False
                if not inside[i]:
                    continue
                kp[name] = (0.0, 0.0, 0.0) if dropped else (
                    round(float(noisy[i, 0]), 2), round(float(noisy[i, 1]), 2), round(float(conf[i]), 3))
            if int(inside.sum()) < MIN_VISIBLE_KP:
                continue
            bbox = None
            if with_bbox:
                vis = px[inside]
                x1, y1 = vis.min(axis=0)
                x2, y2 = vis.max(axis=0)
                mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
                bbox = (round(max(x1 - mx, 0.0), 1), round(max(y1 - my, 0.0), 1),
                        round(min(x2 + mx, scene.width - 1.0), 1), round(min(y2 + my, scene.height - 1.0), 1))
            persons.append(Person(kp=kp, bbox=bbox))
        frames.append(Frame(t=round(t0 + t, 6), persons=persons))
    return frames


def _down_pose(b: Body, rng: np.random.Generator, x: float, facing: int, kind: str) -> Pose:
    """Lying pose at the end of a fall or lie-down."""
    if kind == "forward":
        th = rng.uniform(80, 90)
        return Pose(hx=x + facing * rng.uniform(0.5, 0.8), hy=rng.uniform(0.10, 0.18), theta=th,
                    aL=-th + rng.uniform(0, 10), aR=-th + rng.uniform(0, 10), kL=rng.uniform(0, 20),
                    kR=rng.uniform(0, 20), armL=rng.uniform(100, 150), armR=rng.uniform(100, 150), facing=facing)
    if kind == "backward":
        th = -rng.uniform(80, 90)
        return Pose(hx=x - facing * rng.uniform(0.3, 0.6), hy=rng.uniform(0.08, 0.15), theta=th,
                    aL=rng.uniform(70, 90), aR=rng.uniform(70, 90), kL=rng.uniform(10, 40),
                    kR=rng.uniform(10, 40), armL=rng.uniform(30, 90), armR=rng.uniform(30, 90), facing=facing)
    side = 1 if rng.random() < 0.5 else -1
    return Pose(hx=x + side * rng.uniform(0.3, 0.6), hy=rng.uniform(0.12, 0.20), psi=side * rng.uniform(80, 90),
                kL=rng.uniform(0, 30), kR=rng.uniform(0, 30), armL=rng.uniform(20, 90), armR=rng.uniform(20, 90),
                torso_scale=rng.uniform(0.85, 1.0), facing=facing, view="front", theta=0.0)


def _fall_like(b: Body, rng: np.random.Generator, x: float, facing: int, view: str, t_start: float,
               down: Pose, descend_s: float, theta_lag: float = 0.0) -> Callable[[float], Pose]:
    """Standing, then hip and trunk move to ``down`` over ``descend_s`` (trunk may lag)."""
    stand = standing(b, x, facing, view, rng)
    start = stand(t_start)

    def fn(t: float) -> Pose:
        if t < t_start:
            return stand(t)
        tau = (t - t_start) / descend_s
        p = lerp_pose(start, down, smooth(tau))
        s_rot = smooth((tau - theta_lag) / (1.0 - theta_lag)) if theta_lag > 0 else smooth(tau)
        return replace(p, theta=start.theta + (down.theta - start.theta) * s_rot,
                       psi=start.psi + (down.psi - start.psi) * s_rot)
    return fn


def _get_up(down_fn: Callable[[float], Pose], t_up: float, b: Body, x_end: float, facing: int, view: str,
            up_s: float, rng: np.random.Generator) -> Callable[[float], Pose]:
    """After ``t_up``, rise from the down pose through kneeling to standing (same view)."""
    lying = down_fn(t_up)
    if view == "front":
        kneel = Pose(hx=(lying.hx + x_end) / 2, hy=0.55, psi=math.copysign(20.0, lying.psi), kL=60.0, kR=60.0,
                     theta=0.0, facing=facing, view="front")
    else:
        kneel = Pose(hx=(lying.hx + x_end) / 2, hy=0.55, theta=25.0, aL=10.0, kL=90.0, aR=10.0, kR=90.0,
                     facing=facing)
    stand_fn = standing(b, x_end, facing, view, rng)
    rise = keyframed([(0.0, lying), (0.5 * up_s, kneel), (up_s, stand_fn(0.0))])
    return lambda t: down_fn(t) if t < t_up else (rise(t - t_up) if t < t_up + up_s else stand_fn(t - t_up - up_s))


def make_clip(kind: str, variant: str, seed: int, name: str) -> Clip:
    """Build one labelled clip of the given class/variant."""
    rng = np.random.default_rng(seed)
    body = Body().scaled(rng.uniform(0.95, 1.05))
    scene = Scene()
    facing = 1 if rng.random() < 0.5 else -1
    x = rng.uniform(-0.8, 0.8)
    t0 = round(float(rng.uniform(10.0, 1000.0)), 3)
    label: dict[str, Any] = {"class": kind, "variant": variant, "is_fall": False, "expected_event": None,
                             "t_fall_start": None, "px_per_m": scene.px_per_m}
    view = "side"

    if kind == "fall":
        fall_dir, recover = variant.split("_")
        view = "front" if fall_dir == "sideways" else "side"
        t_start = rng.uniform(2.5, 8.0)
        down = _down_pose(body, rng, x, facing, fall_dir)
        drop_L = (body.stand_hip - down.hy) / body.torso
        descend = drop_L / rng.uniform(2.0, 3.0)
        fn = _fall_like(body, rng, x, facing, view, t_start, down, descend,
                        theta_lag=0.1 if fall_dir == "backward" else 0.0)
        stay = rng.uniform(5.5, 8.0)
        duration = t_start + descend + stay + rng.uniform(0.5, 1.5)
        if recover == "recover":
            up_s = rng.uniform(2.0, 3.5)
            fn = _get_up(fn, t_start + descend + stay, body, down.hx, facing, view, up_s, rng)
            duration = t_start + descend + stay + up_s + rng.uniform(6.5, 8.0)
        label.update(is_fall=True, expected_event="FALL_CONFIRMED", t_fall_start=t0 + t_start,
                     descend_s=descend, drop_Ls=drop_L)
    elif kind == "lie_down_slow":
        t_start = rng.uniform(3.0, 6.0)
        D = rng.uniform(3.5, 4.5)
        down = _down_pose(body, rng, x, facing, "forward")
        stand = standing(body, x, facing, "side", rng)
        kneel = Pose(hx=x + facing * 0.1, hy=0.55, theta=10.0, aL=0.0, kL=90.0, aR=0.0, kR=90.0, facing=facing)
        quad = Pose(hx=x + facing * 0.25, hy=0.45, theta=70.0, aL=-10.0, kL=80.0, aR=-10.0, kR=80.0,
                    armL=80.0, armR=80.0, facing=facing)
        lower = keyframed([(0.0, stand(t_start)), (0.45 * D, kneel), (0.75 * D, quad), (D, down)])
        fn = then(stand, t_start, lower)
        if variant == "stay":
            duration = t_start + D + rng.uniform(12.0, 15.0)
            label.update(is_fall=True, expected_event="PROLONGED_DOWN", t_fall_start=t0 + t_start)
        else:
            lie = rng.uniform(4.0, 6.0)
            up_s = rng.uniform(2.5, 3.5)
            fn = _get_up(fn, t_start + D + lie, body, x, facing, "side", up_s, rng)
            duration = t_start + D + lie + up_s + 4.0
    elif kind == "lie_quick_recover":
        t_start = rng.uniform(3.0, 6.0)
        D = rng.uniform(1.0, 1.4)
        down = _down_pose(body, rng, x, facing, "forward")
        stand = standing(body, x, facing, "side", rng)
        fn = then(stand, t_start, keyframed([(0.0, stand(t_start)), (D, down)]))
        lie = rng.uniform(0.3, 0.8)
        fn = _get_up(fn, t_start + D + lie, body, x, facing, "side", rng.uniform(0.9, 1.2), rng)
        duration = t_start + D + lie + 1.2 + rng.uniform(5.0, 7.0)
    elif kind == "sit_fast":
        t_start = rng.uniform(3.0, 6.0)
        D = rng.uniform(0.6, 1.0)
        drop = rng.uniform(0.4, 0.8) * body.torso
        stand = standing(body, x, facing, "side", rng)
        mid = Pose(hx=x - facing * 0.08, hy=body.stand_hip - 0.6 * drop, theta=rng.uniform(10, 18),
                   aL=45.0, kL=45.0, aR=45.0, kR=45.0, facing=facing)
        seated = Pose(hx=x - facing * 0.15, hy=body.stand_hip - drop, theta=rng.uniform(3, 10),
                      aL=85.0, kL=85.0, aR=85.0, kR=85.0, armL=30.0, armR=30.0, facing=facing)
        fn = then(stand, t_start, keyframed([(0.0, stand(t_start)), (0.5 * D, mid), (D, seated)]))
        duration = t_start + D + rng.uniform(4.0, 8.0)
    elif kind == "bend":
        t_start = rng.uniform(3.0, 6.0)
        D, hold, back = rng.uniform(1.0, 1.6), rng.uniform(1.0, 3.0), rng.uniform(1.0, 1.5)
        stand = standing(body, x, facing, "side", rng)
        knees = rng.uniform(10, 30)
        bent = Pose(hx=x - facing * 0.1, hy=body.stand_hip - rng.uniform(0.05, 0.15), theta=rng.uniform(60, 80),
                    aL=knees / 2, kL=knees, aR=knees / 2, kR=knees, armL=60.0, armR=60.0, facing=facing)
        fn = then(stand, t_start, keyframed([(0.0, stand(t_start)), (D, bent), (D + hold, bent),
                                             (D + hold + back, stand(0.0))]))
        duration = t_start + D + hold + back + rng.uniform(3.0, 5.0)
    elif kind in ("walk", "leave_frame"):
        g = Gait(speed=rng.uniform(0.8, 1.3), phase0=rng.uniform(0, 1.2))
        g.T_L = g.T_R = 60.0 / rng.uniform(90, 115)
        span = scene.width / scene.px_per_m
        x_start = -facing * (span / 2 - 0.6)
        if kind == "walk" and variant == "turn":
            turn_at = (span - 1.6) / g.speed * rng.uniform(0.6, 0.9)
            fn = walking(body, g, x_start, facing, turn_at=turn_at)
            duration = turn_at + 1.2 + rng.uniform(2.0, 3.0)
        elif kind == "walk":
            fn = walking(body, g, x_start, facing)
            duration = (span - 1.4) / g.speed
        else:
            fn = walking(body, g, -facing * 0.5, facing)
            duration = (span / 2 + 1.5) / g.speed + rng.uniform(2.0, 4.0)
        label.update(speed_mps=g.speed, cadence_spm=g.cadence)
    else:  # standing
        fn = standing(body, x, facing, "side", rng)
        duration = rng.uniform(10.0, 15.0)

    bystander = None
    if rng.random() < 0.1:
        bb = body.scaled(BYSTANDER_SCALE)
        bx = -math.copysign(1.0, x) * rng.uniform(1.5, 2.0)
        bystander = (standing(bb, bx, 1, "side", rng), bb)
    frames = render(fn, body, scene, duration, rng, t0, with_bbox=bool(rng.random() < 0.5), bystander=bystander)
    label.update(t0=t0, duration_s=round(duration, 3), view=view, bystander=bystander is not None, seed=seed)
    return Clip(name=name, frames=frames, label=label)


def make_walk(seed: int, speed: float = 1.0, cadence: float = 100.0, left_slower: float = 0.0,
              px_per_m: float = 160.0, duration: float = 8.0, turn_at: float | None = None) -> Clip:
    """Gait test clip: steady side-view walk with known speed, cadence and asymmetry.

    ``left_slower`` = 0.15 makes left-led step times 15 % longer than right-led.
    The image is made wide enough to keep the whole walk in view.
    """
    rng = np.random.default_rng(seed)
    T = 60.0 / cadence
    T_R = 2 * T / (2.0 + left_slower)
    g = Gait(speed=speed, T_L=T_R * (1 + left_slower), T_R=T_R, phase0=float(rng.uniform(0, 1)))
    body = Body()
    path_m = speed * (turn_at if turn_at is not None else duration) + 2.0
    width = int(path_m * px_per_m + 200)
    scene = Scene(px_per_m=px_per_m, width=width, height=int(2.3 * px_per_m) + 60,
                  floor_px=int(2.3 * px_per_m) + 20, x_origin_px=100 + px_per_m)
    fn = walking(body, g, 0.0, 1, turn_at=turn_at)
    frames = render(fn, body, scene, duration, rng, t0=100.0, with_bbox=True)
    si = 100 * abs(g.T_L - g.T_R) / (0.5 * (g.T_L + g.T_R))
    label = {"class": "walk", "variant": "gait", "speed_mps": speed, "cadence_spm": g.cadence, "si_pct": si,
             "T_L": g.T_L, "T_R": g.T_R, "px_per_m": px_per_m, "marker_px": [100 + px_per_m, 100 + 4 * px_per_m],
             "marker_m": 3.0, "is_fall": False, "expected_event": None, "t_fall_start": None}
    return Clip(name=f"walk_gait_{seed}", frames=frames, label=label)


CHAIR_HIP_M = 0.48  # hip height when seated


def _crossings(t: np.ndarray, y: np.ndarray, level: float) -> list[tuple[float, int]]:
    """(time, +1 rising / -1 falling) where ``y`` crosses ``level``."""
    out = []
    for i in range(len(t) - 1):
        if (y[i] - level) * (y[i + 1] - level) < 0:
            tc = t[i] + (level - y[i]) / (y[i + 1] - y[i]) * (t[i + 1] - t[i])
            out.append((float(tc), 1 if y[i + 1] > y[i] else -1))
    return out


def make_tug(seed: int, px_per_m: float = 160.0, speed: float | None = None) -> Clip:
    """Timed Up and Go clip: sit, stand up, walk 3 m, turn, walk back, turn, sit down.

    Ground truth uses the same definition as ``logic.tug`` on the noise-free
    hip height: stand-up from y_n 0.1 to 0.9, TUG from stand-up start (y_n
    rising through 0.1) to sit-down end (y_n falling through 0.1).
    """
    rng = np.random.default_rng(seed)
    body = Body().scaled(rng.uniform(0.95, 1.05))
    v = speed if speed is not None else rng.uniform(0.6, 1.2)
    g = Gait(speed=v, phase0=float(rng.uniform(0, 1.2)))
    g.T_L = g.T_R = 60.0 / rng.uniform(85, 110)
    sit0, sts, turn_s = rng.uniform(2.0, 3.0), rng.uniform(1.0, 1.8), rng.uniform(1.0, 2.0)
    out_s, turn2, sit_s = 3.0 / v, rng.uniform(0.8, 1.2), rng.uniform(1.2, 2.0)
    x_up = 0.15
    seated = Pose(hx=0.0, hy=CHAIR_HIP_M, theta=5.0, aL=85.0, kL=85.0, aR=85.0, kR=85.0, armL=30.0, armR=30.0)
    walk = walking(body, g, x_up, 1, turn_at=out_s, turn_s=turn_s)
    walk_total = 2 * out_s + turn_s
    mid_hy = CHAIR_HIP_M + 0.45 * (body.stand_hip - CHAIR_HIP_M)
    stand_up = keyframed([(0.0, seated), (0.5 * sts, Pose(hx=0.08, hy=mid_hy, theta=35.0, aL=40.0, kL=50.0,
                                                          aR=40.0, kR=50.0)), (sts, walk(0.0))])
    upright_back = Pose(hx=x_up, hy=body.stand_hip, theta=3.0, facing=-1)
    upright_fwd = replace(upright_back, facing=1)
    turn_to_sit = keyframed([(0.0, walk(walk_total)), (0.5 * turn2, upright_back)])
    sit_down = keyframed([(0.0, upright_fwd), (0.5 * sit_s, Pose(hx=0.05, hy=mid_hy, theta=35.0, aL=40.0,
                                                                 kL=50.0, aR=40.0, kR=50.0)), (sit_s, seated)])
    T2, T3 = sit0 + sts, sit0 + sts + walk_total
    T4, T5 = T3 + turn2, T3 + turn2 + sit_s

    def fn(t: float) -> Pose:
        if t < sit0:
            return seated
        if t < T2:
            return stand_up(t - sit0)
        if t < T3:
            return walk(t - T2)
        if t < T3 + 0.5 * turn2:
            return turn_to_sit(t - T3)
        if t < T4:
            return upright_fwd
        if t < T5:
            return sit_down(t - T4)
        return seated

    duration = T5 + 2.0
    reach = x_up + v * out_s + v * turn_s / math.pi
    scene = Scene(px_per_m=px_per_m, width=int((reach + 1.6) * px_per_m), height=int(2.3 * px_per_m) + 60,
                  floor_px=int(2.3 * px_per_m) + 20, x_origin_px=0.7 * px_per_m)
    t0 = 100.0
    frames = render(fn, body, scene, duration, rng, t0, with_bbox=True)
    fine = np.arange(0.0, duration, 0.002)
    yn = (np.array([fn(x).hy for x in fine]) - CHAIR_HIP_M) / (body.stand_hip - CHAIR_HIP_M)
    lo, hi = _crossings(fine, yn, 0.1), _crossings(fine, yn, 0.9)
    t_a = next(tc for tc, s in lo if s > 0)
    t_b = next(tc for tc, s in hi if s > 0)
    t_d = [tc for tc, s in lo if s < 0][-1]
    label = {"class": "tug", "variant": "standard", "is_fall": False, "expected_event": None, "t_fall_start": None,
             "speed_mps": v, "px_per_m": px_per_m, "tug_s": t_d - t_a, "t_sts": t_b - t_a,
             "t_standup_start": t0 + t_a, "t_turn_mid": t0 + T2 + out_s + 0.5 * turn_s}
    return Clip(name=f"tug_{seed}", frames=frames, label=label)


# ---------------------------------------------------------------- dataset

CLASS_MIX: list[tuple[str, str, float]] = [
    ("fall", "forward_stay", 0.075), ("fall", "forward_recover", 0.075),
    ("fall", "backward_stay", 0.075), ("fall", "backward_recover", 0.075),
    ("fall", "sideways_stay", 0.075), ("fall", "sideways_recover", 0.075),
    ("lie_down_slow", "stay", 0.05),
    ("standing", "quiet", 0.05), ("walk", "straight", 0.05), ("walk", "turn", 0.03),
    ("sit_fast", "chair", 0.10), ("bend", "pickup", 0.09), ("lie_down_slow", "brief", 0.05),
    ("lie_quick_recover", "quick", 0.08), ("leave_frame", "walk_out", 0.05),
]


def dataset_plan(n: int, seed: int) -> list[tuple[str, str, int]]:
    """Deterministic list of (class, variant, clip seed) of length ``n``."""
    counts = [max(1, round(frac * n)) for _, _, frac in CLASS_MIX]
    while sum(counts) > n:
        counts[int(np.argmax(counts))] -= 1
    while sum(counts) < n:
        counts[int(np.argmin(counts))] += 1
    plan = [(k, v) for (k, v, _), c in zip(CLASS_MIX, counts) for _ in range(c)]
    order = np.random.default_rng(seed).permutation(len(plan))
    return [(plan[i][0], plan[i][1], seed * 100_003 + j) for j, i in enumerate(order)]


def generate_dataset(n: int, seed: int) -> list[Clip]:
    """Generate ``n`` labelled clips."""
    return [make_clip(kind, variant, s, f"{j:04d}_{kind}_{variant}")
            for j, (kind, variant, s) in enumerate(dataset_plan(n, seed))]


def save_dataset(clips: list[Clip], out_dir: str | Path) -> None:
    """Write ``<name>.jsonl`` per clip plus ``labels.json``."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    labels = {}
    for c in clips:
        write_jsonl(d / f"{c.name}.jsonl", c.frames)
        labels[c.name] = c.label
    with open(d / "labels.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(labels, fh, indent=1, sort_keys=True)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m logic.synth", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    clips = generate_dataset(args.n, args.seed)
    save_dataset(clips, args.out)
    n_fall = sum(1 for c in clips if c.label["expected_event"])
    print(f"wrote {len(clips)} clips ({n_fall} expecting an alert) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
