"""Small scripted clips for end-to-end tests (independent of logic.synth)."""
from __future__ import annotations

import math

from logic.io import Frame, Person

FLOOR_Y = 1000.0
TORSO = 200.0


def pose(hip_x: float, hip_y: float, phi_deg: float, conf: float = 0.9) -> Person:
    """Crude side-view stick figure: torso tilted ``phi_deg`` towards +x, legs to the floor."""
    a = math.radians(phi_deg)
    ux, uy = math.sin(a), -math.cos(a)
    sx, sy = hip_x + TORSO * ux, hip_y + TORSO * uy
    ankle_y = min(FLOOR_Y - 26.0, hip_y + 340.0)
    ankle_x = hip_x - max(0.0, 340.0 - (ankle_y - hip_y))
    kp = {
        "nose": (sx + 80.0 * ux, sy + 80.0 * uy, conf),
        "l_shoulder": (sx - 4.0, sy, conf), "r_shoulder": (sx + 4.0, sy, conf),
        "l_hip": (hip_x - 4.0, hip_y, conf), "r_hip": (hip_x + 4.0, hip_y, conf),
        "l_knee": ((hip_x + ankle_x) / 2, (hip_y + ankle_y) / 2, conf),
        "r_knee": ((hip_x + ankle_x) / 2 + 6, (hip_y + ankle_y) / 2, conf),
        "l_ankle": (ankle_x, ankle_y, conf), "r_ankle": (ankle_x + 10.0, ankle_y, conf),
    }
    return Person(kp=kp, bbox=None)


def scripted_fall(t0: float = 50.0, fps: float = 30.0, stand_s: float = 6.0, drop_s: float = 0.5,
                  lie_s: float = 5.0) -> list[Frame]:
    """Stand, fall fast (hip 1.8 L -> 0.2 L above the ankle line, torso to 85 deg), lie still."""
    hip_stand, hip_lie = FLOOR_Y - 26.0 - 1.7 * TORSO, FLOOR_Y - 26.0 - 0.2 * TORSO
    frames, n = [], int((stand_s + drop_s + lie_s) * fps)
    for i in range(n):
        t = i / fps
        if t < stand_s:
            hy, phi = hip_stand, 3.0
        elif t < stand_s + drop_s:
            s = (t - stand_s) / drop_s
            s = s * s * (3 - 2 * s)
            hy, phi = hip_stand + (hip_lie - hip_stand) * s, 3.0 + 82.0 * s
        else:
            hy, phi = hip_lie, 85.0
        frames.append(Frame(t=round(t0 + t, 6), persons=[pose(400.0, hy, phi)]))
    return frames


def scripted_sit(t0: float = 50.0, fps: float = 30.0) -> list[Frame]:
    """Stand 6 s, sit down quickly (hip drops 0.6 L in 0.5 s, torso upright), stay seated."""
    hip_stand = FLOOR_Y - 26.0 - 1.7 * TORSO
    frames = []
    for i in range(int(12 * fps)):
        t = i / fps
        s = min(max((t - 6.0) / 0.5, 0.0), 1.0)
        s = s * s * (3 - 2 * s)
        frames.append(Frame(t=round(t0 + t, 6), persons=[pose(400.0, hip_stand + 0.6 * TORSO * s, 8.0)]))
    return frames
