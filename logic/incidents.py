"""Incident log (JSON Lines) and the frame-time to wall-clock mapping.

This is the output boundary: the only place outside ``run.py`` where wall-clock
time is allowed.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from logic.fall import Event


class WallClock:
    """Turns frame timestamps into ISO-8601 wall-clock strings.

    With a ``start`` time, the first frame maps to ``start`` and later frames to
    ``start + (t - t_first)``, which makes replays reproducible. Without one,
    the current local time is used (live mode).
    """

    def __init__(self, start: datetime | None = None) -> None:
        self.start = start
        self._t0: float | None = None

    def observe(self, t: float) -> None:
        """Note a frame timestamp (the first one anchors replay time)."""
        if self._t0 is None:
            self._t0 = t

    def iso(self, t: float) -> str:
        """Wall-clock time of frame ``t``, with milliseconds and UTC offset."""
        if self.start is None:
            when = datetime.now().astimezone()
        else:
            when = self.start + timedelta(seconds=t - (self._t0 if self._t0 is not None else t))
        return when.isoformat(timespec="milliseconds")


def event_record(ev: Event, ts: str, camera: str, model: str, frame_saved: str | None) -> dict[str, Any]:
    """Incident record in the documented schema (plus the frame time ``t_s``)."""
    return {
        "ts": ts,
        "event": ev.kind,
        "confidence": round(ev.confidence, 3),
        "torso_angle_deg": round(ev.torso_angle_deg, 1),
        "hip_drop_speed_Lps": round(ev.hip_drop_speed_Lps, 2),
        "time_down_s": round(ev.time_down_s, 2),
        "camera": camera,
        "model": model,
        "frame_saved": frame_saved,
        "t_s": round(ev.ts_s, 3),
    }


class IncidentLog:
    """Append-only JSONL log; every record is flushed and fsynced."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8", newline="\n")

    def append(self, record: dict[str, Any]) -> None:
        """Write one record durably."""
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        """Close the file."""
        self._fh.close()
