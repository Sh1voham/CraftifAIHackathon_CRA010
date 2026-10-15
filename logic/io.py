"""Frame I/O: replay JSONL, live transports and person selection.

Every assumption about PipeGen's output format lives in ``_from_pipegen`` and
``PIPEGEN_NAME_MAP``. The transports below (``tail_jsonl``, ``tail_folder``,
``mqtt_objects``) only move JSON objects around and know nothing about fields.
"""
from __future__ import annotations

import json
import logging
import math
import queue
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

logger = logging.getLogger(__name__)

KEYPOINT_NAMES: tuple[str, ...] = (
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
)
KP_INDEX: dict[str, int] = {n: i for i, n in enumerate(KEYPOINT_NAMES)}

Keypoint = tuple[float, float, float]  # (x, y, confidence), pixels, y downward
BBox = tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass
class Person:
    """One detected person: named keypoints plus optional id and box."""

    kp: dict[str, Keypoint] = field(default_factory=dict)
    id: int | None = None
    bbox: BBox | None = None


@dataclass
class Frame:
    """All persons seen at timestamp ``t`` (seconds)."""

    t: float
    persons: list[Person] = field(default_factory=list)


# ---------------------------------------------------------------- parsing

def _parse_kp(raw: Any, names: dict[str, str] | None = None) -> dict[str, Keypoint]:
    """Accept {name: [x, y, c]}, a list of 17 triples or a flat list of 51 numbers."""
    out: dict[str, Keypoint] = {}
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list):
        if len(raw) == 3 * len(KEYPOINT_NAMES) and all(isinstance(v, (int, float)) for v in raw):
            raw = [raw[i:i + 3] for i in range(0, len(raw), 3)]
        items = zip(KEYPOINT_NAMES, raw)
    else:
        raise ValueError(f"unsupported keypoint container {type(raw).__name__}")
    for name, val in items:
        if names is not None:
            name = names.get(name, name)
        if name not in KP_INDEX or val is None:
            continue
        conf = float(val[2]) if len(val) > 2 else 1.0
        out[name] = (float(val[0]), float(val[1]), conf)
    return out


def _person_from_dict(obj: dict[str, Any], names: dict[str, str] | None = None) -> Person:
    bbox = obj.get("bbox")
    pid = obj.get("id")
    return Person(
        kp=_parse_kp(obj.get("kp", {}), names),
        id=int(pid) if pid is not None else None,
        bbox=tuple(float(v) for v in bbox) if bbox is not None else None,  # type: ignore[arg-type]
    )


def frame_from_dict(obj: dict[str, Any], names: dict[str, str] | None = None) -> Frame:
    """Parse one replay-format frame object."""
    return Frame(t=float(obj["t"]), persons=[_person_from_dict(p, names) for p in obj.get("persons", [])])


def frame_to_dict(frame: Frame) -> dict[str, Any]:
    """Replay-format dict for ``frame``."""
    persons = []
    for p in frame.persons:
        d: dict[str, Any] = {}
        if p.id is not None:
            d["id"] = p.id
        if p.bbox is not None:
            d["bbox"] = list(p.bbox)
        d["kp"] = {n: list(p.kp[n]) for n in KEYPOINT_NAMES if n in p.kp}
        persons.append(d)
    return {"t": frame.t, "persons": persons}


# ---------------------------------------------------------------- replay files

def _iter_json_lines(stream: Iterable[str], source: str) -> Iterator[Any]:
    for lineno, line in enumerate(stream, 1):
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("%s:%d: skipping malformed JSON line (%s)", source, lineno, exc)


def _stdin_lines() -> Iterator[str]:
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        yield line


def _frames(objs: Iterable[Any], parse: Any, source: str) -> Iterator[Frame]:
    for obj in objs:
        try:
            frame = parse(obj)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            logger.warning("%s: skipping unparseable frame (%r)", source, exc)
            continue
        if frame is not None:
            yield frame


def read_jsonl(path: str | Path) -> Iterator[Frame]:
    """Stream frames from a replay JSONL file, or from stdin when ``path`` is '-'."""
    if str(path) == "-":
        yield from _frames(_iter_json_lines(_stdin_lines(), "<stdin>"), frame_from_dict, "<stdin>")
        return
    with open(path, encoding="utf-8") as fh:
        yield from _frames(_iter_json_lines(fh, str(path)), frame_from_dict, str(path))


def write_jsonl(path: str | Path, frames: Iterable[Frame]) -> None:
    """Write frames in replay format (LF line endings on every platform)."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        write_frames(fh, frames)


def write_frames(fh: TextIO, frames: Iterable[Frame]) -> None:
    """Write frames as compact JSON lines to an open text stream."""
    for frame in frames:
        fh.write(json.dumps(frame_to_dict(frame), separators=(",", ":")) + "\n")


# ---------------------------------------------------------------- live transports

def tail_jsonl(path: str | Path, poll_s: float = 0.05, idle_timeout_s: float | None = None) -> Iterator[Any]:
    """Follow a growing JSONL file (like ``tail -f``) and yield each complete object."""
    p = Path(path)
    last_data = time.monotonic()
    while not p.exists():
        if idle_timeout_s is not None and time.monotonic() - last_data > idle_timeout_s:
            return
        time.sleep(poll_s)
    with open(p, encoding="utf-8") as fh:
        pending = ""
        lineno = 0
        while True:
            chunk = fh.readline()
            if chunk:
                last_data = time.monotonic()
                pending += chunk
                if not pending.endswith("\n"):
                    continue  # partial line; wait for the writer to finish it
                lineno += 1
                line, pending = pending.strip(), ""
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        logger.warning("%s:%d: skipping malformed JSON line (%s)", p, lineno, exc)
                continue
            if idle_timeout_s is not None and time.monotonic() - last_data > idle_timeout_s:
                return
            time.sleep(poll_s)


def tail_folder(folder: str | Path, pattern: str = "*.json", poll_s: float = 0.05,
                idle_timeout_s: float | None = None, settle_s: float = 1.0) -> Iterator[Any]:
    """Yield JSON objects from per-frame files in ``folder``, in filename order.

    A file that does not parse yet is retried on later polls (it may still be
    being written); it is skipped with a warning once its size has not changed
    for ``settle_s``. A file holding a JSON list yields each element.
    """
    d = Path(folder)
    done: set[str] = set()
    failed: dict[str, tuple[int, float]] = {}  # name -> (size, first failure at that size)
    last_data = time.monotonic()
    while True:
        progressed = False
        for f in sorted(d.glob(pattern)):
            if f.name in done:
                continue
            try:
                size = f.stat().st_size
                obj = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                now = time.monotonic()
                prev = failed.get(f.name)
                if prev is None or prev[0] != size:
                    failed[f.name] = (size, now)
                elif now - prev[1] >= settle_s:
                    logger.warning("%s: skipping unreadable file (%s)", f, exc)
                    done.add(f.name)
                break  # keep frame order: do not jump past an unfinished file
            done.add(f.name)
            progressed = True
            if isinstance(obj, list):
                yield from obj
            else:
                yield obj
        if progressed:
            last_data = time.monotonic()
            continue
        if idle_timeout_s is not None and time.monotonic() - last_data > idle_timeout_s:
            return
        time.sleep(poll_s)


def mqtt_objects(host: str, port: int, topic: str, *, keepalive: int = 30, poll_s: float = 0.05,
                 idle_timeout_s: float | None = None, client: Any = None) -> Iterator[Any]:
    """Subscribe to ``topic`` and yield each message payload parsed as JSON."""
    if client is None:
        from logic.mqtt_out import make_client

        client = make_client()
    inbox: queue.Queue[bytes] = queue.Queue()

    def on_connect(c: Any, _userdata: Any, *_args: Any) -> None:
        c.subscribe(topic, qos=1)

    def on_message(_c: Any, _userdata: Any, msg: Any) -> None:
        inbox.put(msg.payload)

    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(host, port, keepalive)
    client.loop_start()
    try:
        last_data = time.monotonic()
        while True:
            try:
                payload = inbox.get(timeout=poll_s)
            except queue.Empty:
                if idle_timeout_s is not None and time.monotonic() - last_data > idle_timeout_s:
                    return
                continue
            last_data = time.monotonic()
            try:
                obj = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning("mqtt %s: skipping malformed payload (%s)", topic, exc)
                continue
            if isinstance(obj, list):
                yield from obj
            else:
                yield obj
    finally:
        client.loop_stop()
        client.disconnect()


# ---------------------------------------------------------------- PipeGen adapter

# Map from PipeGen keypoint names to the canonical COCO-17 names above.
# VERIFY against real PipeGen output; identity until then.
PIPEGEN_NAME_MAP: dict[str, str] = {}


def _from_pipegen(obj: Any) -> Frame | None:
    """Convert one PipeGen output object into a Frame.

    VERIFY at the real-data handoff. Today PipeGen output is assumed to be the
    replay format already: {"t": seconds, "persons": [{"id", "bbox", "kp"}]}
    with pixel coordinates. Field renames, unit conversions (e.g. PTS in ns,
    normalised coordinates) and per-object to per-frame grouping belong here.
    """
    return frame_from_dict(obj, PIPEGEN_NAME_MAP or None)


def read_keypoints(source: str | Path, *, follow: bool = False, poll_s: float = 0.05,
                   idle_timeout_s: float | None = None, settle_s: float = 1.0) -> Iterator[Frame]:
    """Yield frames from a PipeGen source.

    ``source`` may be '-' (stdin JSONL), a JSONL file (``follow=True`` tails it),
    a folder of per-frame JSON files, or ``mqtt://host:port/topic``.
    """
    src = str(source)
    if src == "-":
        objs: Iterable[Any] = _iter_json_lines(_stdin_lines(), "<stdin>")
    elif src.startswith("mqtt://"):
        hostport, _, topic = src[len("mqtt://"):].partition("/")
        host, _, port = hostport.partition(":")
        objs = mqtt_objects(host, int(port or 1883), topic or "#", poll_s=poll_s,
                            idle_timeout_s=idle_timeout_s)
    elif Path(src).is_dir():
        objs = tail_folder(src, poll_s=poll_s, idle_timeout_s=idle_timeout_s, settle_s=settle_s)
    elif follow:
        objs = tail_jsonl(src, poll_s=poll_s, idle_timeout_s=idle_timeout_s)
    else:
        def _file() -> Iterator[Any]:
            with open(src, encoding="utf-8") as fh:
                yield from _iter_json_lines(fh, src)
        objs = _file()
    yield from _frames(objs, _from_pipegen, src)


# ---------------------------------------------------------------- person selection

@dataclass
class TrackState:
    """Which person we are following, across frames."""

    c_min: float = 0.30
    lost_grace_s: float = 2.0
    anchor: tuple[float, float] | None = None
    last_t: float | None = None


def _valid(p: Person, name: str, c_min: float) -> Keypoint | None:
    k = p.kp.get(name)
    return k if k is not None and k[2] >= c_min else None


def _anchor(p: Person, c_min: float) -> tuple[float, float] | None:
    """Mid-hip, else bbox centre, else mean of valid keypoints."""
    lh, rh = _valid(p, "l_hip", c_min), _valid(p, "r_hip", c_min)
    if lh and rh:
        return ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
    if p.bbox is not None:
        return ((p.bbox[0] + p.bbox[2]) / 2, (p.bbox[1] + p.bbox[3]) / 2)
    pts = [k for k in p.kp.values() if k[2] >= c_min]
    if pts:
        return (sum(k[0] for k in pts) / len(pts), sum(k[1] for k in pts) / len(pts))
    return None


def _size_score(p: Person, c_min: float) -> tuple[float, int]:
    """(torso length x mean confidence, number of valid keypoints)."""
    pts = [_valid(p, n, c_min) for n in ("l_shoulder", "r_shoulder", "l_hip", "r_hip")]
    n_valid = sum(1 for k in p.kp.values() if k[2] >= c_min)
    mean_conf = sum(p.kp[n][2] for n in p.kp) / len(KEYPOINT_NAMES)
    if all(pts):
        ls, rs, lh, rh = pts  # type: ignore[misc]
        sx, sy = (ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2
        hx, hy = (lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2
        return (math.hypot(sx - hx, sy - hy) * mean_conf, n_valid)
    return (0.0, n_valid)


def select_person(frame: Frame, state: TrackState) -> Person | None:
    """Pick the person to follow in ``frame`` and update ``state``.

    Keeps the person nearest (by mid-hip) to the last chosen one. On the first
    frame, or after losing the person for more than ``lost_grace_s``, picks the
    one with the largest torso length times mean confidence.
    """
    candidates = [(p, _anchor(p, state.c_min)) for p in frame.persons]
    candidates = [(p, a) for p, a in candidates if a is not None]
    if not candidates:
        return None
    fresh = state.anchor is None or state.last_t is None or frame.t - state.last_t > state.lost_grace_s
    if fresh:
        chosen, anchor = max(candidates, key=lambda pa: _size_score(pa[0], state.c_min))
    else:
        ax, ay = state.anchor  # type: ignore[misc]
        chosen, anchor = min(candidates, key=lambda pa: math.hypot(pa[1][0] - ax, pa[1][1] - ay))
    state.anchor, state.last_t = anchor, frame.t
    return chosen
