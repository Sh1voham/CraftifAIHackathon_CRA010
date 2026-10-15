"""Frame I/O: replay JSONL, live transports, format adaptation and person selection.

Every assumption about the vision side's output format lives in this module,
in ``auto_adapt``/``adapt_stream`` (layout, timestamp unit, coordinate mode,
keypoint names). The raw transports (``tail_objects``, ``folder_objects``,
``mqtt_objects``) only move JSON objects; ``tail_jsonl``, ``watch_folder``,
``mqtt_frames`` and ``read_keypoints`` yield adapted Frames.
"""
from __future__ import annotations

import json
import logging
import math
import queue
import re
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

import numpy as np

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


# ---------------------------------------------------------------- live transports (raw JSON objects)

def _split_payload(obj: Any) -> Iterator[Any]:
    """A JSON list of frame objects yields each frame; anything else is one object."""
    if isinstance(obj, list) and obj and all(
            isinstance(o, dict) and any(k in o for k in TIME_KEYS + FRAME_KEYS + PERSON_LIST_KEYS) for o in obj):
        yield from obj
    else:
        yield obj


def tail_objects(path: str | Path, poll_s: float = 0.05, idle_timeout_s: float | None = None) -> Iterator[Any]:
    """Follow a growing JSONL file (like ``tail -f``) and yield each complete JSON object.

    Partial lines wait for their newline; if the file shrinks (truncated or
    rewritten) reading restarts from the beginning.
    """
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
                        yield from _split_payload(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.warning("%s:%d: skipping malformed JSON line (%s)", p, lineno, exc)
                continue
            try:
                if p.stat().st_size < fh.tell():
                    logger.warning("%s: file was truncated; reading again from the start", p)
                    fh.seek(0)
                    pending, lineno = "", 0
                    continue
            except OSError:
                pass  # file briefly missing while being replaced
            if idle_timeout_s is not None and time.monotonic() - last_data > idle_timeout_s:
                return
            time.sleep(poll_s)


def folder_objects(folder: str | Path, pattern: str = "*.json", poll_s: float = 0.1,
                   idle_timeout_s: float | None = None, settle_s: float = 1.0) -> Iterator[Any]:
    """Yield the JSON object of each new per-frame file in ``folder`` once, in filename order.

    A file that does not parse yet is retried on later polls (it may still be
    being written); it is skipped with a warning once its size has not changed
    for ``settle_s``.
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
            yield from _split_payload(obj)
        if progressed:
            last_data = time.monotonic()
            continue
        if idle_timeout_s is not None and time.monotonic() - last_data > idle_timeout_s:
            return
        time.sleep(poll_s)


def mqtt_objects(host: str, port: int, topic: str, *, keepalive: int = 30, poll_s: float = 0.05,
                 idle_timeout_s: float | None = None, client: Any = None,
                 reconnect_min_s: int = 1, reconnect_max_s: int = 30) -> Iterator[Any]:
    """Subscribe to ``topic`` and yield each message payload parsed as JSON.

    paho's network thread reconnects with backoff (and re-subscribes on every
    connect); a broker problem is logged and never raised to the consumer.
    """
    if client is None:
        from logic.mqtt_out import make_client

        client = make_client()
    inbox: queue.Queue[bytes] = queue.Queue()

    def on_connect(c: Any, _userdata: Any, *_args: Any) -> None:
        c.subscribe(topic, qos=1)
        logger.info("mqtt input: subscribed to %s on %s:%s", topic, host, port)

    def on_disconnect(*_args: Any) -> None:
        logger.warning("mqtt input: disconnected from %s:%s; reconnecting in the background", host, port)

    def on_message(_c: Any, _userdata: Any, msg: Any) -> None:
        inbox.put(msg.payload)

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    try:
        if hasattr(client, "reconnect_delay_set"):
            client.reconnect_delay_set(reconnect_min_s, reconnect_max_s)
        client.connect_async(host, port, keepalive)
        client.loop_start()
    except Exception as exc:  # noqa: BLE001 - keep the consumer alive
        logger.warning("mqtt input: cannot start client for %s:%s (%r)", host, port, exc)
        return
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
            yield from _split_payload(obj)
    finally:
        client.loop_stop()
        client.disconnect()


# ---------------------------------------------------------------- auto_adapt: every vision-side format assumption

PERSON_LIST_KEYS = ("persons", "objects", "detections", "people", "instances", "poses")
KP_KEYS = ("kp", "keypoints", "landmarks", "kps", "joints")
TIME_KEYS = ("t", "ts", "timestamp", "pts", "time")
FRAME_KEYS = ("frame_num", "frame_id", "frame_number", "frame_idx", "frame")
ID_KEYS = ("id", "track_id", "object_id", "trackId")
BBOX_KEYS = ("bbox", "box", "rect", "rect_params")
SCORE_KEYS = ("score", "scores", "conf", "confidence", "c", "visibility", "prob")
SIZE_KEYS = (("frame_width", "frame_height"), ("width", "height"), ("source_width", "source_height"))
UNIT_SCALE = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}


@dataclass
class AdaptOpts:
    """How to interpret raw vision-side JSON (see ``add_adapter_args``)."""

    fps: float = 30.0  # only used when frames carry a frame number instead of a timestamp
    ts_unit: str = "auto"  # auto | s | ms | us | ns
    coords: str = "auto"  # auto | px | norm
    frame_size: tuple[int, int] | None = None  # (width, height), needed for normalised coordinates
    kp_map: dict[str, str] | None = None  # source index or name -> canonical COCO-17 name
    normalise_t: bool = True  # shift time so the stream starts near zero
    probe_frames: int = 10  # frames inspected to infer "auto" settings
    frame_dt_min_s: float = 0.005  # plausible frame interval range for unit inference
    frame_dt_max_s: float = 0.5
    norm_coord_max: float = 1.5  # all coordinates within [0, this] -> normalised


@dataclass
class _Parsed:
    raw_t: float | None
    time_field: str
    persons: list[tuple[dict[str, tuple[float, float, float]], Any, int | None]]
    layout: str
    frame_size: tuple[int, int] | None


def _canon_name(name: str) -> str:
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", str(name)).lower().replace("-", "_").replace(" ", "_")
    s = re.sub(r"^left_?", "l_", s)
    return re.sub(r"^right_?", "r_", s)


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _kp_value(v: Any) -> tuple[float, float, float] | None:
    """[x, y, c] / [x, y] / {"x", "y", "score"} -> (x, y, c)."""
    if isinstance(v, dict):
        x, y = _num(v.get("x")), _num(v.get("y"))
        c = next((_num(v[k]) for k in SCORE_KEYS if k in v), 1.0)
        return None if x is None or y is None else (x, y, 1.0 if c is None else c)
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        x, y = _num(v[0]), _num(v[1])
        c = _num(v[2]) if len(v) > 2 else 1.0
        return None if x is None or y is None else (x, y, 1.0 if c is None else c)
    return None


def _name_for(key: Any, opts: AdaptOpts) -> str | None:
    if opts.kp_map is not None and str(key) in opts.kp_map:
        name = opts.kp_map[str(key)]
    elif isinstance(key, int):
        name = KEYPOINT_NAMES[key] if key < len(KEYPOINT_NAMES) and opts.kp_map is None else ""
    else:
        name = _canon_name(key)
    return name if name in KP_INDEX else None


def _parse_keypoints(person: dict[str, Any], opts: AdaptOpts) -> tuple[dict[str, tuple[float, float, float]], str]:
    container: Any = next((person[k] for k in KP_KEYS if k in person), None)
    if container is None or (isinstance(container, dict) and "x" in container and "y" in container):
        src = person if container is None else container
        xs, ys = src.get("x"), src.get("y")
        if not isinstance(xs, list) or not isinstance(ys, list):
            return {}, "none"
        cs = next((src[k] for k in SCORE_KEYS if isinstance(src.get(k), list)), [1.0] * len(xs))
        items = [(i, (xs[i], ys[i], cs[i] if i < len(cs) else 1.0)) for i in range(min(len(xs), len(ys)))]
        layout = "x/y/score arrays"
    elif isinstance(container, dict):
        items = list(container.items())
        layout = "dict by name"
    elif isinstance(container, list) and container and all(isinstance(v, (int, float)) for v in container):
        step = 3 if len(container) % 3 == 0 else 2
        items = [(i, container[i * step:(i + 1) * step]) for i in range(len(container) // step)]
        layout = f"flat list of {'x,y,c' if step == 3 else 'x,y'}"
    elif isinstance(container, list):
        items = list(enumerate(container))
        layout = "list of [x,y,c] per index"
    else:
        return {}, "none"
    out: dict[str, tuple[float, float, float]] = {}
    for key, val in items:
        name = _name_for(key, opts)
        kv = _kp_value(val)
        if name is not None and kv is not None:
            out[name] = kv
    return out, layout


def _parse_bbox(raw: Any, kp: dict[str, tuple[float, float, float]]) -> BBox | None:
    if isinstance(raw, dict):
        if {"left", "top", "width", "height"} <= raw.keys():
            x, y, w, h = (float(raw[k]) for k in ("left", "top", "width", "height"))
            return (x, y, x + w, y + h)
        if {"x1", "y1", "x2", "y2"} <= raw.keys():
            return tuple(float(raw[k]) for k in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]
        if {"x", "y", "w", "h"} <= raw.keys():
            x, y, w, h = (float(raw[k]) for k in ("x", "y", "w", "h"))
            return (x, y, x + w, y + h)
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    a, b, c, d = (float(v) for v in raw)
    xyxy, xywh = (a, b, c, d), (a, b, a + c, b + d)
    if c <= a or d <= b:
        return xywh
    pts = [(x, y) for x, y, conf in kp.values() if conf > 0]
    if pts:
        inside = [sum(bx1 <= x <= bx2 and by1 <= y <= by2 for x, y in pts)
                  for bx1, by1, bx2, by2 in (xyxy, xywh)]
        return xywh if inside[1] > inside[0] else xyxy
    return xyxy


def _parse(obj: Any, opts: AdaptOpts) -> _Parsed:
    if isinstance(obj, list):
        container, persons_raw, list_key = {}, obj, "list"
    elif isinstance(obj, dict):
        container = obj
        list_key = next((k for k in PERSON_LIST_KEYS if isinstance(obj.get(k), list)), None)
        if list_key is not None:
            persons_raw = obj[list_key]
        elif any(k in obj for k in KP_KEYS) or ("x" in obj and "y" in obj):
            persons_raw, list_key = [obj], "single person"
        else:
            persons_raw, list_key = [], "empty"
    else:
        raise ValueError(f"unsupported JSON value of type {type(obj).__name__}")
    tkey = next((k for k in TIME_KEYS if _num(container.get(k)) is not None), None)
    if tkey is not None:
        raw_t, time_field = _num(container[tkey]), tkey
    else:
        fkey = next((k for k in FRAME_KEYS if _num(container.get(k)) is not None), None)
        raw_t, time_field = (_num(container[fkey]), f"frame:{fkey}") if fkey else (None, "none")
    size = opts.frame_size
    if size is None:
        for wk, hk in SIZE_KEYS:
            if _num(container.get(wk)) and _num(container.get(hk)):
                size = (int(container[wk]), int(container[hk]))
                break
    persons, layouts = [], set()
    for p in persons_raw:
        if not isinstance(p, dict):
            continue
        kp, layout = _parse_keypoints(p, opts)
        layouts.add(layout)
        bbox_raw = next((p[k] for k in BBOX_KEYS if k in p), None)
        pid = next((p[k] for k in ID_KEYS if k in p), None)
        persons.append((kp, bbox_raw, int(pid) if _num(pid) is not None else None))
    layout = f"{list_key}: " + (", ".join(sorted(layouts)) if layouts else "no persons")
    return _Parsed(raw_t, time_field, persons, layout, size)


def _all_coords(parsed: _Parsed) -> list[float]:
    return [v for kp, _, _ in parsed.persons for x, y, c in kp.values() if c > 0 for v in (x, y)]


def _is_normalised(vals: list[float], opts: AdaptOpts) -> bool:
    return bool(vals) and all(-0.5 <= v <= opts.norm_coord_max for v in vals)


def _to_frame(parsed: _Parsed, t: float, normalised: bool, norm_max: float) -> Frame:
    sx = sy = 1.0
    if normalised:
        if parsed.frame_size is None:
            raise ValueError("keypoints look normalised (all within [0, 1.5]) but no frame size is known: "
                             "pass --frame-size WxH (or --coords px if they really are pixels)")
        sx, sy = parsed.frame_size
    persons = []
    for kp, bbox_raw, pid in parsed.persons:
        kp_px = {n: (x * sx, y * sy, c) for n, (x, y, c) in kp.items()}
        bbox = _parse_bbox(bbox_raw, kp) if bbox_raw is not None else None
        if bbox is not None and normalised and max(bbox) <= norm_max:
            bbox = (bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy)
        persons.append(Person(kp=kp_px, id=pid, bbox=bbox))
    return Frame(t=t, persons=persons)


def auto_adapt(obj: Any, opts: AdaptOpts | None = None) -> Frame:
    """Convert one raw JSON object from the vision side into a Frame.

    Detects the layout: our replay format; a ``persons``/``objects``/
    ``detections``/``people`` list whose items carry ``kp``/``keypoints``/
    ``landmarks`` as a dict by name, a list of [x, y, conf] per COCO index, a
    flat [x1, y1, c1, ...] list or separate ``x``/``y``/``score`` arrays; an
    optional ``bbox`` ([x1,y1,x2,y2] or [x,y,w,h]) and ``id``/``track_id``/
    ``object_id``. The timestamp comes from ``t``/``ts``/``timestamp``/``pts``
    (converted with ``opts.ts_unit``; "auto" means seconds for a single
    object) or from a frame number divided by ``opts.fps``. Time is not shifted
    here; ``adapt_stream`` does stream-level inference and normalisation.
    """
    opts = opts or AdaptOpts()
    parsed = _parse(obj, opts)
    if parsed.raw_t is None:
        raise ValueError("no timestamp or frame number in object")
    if parsed.time_field.startswith("frame:"):
        t = parsed.raw_t / opts.fps
    else:
        t = parsed.raw_t * UNIT_SCALE.get(opts.ts_unit, 1.0)
    normalised = opts.coords == "norm" or (opts.coords == "auto" and _is_normalised(_all_coords(parsed), opts))
    return _to_frame(parsed, t, normalised, opts.norm_coord_max)


def _infer_unit(raw_ts: list[float], opts: AdaptOpts) -> str:
    d = np.diff(np.array(raw_ts, dtype=float))
    d = d[d > 0]
    if len(d) == 0:
        return "s"
    med = float(np.median(d))
    for unit, scale in UNIT_SCALE.items():
        if opts.frame_dt_min_s <= med * scale <= opts.frame_dt_max_s:
            return unit
    logger.warning("cannot infer timestamp unit from median frame delta %.6g; assuming seconds", med)
    return "s"


def adapt_stream(objs: Iterable[Any], opts: AdaptOpts | None = None, source: str = "input") -> Iterator[Frame]:
    """Turn raw JSON objects into Frames, resolving "auto" options from the first frames.

    Logs once which layout, timestamp unit and coordinate mode were detected.
    Unparseable objects are logged and skipped; a missing frame size for
    normalised coordinates raises a clear error.
    """
    opts = opts or AdaptOpts()
    it = iter(objs)
    probe: list[tuple[Any, _Parsed]] = []
    for obj in it:
        try:
            probe.append((obj, _parse(obj, opts)))
        except ValueError as exc:
            logger.warning("%s: skipping unparseable object (%s)", source, exc)
            continue
        if len(probe) >= opts.probe_frames:
            break
    if not probe:
        return
    fields = {p.time_field for _, p in probe}
    timed = [p.raw_t for _, p in probe if p.raw_t is not None and not p.time_field.startswith("frame:")]
    unit = opts.ts_unit if opts.ts_unit != "auto" else (_infer_unit(timed, opts) if timed else "s")
    coords = opts.coords
    if coords == "auto":
        vals = [v for _, p in probe for v in _all_coords(p)]
        coords = "norm" if _is_normalised(vals, opts) else "px"
    resolved = replace(opts, ts_unit=unit, coords=coords)
    size = next((p.frame_size for _, p in probe if p.frame_size), None)
    if "none" in fields:
        logger.warning("%s: no timestamp or frame number in some objects; they are skipped", source)
    if any(f.startswith("frame:") for f in fields):
        logger.warning("%s: no timestamp field; using frame number / %.3g fps", source, opts.fps)
    logger.info("%s: layout '%s'; time from %s%s; coordinates %s%s", source, probe[0][1].layout,
                "/".join(sorted(fields)), f" in {unit}" if timed else "",
                "normalised" if coords == "norm" else "pixels",
                f" x {size[0]}x{size[1]}" if coords == "norm" and size else "")
    t_first: float | None = None
    counter = 0

    def emit(obj: Any) -> Iterator[Frame]:
        nonlocal t_first, counter
        counter += 1
        try:
            frame = auto_adapt(obj, resolved)
        except ValueError as exc:
            if "frame size" in str(exc):
                raise
            logger.warning("%s: skipping object %d (%s)", source, counter, exc)
            return
        if resolved.normalise_t:
            t_first = frame.t if t_first is None else t_first
            frame = Frame(t=round(frame.t - t_first, 9), persons=frame.persons)
        yield frame

    for obj, _ in probe:
        yield from emit(obj)
    for obj in it:
        yield from emit(obj)


def tail_jsonl(path: str | Path, poll_s: float = 0.05, idle_timeout_s: float | None = None,
               opts: AdaptOpts | None = None) -> Iterator[Frame]:
    """Frames from a growing JSONL file (partial lines and truncation handled)."""
    yield from adapt_stream(tail_objects(path, poll_s, idle_timeout_s), opts, str(path))


def watch_folder(folder: str | Path, pattern: str = "*.json", poll_s: float = 0.1,
                 idle_timeout_s: float | None = None, settle_s: float = 1.0,
                 opts: AdaptOpts | None = None) -> Iterator[Frame]:
    """Frames from per-frame JSON files, each new file once, in filename order."""
    yield from adapt_stream(folder_objects(folder, pattern, poll_s, idle_timeout_s, settle_s), opts, str(folder))


def mqtt_frames(host: str, port: int, topic: str, timeout_s: float | None = None, opts: AdaptOpts | None = None,
                client: Any = None) -> Iterator[Frame]:
    """Frames from an MQTT topic (reconnects with backoff; never raises on broker trouble)."""
    yield from adapt_stream(mqtt_objects(host, port, topic, idle_timeout_s=timeout_s, client=client), opts,
                            f"mqtt://{host}:{port}/{topic}")


def add_adapter_args(ap: Any) -> None:
    """Add --fps/--ts-unit/--coords/--frame-size/--kp-map to an argparse parser."""
    ap.add_argument("--fps", type=float, default=None,
                    help="frame rate for inputs that carry frame numbers instead of timestamps (default from config)")
    ap.add_argument("--ts-unit", choices=["auto", "s", "ms", "us", "ns"], default="auto")
    ap.add_argument("--coords", choices=["auto", "px", "norm"], default="auto")
    ap.add_argument("--frame-size", default=None, help="WxH, needed for normalised coordinates")
    ap.add_argument("--kp-map", default=None, help="JSON file mapping source keypoint index/name -> COCO-17 name")


def opts_from_args(args: Any, io_cfg: Any) -> AdaptOpts:
    """AdaptOpts from parsed CLI args and the ``io`` config section."""
    size = None
    if args.frame_size:
        w, _, h = args.frame_size.lower().partition("x")
        size = (int(w), int(h))
    kp_map = None
    if args.kp_map:
        kp_map = {str(k): v for k, v in json.loads(Path(args.kp_map).read_text(encoding="utf-8")).items()}
    return AdaptOpts(fps=args.fps or io_cfg.default_fps, ts_unit=args.ts_unit, coords=args.coords,
                     frame_size=size, kp_map=kp_map, probe_frames=io_cfg.ts_probe_frames,
                     frame_dt_min_s=io_cfg.frame_dt_min_s, frame_dt_max_s=io_cfg.frame_dt_max_s,
                     norm_coord_max=io_cfg.norm_coord_max)


def read_keypoints(source: str | Path, *, follow: bool = False, poll_s: float = 0.05,
                   idle_timeout_s: float | None = None, settle_s: float = 1.0,
                   opts: AdaptOpts | None = None) -> Iterator[Frame]:
    """Frames from a vision-side source, through ``auto_adapt``.

    ``source`` may be '-' (stdin JSONL), a JSONL file (``follow=True`` tails
    it), a folder of per-frame JSON files, or ``mqtt://host:port/topic``.
    """
    src = str(source)
    if src.startswith("mqtt://"):
        hostport, _, topic = src[len("mqtt://"):].partition("/")
        host, _, port = hostport.partition(":")
        yield from mqtt_frames(host, int(port or 1883), topic or "#", idle_timeout_s, opts)
    elif src != "-" and Path(src).is_dir():
        yield from watch_folder(src, poll_s=poll_s, idle_timeout_s=idle_timeout_s, settle_s=settle_s, opts=opts)
    elif follow:
        yield from tail_jsonl(src, poll_s, idle_timeout_s, opts)
    elif src == "-":
        yield from adapt_stream(_iter_json_lines(_stdin_lines(), "<stdin>"), opts, "<stdin>")
    else:
        with open(src, encoding="utf-8") as fh:
            yield from adapt_stream(_iter_json_lines(fh, src), opts, src)


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
