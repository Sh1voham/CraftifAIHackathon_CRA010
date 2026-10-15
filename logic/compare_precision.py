"""Compare keypoint streams from two model precisions (e.g. FP16 vs INT8).

    python -m logic.compare_precision --a fp16.jsonl --b int8.jsonl
    python -m logic.compare_precision --a runs/fp16/ --b runs/int8/ --labels runs/labels.json --out precision.md

``--a``/``--b`` are replay JSONL files or folders of them (matched by file
name). Run A is the reference. Frames are aligned by timestamp (nearest,
within half the median frame interval). Reported: mean keypoint error
normalised by L_ref (overall and per keypoint), the share of keypoints whose
confidence drops by more than ``eval.precision_conf_drop`` (0.2), and the
fall alerts each run produces.
Latency, FPS and power come from PipeGen's speed reports, not from here.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from logic.config import Config, load_config
from logic.eval import ClipData, clip_features, run_detector, score_clip
from logic.fall import ALERT_KINDS
from logic.filters import RefScale
from logic.io import KEYPOINT_NAMES, Frame, TrackState, read_jsonl, select_person


@dataclass
class Comparison:
    """Accumulated A-vs-B statistics."""

    frames_a: int = 0
    frames_b: int = 0
    aligned: int = 0
    errors: dict[str, list[float]] = field(default_factory=lambda: {n: [] for n in KEYPOINT_NAMES})
    kp_in_a: int = 0
    conf_drops: int = 0
    alerts_a: int = 0
    alerts_b: int = 0
    caught_a: int = 0
    caught_b: int = 0
    positives: int = 0
    labelled: bool = False
    conf_drop_threshold: float = 0.2

    @property
    def mean_error(self) -> float | None:
        all_e = [e for v in self.errors.values() for e in v]
        return float(np.mean(all_e)) if all_e else None

    @property
    def conf_drop_frac(self) -> float | None:
        return self.conf_drops / self.kp_in_a if self.kp_in_a else None


def align(a: list[Frame], b: list[Frame], frac: float) -> list[tuple[Frame, Frame]]:
    """Pair each frame of ``a`` with the nearest frame of ``b`` within ``frac`` x median interval of ``a``."""
    if len(a) < 2 or not b:
        return []
    tol = frac * float(np.median(np.diff([f.t for f in a])))
    tb = np.array([f.t for f in b])
    pairs = []
    for fa in a:
        j = int(np.searchsorted(tb, fa.t))
        best = min((k for k in (j - 1, j) if 0 <= k < len(b)), key=lambda k: abs(tb[k] - fa.t))
        if abs(tb[best] - fa.t) <= tol:
            pairs.append((fa, b[best]))
    return pairs


def _torso(kp: dict, c_min: float) -> float | None:
    pts = [kp.get(n) for n in ("l_shoulder", "r_shoulder", "l_hip", "r_hip")]
    if any(p is None or p[2] < c_min for p in pts):
        return None
    ls, rs, lh, rh = pts  # type: ignore[misc]
    return math.hypot((ls[0] + rs[0] - lh[0] - rh[0]) / 2, (ls[1] + rs[1] - lh[1] - rh[1]) / 2)


def compare_streams(a: list[Frame], b: list[Frame], cfg: Config, out: Comparison) -> None:
    """Add keypoint error and confidence statistics for one clip to ``out``."""
    c_min = cfg.filters.c_min
    ta, tb = TrackState(c_min, cfg.fall.lost_person_grace_s), TrackState(c_min, cfg.fall.lost_person_grace_s)
    ref = RefScale(cfg.filters.Lref_window_s)
    out.frames_a += len(a)
    out.frames_b += len(b)
    for fa, fb in align(a, b, cfg.eval.precision_align_frac):
        pa, pb = select_person(fa, ta), select_person(fb, tb)
        if pa is None:
            continue
        L = ref.update(fa.t, _torso(pa.kp, c_min))
        out.aligned += 1
        for name in KEYPOINT_NAMES:
            ka = pa.kp.get(name)
            if ka is None or ka[2] < c_min:
                continue
            out.kp_in_a += 1
            kb = pb.kp.get(name) if pb is not None else None
            cb = kb[2] if kb is not None else 0.0
            if ka[2] - cb > cfg.eval.precision_conf_drop:
                out.conf_drops += 1
            if kb is not None and cb >= c_min and L:
                out.errors[name].append(math.hypot(ka[0] - kb[0], ka[1] - kb[1]) / L)


def count_alerts(frames: list[Frame], cfg: Config, label: dict | None) -> tuple[int, bool]:
    """(number of alerts, caught the labelled event?) for one run of a clip."""
    events = run_detector(clip_features(frames, cfg), cfg.fall)
    n = sum(1 for e in events if e.kind in ALERT_KINDS)
    if not label or not frames:
        return n, False
    clip = ClipData("c", label, [], frames[-1].t - frames[0].t, len(frames))
    return n, score_clip(clip, events, cfg.eval.match_window_s).outcome == "TP"


def _pairs(a: Path, b: Path) -> list[tuple[str, Path, Path]]:
    if a.is_dir():
        return [(f.stem, f, b / f.name) for f in sorted(a.glob("*.jsonl")) if (b / f.name).exists()]
    return [(a.stem, a, b)]


def run(a: str | Path, b: str | Path, cfg: Config, labels: dict | None = None) -> Comparison:
    """Compare every clip pair."""
    out = Comparison(labelled=labels is not None, conf_drop_threshold=cfg.eval.precision_conf_drop)
    for name, pa, pb in _pairs(Path(a), Path(b)):
        fa, fb = list(read_jsonl(pa)), list(read_jsonl(pb))
        compare_streams(fa, fb, cfg, out)
        label = labels.get(name) if labels else None
        na, ca = count_alerts(fa, cfg, label)
        nb, cb = count_alerts(fb, cfg, label)
        out.alerts_a += na
        out.alerts_b += nb
        if label and label.get("expected_event"):
            out.positives += 1
            out.caught_a += int(ca)
            out.caught_b += int(cb)
    return out


def markdown(c: Comparison, name_a: str = "FP16", name_b: str = "INT8") -> str:
    """Markdown tables for the Edge Optimization write-up."""
    def pct(x: float | None) -> str:
        return "-" if x is None else f"{100 * x:.1f} %"

    me = c.mean_error
    lines = [
        f"| Metric | {name_a} (reference) | {name_b} |",
        "|---|---|---|",
        f"| Frames | {c.frames_a} | {c.frames_b} |",
        f"| Aligned frames | {c.aligned} | {c.aligned} |",
        f"| Mean keypoint error (/ L_ref) | 0 | {'-' if me is None else f'{me:.4f}'} |",
        f"| Keypoints with confidence drop > {c.conf_drop_threshold:g} | - | {pct(c.conf_drop_frac)} |",
        f"| Fall alerts produced | {c.alerts_a} | {c.alerts_b} |",
    ]
    if c.labelled:
        lines.append(f"| Labelled falls caught | {c.caught_a} / {c.positives} | {c.caught_b} / {c.positives} |")
    lines += ["", "Latency, FPS and power: copy from PipeGen's speed reports for each engine.", "",
              "| Keypoint | Mean error (/ L_ref) | Samples |", "|---|---|---|"]
    for n in KEYPOINT_NAMES:
        e = c.errors[n]
        lines.append(f"| {n} | {np.mean(e):.4f} | {len(e)} |" if e else f"| {n} | - | 0 |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m logic.compare_precision", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="reference run (file or folder), e.g. FP16")
    ap.add_argument("--b", required=True, help="compared run (file or folder), e.g. INT8")
    ap.add_argument("--name-a", default="FP16")
    ap.add_argument("--name-b", default="INT8")
    ap.add_argument("--labels", default=None, help="labels.json (same format as eval) to count caught falls")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="also write the Markdown here")
    args = ap.parse_args(argv)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8")) if args.labels else None
    md = markdown(run(args.a, args.b, load_config(args.config), labels), args.name_a, args.name_b)
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
