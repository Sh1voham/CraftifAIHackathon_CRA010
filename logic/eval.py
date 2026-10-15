"""Evaluation harness and threshold tuning.

Replays every clip in one or more folders (``<name>.jsonl`` + ``labels.json``)
through read_jsonl -> filters -> features -> FallDetector and scores alerts
against the labels. Features do not depend on the tuned fall thresholds, so
they are computed once per clip and only the state machine is re-run.

    python -m logic.eval --data data/synth
    python -m logic.eval --data data/synth --tune
    python -m logic.eval --data data/synth --bench
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from logic.config import Config, FallCfg, load_config, save_config
from logic.fall import ALERT_KINDS, Event, FallDetector
from logic.features import Features
from logic.io import Frame, read_jsonl
from logic.pipeline import FramePipeline

TUNED_PATH = Path(__file__).with_name("config.tuned.json")
TUNED_PARAMS = ("V_DROP", "PHI_DOWN", "H_LOW", "T_HOLD")


@dataclass
class ClipData:
    """A labelled clip with its cached features."""

    name: str
    label: dict[str, Any]
    feats: list[Features]
    duration_s: float
    n_frames: int


@dataclass
class ClipResult:
    """Outcome of one clip."""

    name: str
    cls: str
    variant: str
    expected: str | None
    t_fall_start: float | None
    outcome: str  # TP | FN | FP | TN
    alert_kind: str | None = None
    alert_t: float | None = None
    delay_s: float | None = None
    false_alarms: int = 0
    events: list[str] = field(default_factory=list)


@dataclass
class Metrics:
    """Aggregate scores over a set of clips."""

    TP: int = 0
    FN: int = 0
    FP: int = 0
    TN: int = 0
    false_alarms: int = 0
    hours: float = 0.0
    delays: dict[str, list[float]] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        return self.TP / (self.TP + self.FP) if self.TP + self.FP else 0.0

    @property
    def recall(self) -> float:
        return self.TP / (self.TP + self.FN) if self.TP + self.FN else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def fa_per_hour(self) -> float:
        return self.false_alarms / self.hours if self.hours else 0.0

    def delay(self, kind: str) -> tuple[float | None, float | None]:
        d = self.delays.get(kind, [])
        return (float(np.mean(d)), float(np.max(d))) if d else (None, None)


# ---------------------------------------------------------------- replay

def clip_features(frames: Iterable[Frame], cfg: Config) -> list[Features]:
    """Run the front half of the pipeline over a clip."""
    pipe = FramePipeline(cfg)
    out: list[Features] = []
    for f in frames:
        out += [s.features for s in pipe.push(f)]
    return out + [s.features for s in pipe.flush()]


def run_detector(feats: list[Features], fall_cfg: FallCfg) -> list[Event]:
    """Run a fresh FallDetector over cached features."""
    det = FallDetector(fall_cfg)
    events: list[Event] = []
    for f in feats:
        events += det.update(f.t, f)
    return events


def load_clips(folders: Iterable[str | Path], cfg: Config) -> list[ClipData]:
    """Load every labelled clip in ``folders`` and compute its features."""
    clips = []
    for folder in folders:
        d = Path(folder)
        labels = json.loads((d / "labels.json").read_text(encoding="utf-8"))
        for name in sorted(labels):
            frames = list(read_jsonl(d / f"{name}.jsonl"))
            if not frames:
                continue
            clips.append(ClipData(name=name, label=labels[name], feats=clip_features(frames, cfg),
                                  duration_s=frames[-1].t - frames[0].t, n_frames=len(frames)))
    return clips


# ---------------------------------------------------------------- scoring

def score_clip(clip: ClipData, events: list[Event], match_window_s: float) -> ClipResult:
    """Clip-level outcome. Alerts outside the match window count as false alarms."""
    lab = clip.label
    expected, t0 = lab.get("expected_event"), lab.get("t_fall_start")
    alerts = [e for e in events if e.kind in ALERT_KINDS]
    res = ClipResult(clip.name, lab.get("class", "?"), lab.get("variant", "?"), expected, t0, "TN",
                     events=[f"{e.kind}@{e.ts_s:.2f}" for e in events])
    if expected:
        hit = next((e for e in alerts if t0 is not None and t0 <= e.ts_s <= t0 + match_window_s), None)
        if hit is not None:
            res.outcome, res.alert_kind, res.alert_t, res.delay_s = "TP", hit.kind, hit.ts_s, hit.ts_s - t0
        else:
            res.outcome = "FN"
        res.false_alarms = sum(1 for e in alerts if e is not hit)
    else:
        res.false_alarms = len(alerts)
        if alerts:
            res.outcome, res.alert_kind, res.alert_t = "FP", alerts[0].kind, alerts[0].ts_s
    return res


def summarize(results: list[ClipResult], clips: list[ClipData]) -> Metrics:
    """Confusion counts, false alarms per hour and delays by alert kind."""
    m = Metrics(hours=sum(c.duration_s for c in clips) / 3600.0)
    for r in results:
        setattr(m, r.outcome, getattr(m, r.outcome) + 1)
        m.false_alarms += r.false_alarms
        if r.outcome == "TP" and r.alert_kind and r.delay_s is not None:
            m.delays.setdefault(r.alert_kind, []).append(r.delay_s)
    return m


def evaluate(clips: list[ClipData], fall_cfg: FallCfg, match_window_s: float) -> tuple[Metrics, list[ClipResult]]:
    """Score ``clips`` under ``fall_cfg``."""
    results = [score_clip(c, run_detector(c.feats, fall_cfg), match_window_s) for c in clips]
    return summarize(results, clips), results


def split(clips: list[ClipData], test_fraction: float, seed: int) -> tuple[list[ClipData], list[ClipData]]:
    """Seeded split, stratified by (class, variant)."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[ClipData]] = {}
    for c in clips:
        groups.setdefault((c.label.get("class", "?"), c.label.get("variant", "?")), []).append(c)
    train, test = [], []
    for key in sorted(groups):
        g = groups[key]
        order = rng.permutation(len(g))
        n_test = int(round(test_fraction * len(g)))
        test += [g[i] for i in order[:n_test]]
        train += [g[i] for i in order[n_test:]]
    return train, test


# ---------------------------------------------------------------- tuning

@dataclass
class Trial:
    """One grid point with its train and test scores."""

    params: dict[str, float]
    train: Metrics
    test: Metrics
    distance: float


def _rank_key(tr: Trial) -> tuple:
    fast_mean = tr.train.delay("FALL_CONFIRMED")[0]
    return (tr.train.false_alarms != 0, -tr.train.f1, tr.distance, fast_mean if fast_mean is not None else 1e9)


def tune(train: list[ClipData], test: list[ClipData], cfg: Config) -> list[Trial]:
    """Grid search over TUNED_PARAMS; best first.

    Objective: maximise train F1 subject to zero train false alarms; ties go to
    the point closest to the defaults (normalised by grid span), then to the
    lowest mean fast-fall alert delay.
    """
    e = cfg.eval
    grids = {"V_DROP": e.grid_V_DROP, "PHI_DOWN": e.grid_PHI_DOWN, "H_LOW": e.grid_H_LOW, "T_HOLD": e.grid_T_HOLD}
    spans = {k: (max(v) - min(v)) or 1.0 for k, v in grids.items()}
    trials = []
    for combo in itertools.product(*(grids[k] for k in TUNED_PARAMS)):
        params = dict(zip(TUNED_PARAMS, combo))
        fc = replace(cfg.fall, **params)
        dist = sum(abs(params[k] - getattr(cfg.fall, k)) / spans[k] for k in TUNED_PARAMS)
        trials.append(Trial(params, evaluate(train, fc, e.match_window_s)[0],
                            evaluate(test, fc, e.match_window_s)[0], dist))
    return sorted(trials, key=_rank_key)


# ---------------------------------------------------------------- reporting

def format_metrics(name: str, m: Metrics) -> str:
    """Multi-line human-readable summary."""
    lines = [
        f"[{name}] clips={m.TP + m.FN + m.FP + m.TN}  footage={m.hours * 60:.1f} min",
        "              predicted alert   predicted none",
        f"  positive    TP={m.TP:<14d} FN={m.FN}",
        f"  negative    FP={m.FP:<14d} TN={m.TN}",
        f"  precision={m.precision:.3f}  recall={m.recall:.3f}  F1={m.f1:.3f}  "
        f"false alarms={m.false_alarms} ({m.fa_per_hour:.2f}/h)",
    ]
    for kind in ALERT_KINDS:
        mean, mx = m.delay(kind)
        if mean is not None:
            lines.append(f"  {kind} delay after fall start: mean={mean:.2f}s max={mx:.2f}s "
                         f"(n={len(m.delays[kind])})")
    return "\n".join(lines)


def write_csv(path: str | Path, rows: list[tuple[str, ClipResult]]) -> None:
    """Per-clip results."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "clip", "class", "variant", "expected", "t_fall_start", "outcome", "alert_kind",
                    "alert_t", "delay_s", "false_alarms", "events"])
        for split_name, r in rows:
            w.writerow([split_name, r.name, r.cls, r.variant, r.expected or "", r.t_fall_start or "", r.outcome,
                        r.alert_kind or "", "" if r.alert_t is None else f"{r.alert_t:.3f}",
                        "" if r.delay_s is None else f"{r.delay_s:.3f}", r.false_alarms, " ".join(r.events)])


def bench(folders: list[str], cfg: Config) -> dict[str, float]:
    """Single-thread throughput: JSON parsing, and pipeline + detector on parsed frames."""
    all_frames: list[list[Frame]] = []
    t = time.perf_counter()
    for folder in folders:
        d = Path(folder)
        for name in sorted(json.loads((d / "labels.json").read_text(encoding="utf-8"))):
            all_frames.append(list(read_jsonl(d / f"{name}.jsonl")))
    parse_s = time.perf_counter() - t
    n = sum(len(f) for f in all_frames)
    t = time.perf_counter()
    for frames in all_frames:
        pipe, det = FramePipeline(cfg), FallDetector(cfg.fall)
        for f in frames:
            for s in pipe.push(f):
                det.update(s.features.t, s.features)
        for s in pipe.flush():
            det.update(s.features.t, s.features)
    proc_s = time.perf_counter() - t
    return {"frames": n, "parse_fps": n / parse_s, "detector_fps": n / proc_s}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m logic.eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True, help="clip folder (repeatable)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", default="out", help="folder for CSV output")
    ap.add_argument("--tuned-out", default=str(TUNED_PATH))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.bench:
        b = bench(args.data, cfg)
        print(f"bench: {b['frames']} frames; JSON parse {b['parse_fps']:.0f} frames/s; "
              f"pipeline+detector {b['detector_fps']:.0f} frames/s (one core)")
        return 0

    clips = load_clips(args.data, cfg)
    train, test = split(clips, cfg.eval.test_fraction, cfg.eval.split_seed)
    print(f"{len(clips)} clips: {len(train)} train / {len(test)} test (seed {cfg.eval.split_seed})")

    if args.tune:
        t = time.perf_counter()
        trials = tune(train, test, cfg)
        print(f"grid search: {len(trials)} combinations in {time.perf_counter() - t:.1f}s")
        print("top 5 (objective: max train F1 with zero train false alarms; ties -> closest to defaults):")
        for i, tr in enumerate(trials[:5], 1):
            p = " ".join(f"{k}={v:g}" for k, v in tr.params.items())
            td = tr.test.delay("FALL_CONFIRMED")[0]
            print(f"  {i}. {p} | train F1={tr.train.f1:.3f} FA={tr.train.false_alarms} | "
                  f"test F1={tr.test.f1:.3f} R={tr.test.recall:.3f} FA={tr.test.false_alarms} "
                  f"delay={'-' if td is None else f'{td:.2f}s'}")
        n_feasible = sum(1 for tr in trials if tr.train.false_alarms == 0)
        print(f"{n_feasible}/{len(trials)} combinations have zero train false alarms")
        best = trials[0]
        cfg = replace(cfg, fall=replace(cfg.fall, **best.params))
        save_config(cfg, args.tuned_out)
        print(f"wrote {args.tuned_out}")

    rows: list[tuple[str, ClipResult]] = []
    for name, subset in (("train", train), ("test", test)):
        m, results = evaluate(subset, cfg.fall, cfg.eval.match_window_s)
        print(format_metrics(name, m))
        rows += [(name, r) for r in results]
        for r in results:
            if r.outcome in ("FN", "FP") or r.false_alarms:
                print(f"    {r.outcome}: {r.name} events={r.events}")
    m_all = summarize([r for _, r in rows], clips)
    print(format_metrics("all", m_all))
    csv_path = Path(args.out) / "eval_results.csv"
    write_csv(csv_path, rows)
    print(f"per-clip results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
