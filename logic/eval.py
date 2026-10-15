"""Evaluation harness, margin report and threshold tuning.

Replays every clip in one or more folders through read_jsonl -> filters ->
features -> FallDetector and scores alerts against labels. Features do not
depend on the tuned fall thresholds, so they are computed once per clip and
only the state machine is re-run during tuning.

Labels come from ``labels.json`` (synthetic) or, with ``--labels-from-filenames``
(real recordings), from the file names: ``fall_*`` expects FALL_CONFIRMED,
``lie_slow_stay_*`` expects PROLONGED_DOWN, anything else expects no event.
Real clips have no fall start time, so the expected event may occur anywhere
and alert delay is not scored.

    python -m logic.eval --data data/synth
    python -m logic.eval --data data/synth --tune
    python -m logic.eval --data data/real --labels-from-filenames --tune
    python -m logic.eval --data data/synth --bench
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
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
    alerts: list[tuple[str, float]] = field(default_factory=list)


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


# ---------------------------------------------------------------- replay and labels

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


def label_from_name(stem: str) -> dict[str, Any]:
    """Label for a real clip named like ``fall_01`` or ``lie_slow_stay_03``."""
    cls = re.sub(r"[_-]?\d+$", "", stem.lower()) or stem.lower()
    expected = "FALL_CONFIRMED" if cls.startswith("fall") else (
        "PROLONGED_DOWN" if cls.startswith("lie_slow_stay") else None)
    return {"class": cls, "variant": "real", "expected_event": expected, "t_fall_start": None, "real": True}


def load_clips(folders: Iterable[str | Path], cfg: Config, labels_from_filenames: bool = False) -> list[ClipData]:
    """Load every labelled clip in ``folders`` and compute its features."""
    clips = []
    for folder in folders:
        d = Path(folder)
        if labels_from_filenames:
            labels = {p.stem: label_from_name(p.stem) for p in sorted(d.glob("*.jsonl"))}
        else:
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
    """Clip-level outcome.

    A TP needs the *expected* event kind, within ``[t_fall_start, +match_window_s]``
    (anywhere when the start time is unknown). Every other alert, including an
    alert of the wrong kind, is a false alarm.
    """
    lab = clip.label
    expected, t0 = lab.get("expected_event"), lab.get("t_fall_start")
    alerts = [(e.kind, e.ts_s) for e in events if e.kind in ALERT_KINDS]
    res = ClipResult(clip.name, lab.get("class", "?"), lab.get("variant", "?"), expected, t0, "TN",
                     events=[f"{e.kind}@{e.ts_s:.2f}" for e in events], alerts=alerts)
    hit = None
    if expected:
        hit = next(((k, t) for k, t in alerts
                    if k == expected and (t0 is None or t0 <= t <= t0 + match_window_s)), None)
        res.outcome = "TP" if hit else "FN"
        if hit:
            res.alert_kind, res.alert_t = hit
            res.delay_s = hit[1] - t0 if t0 is not None else None
    elif alerts:
        res.outcome = "FP"
        res.alert_kind, res.alert_t = alerts[0]
    res.false_alarms = len(alerts) - (1 if hit else 0)
    return res


def summarize(results: list[ClipResult], clips: list[ClipData], kind: str | None = None) -> Metrics:
    """Confusion counts, false alarms per hour and delays.

    With ``kind`` the counts are for that alert kind only: positives are clips
    expecting it, and an alert of that kind on any other clip is a false positive.
    """
    m = Metrics(hours=sum(c.duration_s for c in clips) / 3600.0)
    for r in results:
        if kind is None:
            outcome, fas = r.outcome, r.false_alarms
        else:
            expects = r.expected == kind
            fired = [a for a in r.alerts if a[0] == kind]
            if expects:
                outcome = r.outcome
                fas = len(fired) - (1 if r.outcome == "TP" else 0)
            else:
                outcome, fas = ("FP" if fired else "TN"), len(fired)
        setattr(m, outcome, getattr(m, outcome) + 1)
        m.false_alarms += fas
        if outcome == "TP" and r.alert_kind and r.delay_s is not None:
            m.delays.setdefault(r.alert_kind, []).append(r.delay_s)
    return m


def evaluate(clips: list[ClipData], fall_cfg: FallCfg, match_window_s: float) -> tuple[Metrics, list[ClipResult]]:
    """Score ``clips`` under ``fall_cfg`` (overall metrics)."""
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


# ---------------------------------------------------------------- margins

@dataclass
class Margin:
    """Per-clip feature extremes relative to the thresholds."""

    name: str
    cls: str
    expected: str | None
    v_ratio: float | None  # peak hip drop speed / V_DROP
    phi_max: float | None
    h_min: float | None
    down_s: float  # longest continuous time with phi > PHI_DOWN and (h < H_LOW or h unknown)


def margins(clips: list[ClipData], fall: FallCfg) -> list[Margin]:
    """Where each clip sits relative to V_DROP, PHI_DOWN and H_LOW."""
    out = []
    for c in clips:
        vs = [f.v for f in c.feats if f.v is not None]
        phis = [f.phi for f in c.feats if f.phi is not None]
        hs = [f.h for f in c.feats if f.h is not None]
        longest = run = 0.0
        prev_t = None
        for f in c.feats:
            down = f.phi is not None and f.phi > fall.PHI_DOWN and (f.h is None or f.h < fall.H_LOW)
            if down and prev_t is not None:
                run += f.t - prev_t
            elif not down:
                run = 0.0
            longest = max(longest, run)
            prev_t = f.t if down else None
        out.append(Margin(c.name, c.label.get("class", "?"), c.label.get("expected_event"),
                          max(vs) / fall.V_DROP if vs else None, max(phis) if phis else None,
                          min(hs) if hs else None, longest))
    return out


def format_margins(rows: list[Margin], fall: FallCfg) -> str:
    """Per-clip and per-class margin tables (Markdown)."""
    def f(x: float | None, nd: int = 2) -> str:
        return "-" if x is None else f"{x:.{nd}f}"

    lines = [f"Thresholds: V_DROP={fall.V_DROP:g} L/s, PHI_DOWN={fall.PHI_DOWN:g} deg, H_LOW={fall.H_LOW:g} L, "
             f"T_HOLD={fall.T_HOLD:g} s, prolonged_hold_s={fall.prolonged_hold_s:g} s", "",
             "| Clip | Class | Expected | peak v / V_DROP | peak phi (deg) | min h (L) | longest down (s) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r.name} | {r.cls} | {r.expected or '-'} | {f(r.v_ratio)} | {f(r.phi_max, 0)} | "
                     f"{f(r.h_min)} | {r.down_s:.1f} |")
    lines += ["", "| Class | n | peak v / V_DROP (min-max) | peak phi (min-max) | min h (min-max) | "
              "longest down s (min-max) |", "|---|---|---|---|---|---|"]
    for cls in sorted({r.cls for r in rows}):
        g = [r for r in rows if r.cls == cls]

        def rng(vals: list[float | None], nd: int = 2) -> str:
            v = [x for x in vals if x is not None]
            return "-" if not v else f"{min(v):.{nd}f}-{max(v):.{nd}f}"
        lines.append(f"| {cls} | {len(g)} | {rng([r.v_ratio for r in g])} | {rng([r.phi_max for r in g], 0)} | "
                     f"{rng([r.h_min for r in g])} | {rng([r.down_s for r in g], 1)} |")
    return "\n".join(lines) + "\n"


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


def tune(train: list[ClipData], test: list[ClipData], cfg: Config,
         grids: dict[str, tuple[float, ...]] | None = None) -> list[Trial]:
    """Grid search over TUNED_PARAMS; best first.

    Objective: maximise train F1 subject to zero train false alarms; ties go to
    the point closest to the defaults (normalised by grid span), then to the
    lowest mean fast-fall alert delay.
    """
    e = cfg.eval
    grids = grids or {"V_DROP": e.grid_V_DROP, "PHI_DOWN": e.grid_PHI_DOWN, "H_LOW": e.grid_H_LOW,
                      "T_HOLD": e.grid_T_HOLD}
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


def report(name: str, results: list[ClipResult], clips: list[ClipData]) -> str:
    """Overall metrics plus FALL_CONFIRMED and PROLONGED_DOWN views."""
    parts = [format_metrics(f"{name} | all events", summarize(results, clips))]
    for kind in ALERT_KINDS:
        if any(r.expected == kind for r in results) or any(a[0] == kind for r in results for a in r.alerts):
            parts.append(format_metrics(f"{name} | {kind}", summarize(results, clips, kind)))
    return "\n".join(parts)


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
        for p in sorted(Path(folder).glob("*.jsonl")):
            all_frames.append(list(read_jsonl(p)))
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


def _print_trials(trials: list[Trial], n: int, with_test: bool) -> None:
    for i, tr in enumerate(trials[:n], 1):
        p = " ".join(f"{k}={v:g}" for k, v in tr.params.items())
        line = f"  {i}. {p} | train F1={tr.train.f1:.3f} FA={tr.train.false_alarms}"
        if with_test:
            td = tr.test.delay("FALL_CONFIRMED")[0]
            line += (f" | test F1={tr.test.f1:.3f} R={tr.test.recall:.3f} FA={tr.test.false_alarms} "
                     f"delay={'-' if td is None else f'{td:.2f}s'}")
        print(line)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m logic.eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", action="append", required=True, help="clip folder (repeatable)")
    ap.add_argument("--labels-from-filenames", action="store_true",
                    help="real clips: derive labels from names (fall_*, lie_slow_stay_*, others = no event)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--margins", action="store_true", help="print the margin report (always on for real clips)")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--out", default="out", help="folder for CSV output")
    ap.add_argument("--tuned-out", default=str(TUNED_PATH))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    e = cfg.eval

    if args.bench:
        b = bench(args.data, cfg)
        print(f"bench: {b['frames']} frames; JSON parse {b['parse_fps']:.0f} frames/s; "
              f"pipeline+detector {b['detector_fps']:.0f} frames/s (one core)")
        return 0

    real = args.labels_from_filenames
    clips = load_clips(args.data, cfg, labels_from_filenames=real)
    if real:
        train, test = clips, []
        print(f"{len(clips)} real clips (labels from file names; no held-out split, delay not scored)")
    else:
        train, test = split(clips, e.test_fraction, e.split_seed)
        print(f"{len(clips)} clips: {len(train)} train / {len(test)} test (seed {e.split_seed})")

    if args.tune:
        grids = None
        if real:
            grids = {"V_DROP": e.real_grid_V_DROP, "PHI_DOWN": e.real_grid_PHI_DOWN, "H_LOW": e.real_grid_H_LOW,
                     "T_HOLD": e.real_grid_T_HOLD}
        t = time.perf_counter()
        trials = tune(train, test, cfg, grids)
        print(f"grid search: {len(trials)} combinations in {time.perf_counter() - t:.1f}s")
        print("top 5 (objective: max train F1 with zero train false alarms; ties -> closest to defaults):")
        _print_trials(trials, 5, with_test=not real)
        n_feasible = sum(1 for tr in trials if tr.train.false_alarms == 0)
        print(f"{n_feasible}/{len(trials)} combinations have zero train false alarms")
        best = trials[0]
        if real and len(clips) < e.real_min_clips_to_write:
            print(f"WARNING: only {len(clips)} real clips (< {e.real_min_clips_to_write}); "
                  f"not writing {args.tuned_out}. Use the margin report below to choose thresholds by hand.")
        else:
            cfg = replace(cfg, fall=replace(cfg.fall, **best.params))
            save_config(cfg, args.tuned_out)
            print(f"wrote {args.tuned_out}")

    rows: list[tuple[str, ClipResult]] = []
    for name, subset in (("train", train), ("test", test)) if not real else (("real", train),):
        if not subset:
            continue
        _, results = evaluate(subset, cfg.fall, e.match_window_s)
        print(report(name, results, subset))
        rows += [(name, r) for r in results]
        for r in results:
            if r.outcome in ("FN", "FP") or r.false_alarms:
                print(f"    {r.outcome}: {r.name} events={r.events}")
    if not real:
        print(report("all", [r for _, r in rows], clips))
    if real or args.margins:
        print(format_margins(margins(clips, cfg.fall), cfg.fall))
    csv_path = Path(args.out) / "eval_results.csv"
    write_csv(csv_path, rows)
    print(f"per-clip results: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
