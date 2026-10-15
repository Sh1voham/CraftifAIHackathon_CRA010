"""Generate docs/RESULTS.md from measured runs (no hand-typed numbers).

    python -m logic.make_results                      # synthetic benchmark, gait, TUG, throughput
    python -m logic.make_results --real data/real     # plus real clips named fall_01, sit_01, ...

Everything is regenerated from seeds and the current config, so the file can
be rebuilt at any time.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from logic.config import Config, load_config
from logic.eval import (
    ClipData, Metrics, clip_features, evaluate, format_margins, load_clips, margins, split, summarize, tune,
)
from logic.fall import ALERT_KINDS
from logic.gait import Calibration, analyze_gait, samples_from_steps
from logic.incidents import WallClock
from logic.pipeline import FramePipeline
from logic.run import GaitMonitor, Runner
from logic.synth import generate_dataset, make_tug, make_walk
from logic.tug import segment_tug

OUT = Path(__file__).resolve().parents[1] / "docs" / "RESULTS.md"


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1], timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _confusion(title: str, m: Metrics, kind: str | None = None) -> list[str]:
    lines = [f"**{title}**", "", "| | predicted alert | predicted no alert |", "|---|---|---|",
             f"| expected | TP = {m.TP} | FN = {m.FN} |", f"| not expected | FP = {m.FP} | TN = {m.TN} |", "",
             f"precision {m.precision:.3f} · recall {m.recall:.3f} · F1 {m.f1:.3f} · "
             f"false alarms {m.false_alarms} ({m.fa_per_hour:.2f} per hour of footage, {m.hours * 60:.1f} min)"]
    for k in ([kind] if kind else ALERT_KINDS):
        mean, mx = m.delay(k)
        if mean is not None:
            lines.append(f"· {k} delay after fall start: mean {mean:.2f} s, max {mx:.2f} s (n = {len(m.delays[k])})")
    return lines + [""]


def synthetic_section(cfg: Config, n: int, seed: int) -> tuple[list[str], list[ClipData]]:
    clips = generate_dataset(n, seed)
    data = [ClipData(c.name, c.label, clip_features(c.frames, cfg), c.frames[-1].t - c.frames[0].t, len(c.frames))
            for c in clips]
    train, test = split(data, cfg.eval.test_fraction, cfg.eval.split_seed)
    _, res_test = evaluate(test, cfg.fall, cfg.eval.match_window_s)
    _, res_train = evaluate(train, cfg.fall, cfg.eval.match_window_s)
    lines = [f"## 1. Synthetic fall benchmark", "",
             f"{len(data)} seeded synthetic clips (seed {seed}): "
             f"{sum(1 for c in data if c.label['expected_event'] == 'FALL_CONFIRMED')} fast falls, "
             f"{sum(1 for c in data if c.label['expected_event'] == 'PROLONGED_DOWN')} slow lowering to the floor, "
             f"{sum(1 for c in data if not c.label['expected_event'])} non-fall clips. Stratified "
             f"{1 - cfg.eval.test_fraction:.0%}/{cfg.eval.test_fraction:.0%} train/test split (seed "
             f"{cfg.eval.split_seed}); current config thresholds. Any alert that the clip does not expect "
             f"counts as a false alarm.", "", "### Held-out test split", ""]
    lines += _confusion("All events", summarize(res_test, test))
    lines += _confusion("Fast falls (FALL_CONFIRMED)", summarize(res_test, test, "FALL_CONFIRMED"), "FALL_CONFIRMED")
    lines += _confusion("Slow lowering (PROLONGED_DOWN)", summarize(res_test, test, "PROLONGED_DOWN"),
                        "PROLONGED_DOWN")
    lines += ["### Train split", ""]
    lines += _confusion("All events", summarize(res_train, train))
    per_class: dict[str, list[int]] = {}
    for r in res_train + res_test:
        per_class.setdefault(r.cls, [0, 0])
        per_class[r.cls][0] += 1
        per_class[r.cls][1] += int(bool(r.alerts))
    lines += ["### Alerts per class (all clips)", "", "| Class | clips | clips with any alert |", "|---|---|---|"]
    lines += [f"| {k} | {v[0]} | {v[1]} |" for k, v in sorted(per_class.items())] + [""]
    t = time.perf_counter()
    trials = tune(train, test, cfg)
    tune_s = time.perf_counter() - t
    feasible = sum(1 for tr in trials if tr.train.false_alarms == 0)
    lines += ["### Threshold grid search", "",
              f"{len(trials)} combinations of V_DROP × PHI_DOWN × H_LOW × T_HOLD in {tune_s:.0f} s; "
              f"{feasible} have zero train false alarms. Top 5 (max train F1 with zero false alarms, ties to "
              "the defaults):", "", "| # | V_DROP | PHI_DOWN | H_LOW | T_HOLD | train F1 | test F1 | test FA |",
              "|---|---|---|---|---|---|---|---|"]
    for i, tr in enumerate(trials[:5], 1):
        p = tr.params
        lines.append(f"| {i} | {p['V_DROP']:g} | {p['PHI_DOWN']:g} | {p['H_LOW']:g} | {p['T_HOLD']:g} | "
                     f"{tr.train.f1:.3f} | {tr.test.f1:.3f} | {tr.test.false_alarms} |")
    bad_v = sorted({tr.params["V_DROP"] for tr in trials if tr.train.false_alarms})
    lines += ["", f"Combinations with false alarms all use V_DROP in {bad_v}." if bad_v else "", ""]
    lines += ["### Margins per class (synthetic)", "", format_margins(margins(data, cfg.fall), cfg.fall)
              .split("\n\n")[0], "", _class_margins(data, cfg), ""]
    return lines, data


def _class_margins(data: list[ClipData], cfg: Config) -> str:
    text = format_margins(margins(data, cfg.fall), cfg.fall)
    return text[text.index("| Class | n |"):]


def gait_section(cfg: Config) -> list[str]:
    rows = []
    for seed, (spd, cad, asym, ppm) in enumerate([(1.0, 100, 0.0, 160), (0.8, 90, 0.15, 160), (1.3, 115, 0.0, 400),
                                                    (1.0, 100, 0.15, 400), (0.6, 85, 0.10, 160)], start=1):
        clip = make_walk(seed, speed=spd, cadence=cad, left_slower=asym, px_per_m=ppm,
                         duration=10.0 if spd < 0.8 else 8.0)
        pipe = FramePipeline(cfg)
        steps = []
        for f in clip.frames:
            steps += pipe.push(f)
        steps += pipe.flush()
        m = clip.label["marker_px"]
        g = analyze_gait(samples_from_steps(steps), cfg.gait, Calibration.from_points((m[0], 0), (m[1], 0), 3.0))
        rows.append(f"| {spd:g} m/s, {cad} spm, left {asym:.0%} slower, {ppm:g} px/m | {g.walk_speed_mps:.3f} | "
                    f"{g.cadence_spm:.1f} | {clip.label['si_pct']:.2f} | "
                    f"{'-' if g.symmetry_pct is None else f'{g.symmetry_pct:.2f}'} | "
                    f"{'-' if g.step_cv_pct is None else f'{g.step_cv_pct:.1f}'} |")
    tug_ok, errs = 0, []
    for seed in range(20):
        clip = make_tug(seed, px_per_m=160.0 if seed % 2 else 400.0)
        pipe = FramePipeline(cfg)
        steps = []
        for f in clip.frames:
            steps += pipe.push(f)
        steps += pipe.flush()
        r = segment_tug(samples_from_steps(steps), cfg.tug)
        if r.valid:
            tug_ok += 1
            errs.append(abs(r.tug_s - clip.label["tug_s"]))
    return ["## 2. Gait and TUG (synthetic)", "",
            "Calibrated with two floor markers 3 m apart. True speed and cadence are the generator settings.", "",
            "| Walk | measured speed (m/s) | cadence (spm) | true SI (%) | measured SI (%) | step CV (%) |",
            "|---|---|---|---|---|---|", *rows, "",
            f"TUG: all phases found on {tug_ok}/20 synthetic TUG clips; TUG time error mean "
            f"{np.mean(errs):.3f} s, max {np.max(errs):.3f} s." if errs else "TUG: no valid segmentation.", ""]


def throughput_section(cfg: Config, data_frames: list) -> list[str]:
    n = sum(len(fr) for fr in data_frames)
    t = time.perf_counter()
    for fr in data_frames:
        r = Runner(cfg, WallClock(datetime(2026, 1, 1).astimezone()))
        for f in fr:
            r.process(f)
        r.finish()
    fall_fps = n / (time.perf_counter() - t)
    t = time.perf_counter()
    for fr in data_frames:
        r = Runner(cfg, WallClock(datetime(2026, 1, 1).astimezone()), gait=GaitMonitor(cfg))
        for f in fr:
            r.process(f)
        r.finish()
    gait_fps = n / (time.perf_counter() - t)
    return ["## 3. Throughput (one CPU core, this machine)", "",
            f"{n} frames: filters + features + fall machine {fall_fps:,.0f} frames/s; "
            f"with gait/TUG/risk monitoring {gait_fps:,.0f} frames/s. Target ≥ 2,000 frames/s; a camera "
            "delivers 30.", ""]


def real_section(folder: Path, cfg: Config) -> list[str]:
    clips = load_clips([folder], cfg, labels_from_filenames=True)
    if not clips:
        return ["## 4. Real recordings", "", f"No clips found in `{folder}`.", ""]
    _, res = evaluate(clips, cfg.fall, cfg.eval.match_window_s)
    lines = ["## 4. Real recordings", "",
             f"{len(clips)} clips from `{folder}`, labelled from file names (fall_* expects FALL_CONFIRMED, "
             "lie_slow_stay_* expects PROLONGED_DOWN, others expect nothing). No fall start times, so delay is "
             "not scored.", ""]
    lines += _confusion("All events", summarize(res, clips))
    lines += ["| Clip | expected | events |", "|---|---|---|"]
    lines += [f"| {r.name} | {r.expected or '-'} | {' '.join(k for k, _ in r.alerts) or '-'} |" for r in res]
    return lines + ["", "### Margins", "", format_margins(margins(clips, cfg.fall), cfg.fall), ""]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m logic.make_results", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", default="data/real", help="folder of real clips (used if it exists)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    head = ["# SafeStride results", "",
            f"Generated by `python -m logic.make_results` at {datetime.now().astimezone():%Y-%m-%d %H:%M %z} "
            f"from commit `{_commit()}`. Every number below is measured by the script; nothing is typed by hand.",
            "", "Synthetic data is generated by a 2D stick-figure model and is much easier than real video, so "
            "these numbers show the logic works as designed, not real-world accuracy. Real recordings (section 4) "
            "matter more. SafeStride is a screening aid, not a medical device.", ""]
    syn, data = synthetic_section(cfg, args.n, args.seed)
    frames = [c.frames for c in generate_dataset(min(args.n, 40), args.seed + 1)]
    body = head + syn + gait_section(cfg) + throughput_section(cfg, frames)
    real = Path(args.real)
    if real.is_dir():
        body += real_section(real, cfg)
    else:
        body += ["## 4. Real recordings", "", f"Not available yet (no `{real}` folder).", ""]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
