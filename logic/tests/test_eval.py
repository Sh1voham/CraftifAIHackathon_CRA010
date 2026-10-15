"""Tests for logic.eval, including the Tier 1 synthetic benchmark acceptance check."""
import json
import time

import pytest

from logic.config import Config, config_from_dict, load_config
from logic.eval import (
    ClipData, ClipResult, clip_features, evaluate, format_margins, label_from_name, load_clips, main, margins,
    run_detector, score_clip, split, summarize,
)
from logic.fall import Event
from logic.io import write_jsonl
from logic.synth import generate_dataset, make_clip, save_dataset


def _clip(name: str, expected: str | None, t_fall: float | None, cls: str = "fall", variant: str = "v",
          duration: float = 60.0) -> ClipData:
    label = {"class": cls, "variant": variant, "expected_event": expected, "t_fall_start": t_fall}
    return ClipData(name=name, label=label, feats=[], duration_s=duration, n_frames=0)


def _ev(t: float, kind: str = "FALL_CONFIRMED") -> Event:
    return Event(t, kind, 0.9, 80.0, 2.0, 3.0)


def test_score_clip_outcomes() -> None:
    assert score_clip(_clip("a", "FALL_CONFIRMED", 10.0), [_ev(13.5)], 20.0).outcome == "TP"
    r = score_clip(_clip("a", "FALL_CONFIRMED", 10.0), [_ev(13.5), _ev(13.6, "RECOVERED"), _ev(50.0)], 20.0)
    assert r.outcome == "TP" and r.delay_s == pytest.approx(3.5) and r.false_alarms == 1
    assert score_clip(_clip("a", "FALL_CONFIRMED", 10.0), [_ev(5.0)], 20.0).outcome == "FN"  # before the fall
    slow = score_clip(_clip("s", "PROLONGED_DOWN", 10.0), [_ev(24.0, "PROLONGED_DOWN")], 20.0)
    assert slow.outcome == "TP" and slow.alert_kind == "PROLONGED_DOWN"
    wrong = score_clip(_clip("f", "FALL_CONFIRMED", 10.0), [_ev(15.0, "PROLONGED_DOWN")], 20.0)
    assert wrong.outcome == "FN" and wrong.false_alarms == 1  # an unexpected kind is a false alarm
    anywhere = score_clip(_clip("r", "FALL_CONFIRMED", None), [_ev(99.0)], 20.0)
    assert anywhere.outcome == "TP" and anywhere.delay_s is None  # real clips: no start time
    neg = score_clip(_clip("n", None, None, "sit_fast"), [_ev(5.0, "PROLONGED_DOWN")], 20.0)
    assert neg.outcome == "FP" and neg.false_alarms == 1
    assert score_clip(_clip("n", None, None, "sit_fast"), [_ev(5.0, "RECOVERED")], 20.0).outcome == "TN"


def test_summarize_metrics() -> None:
    clips = [_clip(str(i), None, None, duration=360.0) for i in range(10)]
    results = [ClipResult(str(i), "c", "v", None, None, o, false_alarms=int(o == "FP"))
               for i, o in enumerate(["TP"] * 6 + ["FN", "FP", "TN", "TN"])]
    results[0].alert_kind, results[0].delay_s = "FALL_CONFIRMED", 3.2
    m = summarize(results, clips)
    assert (m.TP, m.FN, m.FP, m.TN) == (6, 1, 1, 2)
    assert m.precision == pytest.approx(6 / 7) and m.recall == pytest.approx(6 / 7)
    assert m.f1 == pytest.approx(6 / 7) and m.fa_per_hour == pytest.approx(1.0)
    assert m.delay("FALL_CONFIRMED") == (3.2, 3.2)


def test_split_is_stratified_and_deterministic() -> None:
    clips = [_clip(f"{c}{i}", None, None, c) for c in "abc" for i in range(10)]
    tr1, te1 = split(clips, 0.3, 0)
    tr2, te2 = split(clips, 0.3, 0)
    assert [c.name for c in te1] == [c.name for c in te2]
    assert sorted(c.name[0] for c in te1) == list("aaabbbccc")
    assert len(tr1) == 21 and not {c.name for c in tr1} & {c.name for c in te1}


def test_replay_is_deterministic() -> None:
    clip = make_clip("fall", "forward_recover", 3, "x")
    runs = [run_detector(clip_features(clip.frames, Config()), Config().fall) for _ in range(2)]
    assert runs[0] == runs[1] and [e.kind for e in runs[0]] == ["FALL_CONFIRMED", "RECOVERED"]
    assert json.dumps([e.__dict__ for e in runs[0]]) == json.dumps([e.__dict__ for e in runs[1]])


@pytest.fixture(scope="module")
def benchmark():
    cfg = load_config()
    clips = generate_dataset(200, 0)
    data = [ClipData(c.name, c.label, clip_features(c.frames, cfg), c.frames[-1].t - c.frames[0].t,
                     len(c.frames)) for c in clips]
    return cfg, data


def test_tier1_synthetic_benchmark(benchmark) -> None:
    cfg, data = benchmark
    assert len(data) >= 200
    n_fast = sum(1 for c in data if c.label["expected_event"] == "FALL_CONFIRMED")
    assert 0.4 <= n_fast / len(data) <= 0.5
    train, test = split(data, cfg.eval.test_fraction, cfg.eval.split_seed)
    m, results = evaluate(test, cfg.fall, cfg.eval.match_window_s)
    assert m.FP == 0 and m.false_alarms == 0
    fast = summarize(results, test, "FALL_CONFIRMED")
    assert fast.recall >= 0.95 and fast.FP == 0
    mean_delay, _ = fast.delay("FALL_CONFIRMED")
    assert 3.0 <= mean_delay <= 3.6
    slow = summarize(results, test, "PROLONGED_DOWN")
    assert slow.TP >= 1 and slow.FN == 0 and slow.FP == 0
    by_class = {}
    _, all_results = evaluate(data, cfg.fall, cfg.eval.match_window_s)
    for r in all_results:
        by_class.setdefault(r.cls, set()).update(k for k, _ in r.alerts)
    assert by_class["lie_slow_stay"] == {"PROLONGED_DOWN"}
    assert by_class["lie_slow_recover"] == set() and by_class["lie_quick_recover"] == set()


def test_detector_throughput_at_least_2000_fps(benchmark) -> None:
    cfg, data = benchmark
    clips = generate_dataset(10, 1)
    frames = [f for c in clips for f in c.frames]
    t = time.perf_counter()
    for c in clips:
        run_detector(clip_features(c.frames, cfg), cfg.fall)
    fps = len(frames) / (time.perf_counter() - t)
    assert fps >= 2000, f"{fps:.0f} frames/s"


def test_cli_eval_and_tune(tmp_path, capsys) -> None:
    save_dataset(generate_dataset(30, 2), tmp_path / "d")
    cfg_path = tmp_path / "c.json"
    cfg_path.write_text(json.dumps({"eval": {"grid_V_DROP": [1.2, 1.5], "grid_PHI_DOWN": [60.0],
                                             "grid_H_LOW": [0.5], "grid_T_HOLD": [3.0, 3.5]}}))
    tuned = tmp_path / "tuned.json"
    assert main(["--data", str(tmp_path / "d"), "--config", str(cfg_path), "--tune", "--tuned-out", str(tuned),
                 "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "grid search: 4 combinations" in out and "[test | all events]" in out and "[test | FALL_CONFIRMED]" in out
    best = config_from_dict(json.loads(tuned.read_text()))
    assert (best.fall.V_DROP, best.fall.T_HOLD) == (1.2, 3.0)  # tie -> closest to defaults
    assert (tmp_path / "out" / "eval_results.csv").read_text().count("\n") == 31
    assert len(load_clips([tmp_path / "d"], Config())) == 30


def test_label_from_name() -> None:
    assert label_from_name("fall_01")["expected_event"] == "FALL_CONFIRMED"
    assert label_from_name("Fall-forward_12")["expected_event"] == "FALL_CONFIRMED"
    assert label_from_name("lie_slow_stay_03")["expected_event"] == "PROLONGED_DOWN"
    for name in ("lie_slow_recover_02", "sit_01", "bend_04", "walk_02", "tug_01"):
        assert label_from_name(name)["expected_event"] is None
    lab = label_from_name("lie_slow_stay_03")
    assert lab["class"] == "lie_slow_stay" and lab["t_fall_start"] is None


def test_margins_report(benchmark) -> None:
    cfg, data = benchmark
    rows = margins(data[:40], cfg.fall)
    falls = [r for r in rows if r.cls == "fall"]
    sits = [r for r in rows if r.cls == "sit_fast"]
    assert falls and sits
    assert min(r.v_ratio for r in falls) > 1.0 and min(r.down_s for r in falls) > cfg.fall.T_HOLD
    assert max(r.phi_max for r in sits) < cfg.fall.PHI_DOWN
    text = format_margins(rows, cfg.fall)
    assert "| fall |" in text and "peak v / V_DROP" in text


def test_cli_real_clips_from_filenames(tmp_path, capsys) -> None:
    d = tmp_path / "real"
    d.mkdir()
    for name, (kind, variant, seed) in {"fall_01": ("fall", "forward_stay", 1), "sit_01": ("sit_fast", "chair", 2),
                                        "lie_slow_stay_01": ("lie_slow_stay", "slow", 3)}.items():
        write_jsonl(d / f"{name}.jsonl", make_clip(kind, variant, seed, name).frames)
    tuned = tmp_path / "tuned.json"
    assert main(["--data", str(d), "--labels-from-filenames", "--tune", "--tuned-out", str(tuned),
                 "--out", str(tmp_path / "out")]) == 0
    out = capsys.readouterr().out
    assert "3 real clips" in out and "WARNING: only 3 real clips" in out and not tuned.exists()
    assert "| fall_01 | fall | FALL_CONFIRMED |" in out and "longest down" in out
    assert "recall=1.000" in out
