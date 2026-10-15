"""Tests for logic.compare_precision."""
import json

import numpy as np
import pytest

from logic.compare_precision import align, main, markdown, run
from logic.config import Config
from logic.io import Frame, Person, write_jsonl
from logic.synth import make_clip


def _perturb(frames, seed: int, sigma_px: float, drop_conf_frac: float, dt: float = 0.002):
    """Copy of ``frames`` with keypoint noise, some confidence drops, shifted timestamps, one frame lost."""
    rng = np.random.default_rng(seed)
    out = []
    for i, f in enumerate(frames):
        if i == 10:
            continue
        persons = []
        for p in f.persons:
            kp = {}
            for n, (x, y, c) in p.kp.items():
                if c > 0:
                    x, y = x + rng.normal(0, sigma_px), y + rng.normal(0, sigma_px)
                    if rng.random() < drop_conf_frac:
                        c = max(c - 0.3, 0.31)
                kp[n] = (x, y, c)
            persons.append(Person(kp=kp, bbox=p.bbox))
        out.append(Frame(t=f.t + dt, persons=persons))
    return out


def test_align_nearest_within_half_frame() -> None:
    a = [Frame(t=i / 30) for i in range(10)]
    b = [Frame(t=i / 30 + 0.004) for i in range(10) if i != 4] + [Frame(t=5.0)]
    pairs = align(a, b, 0.5)
    assert len(pairs) == 9 and all(abs(fa.t - fb.t) < 0.005 for fa, fb in pairs)
    assert align(a, [Frame(t=0.5 / 30 + 0.0)], 0.4) == []


def test_compare_known_perturbation(tmp_path) -> None:
    clip = make_clip("fall", "forward_stay", 5, "c")
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(), b.mkdir()
    write_jsonl(a / "c.jsonl", clip.frames)
    write_jsonl(b / "c.jsonl", _perturb(clip.frames, 1, sigma_px=4.0, drop_conf_frac=0.1))
    labels = {"c": clip.label}
    c = run(a, b, Config(), labels)
    assert c.aligned >= len(clip.frames) - 2
    # 4 px per axis at L ~ 200 px: Rayleigh mean = 0.02 * sqrt(pi / 2) ~ 0.025 L
    assert c.mean_error == pytest.approx(0.02 * np.sqrt(np.pi / 2), rel=0.15)
    assert 0.07 < c.conf_drop_frac < 0.13
    assert c.alerts_a == c.alerts_b == 1 and c.caught_a == c.caught_b == 1 == c.positives
    md = markdown(c)
    assert "| Fall alerts produced | 1 | 1 |" in md and "| Labelled falls caught | 1 / 1 | 1 / 1 |" in md
    assert md.count("| l_") == 8  # per-keypoint rows


def test_cli_single_files(tmp_path, capsys) -> None:
    clip = make_clip("sit_fast", "chair", 2, "s")
    fa, fb = tmp_path / "fp16.jsonl", tmp_path / "int8.jsonl"
    write_jsonl(fa, clip.frames)
    write_jsonl(fb, clip.frames)
    out = tmp_path / "r.md"
    assert main(["--a", str(fa), "--b", str(fb), "--out", str(out)]) == 0
    text = out.read_text()
    assert "| Mean keypoint error (/ L_ref) | 0 | 0.0000 |" in text
    assert "| Fall alerts produced | 0 | 0 |" in text and text in capsys.readouterr().out
