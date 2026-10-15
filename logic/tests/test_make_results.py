"""Tests for logic.make_results (small run)."""
from logic.io import write_jsonl
from logic.make_results import main
from logic.synth import make_clip


def test_make_results_writes_measured_report(tmp_path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    write_jsonl(real / "fall_01.jsonl", make_clip("fall", "forward_stay", 1, "f").frames)
    write_jsonl(real / "sit_01.jsonl", make_clip("sit_fast", "chair", 2, "s").frames)
    out = tmp_path / "RESULTS.md"
    assert main(["--n", "24", "--real", str(real), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    for heading in ("## 1. Synthetic fall benchmark", "## 2. Gait and TUG", "## 3. Throughput",
                    "## 4. Real recordings", "Fast falls (FALL_CONFIRMED)", "Margins"):
        assert heading in text
    assert "| fall_01 | FALL_CONFIRMED | FALL_CONFIRMED |" in text and "| sit_01 | - | - |" in text
    assert "nothing is typed by hand" in text
