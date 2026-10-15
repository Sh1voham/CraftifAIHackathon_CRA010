"""Tests for logic.privacy."""
import cv2
import numpy as np
import pytest

import logic.privacy as privacy
from logic.config import PrivacyCfg
from logic.io import Person
from logic.privacy import blur_person, person_box, save_blurred


@pytest.fixture()
def noise() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, size=(400, 600, 3), dtype=np.uint8)


def _std(img: np.ndarray, r1: int, r2: int, c1: int, c2: int) -> float:
    return float(img[r1:r2, c1:c2].astype(float).std())


KP = {"nose": (200.0, 100.0, 0.9), "l_ankle": (300.0, 300.0, 0.9), "r_wrist": (580.0, 20.0, 0.1)}


def test_person_box_sources() -> None:
    assert person_box(KP) == (200.0, 100.0, 300.0, 300.0)  # low-confidence wrist ignored
    assert person_box((1, 2, 3, 4)) == (1.0, 2.0, 3.0, 4.0)
    assert person_box([(5.0, 6.0, 0.9, False), None, (7.0, 9.0, 0.8, True)]) == (5.0, 6.0, 7.0, 9.0)
    assert person_box(Person(kp=KP, bbox=(150.0, 90.0, 250.0, 280.0))) == (150.0, 90.0, 300.0, 300.0)
    assert person_box({"nose": (1.0, 1.0, 0.1)}) is None


def test_blur_person_blurs_expanded_box_only(noise) -> None:
    before = noise.copy()
    out = blur_person(noise, KP)
    assert np.array_equal(noise, before)  # input untouched
    assert _std(out, 110, 290, 210, 290) < 0.25 * _std(noise, 110, 290, 210, 290)  # inside: blurred
    # 15 % expansion each side: x from 185 to 315, y from 70 to 330
    assert _std(out, 75, 95, 188, 198) < 0.5 * _std(noise, 75, 95, 188, 198)
    assert np.array_equal(out[:, 330:], noise[:, 330:])  # well outside: untouched
    assert np.array_equal(out[340:], noise[340:])


def test_blur_whole_frame_when_person_unknown(noise) -> None:
    out = blur_person(noise, {"nose": (1.0, 1.0, 0.0)})
    assert _std(out, 0, 400, 0, 600) < 0.25 * _std(noise, 0, 400, 0, 600)


def test_save_blurred_never_writes_original(noise, tmp_path, monkeypatch) -> None:
    written = []
    real = cv2.imencode

    def spy(ext, img, *a):
        written.append(img.copy())
        return real(ext, img, *a)

    monkeypatch.setattr(privacy.cv2, "imencode", spy)
    path = save_blurred(noise, tmp_path / "ev" / "frame.png", [KP, (400, 300, 500, 380)])
    assert len(written) == 1
    back = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(back, written[0])
    assert not np.array_equal(back, noise)
    for r1, r2, c1, c2 in [(110, 290, 210, 290), (310, 370, 410, 490)]:  # both persons blurred
        assert _std(back, r1, r2, c1, c2) < 0.25 * _std(noise, r1, r2, c1, c2)


def test_save_blurred_with_no_persons_blurs_everything(noise, tmp_path) -> None:
    path = save_blurred(noise, tmp_path / "f.png", [], PrivacyCfg(kernel=50))  # even kernel is bumped
    back = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert _std(back, 0, 400, 0, 600) < 0.25 * _std(noise, 0, 400, 0, 600)
