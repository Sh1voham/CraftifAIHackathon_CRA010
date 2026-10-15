"""Privacy blur: any frame that leaves memory is blurred first.

The default product mode stores keypoints only; frames are saved (blurred)
only when an image source is available, e.g. as fall evidence.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from logic.config import PrivacyCfg
from logic.io import Person

logger = logging.getLogger(__name__)

Box = tuple[float, float, float, float]


def person_box(subject: Any, c_min: float = 0.3) -> Box | None:
    """Bounding box (x1, y1, x2, y2) of a person.

    ``subject`` may be a Person (union of its valid keypoints and its bbox), a
    dict of name -> (x, y, c), a sequence of keypoints (x, y, c[, ...]) or None
    entries, or a bbox of four numbers.
    """
    pts: list[tuple[float, float]] = []
    boxes: list[Box] = []
    if isinstance(subject, Person):
        pts = [(k[0], k[1]) for k in subject.kp.values() if k[2] >= c_min]
        if subject.bbox is not None:
            boxes.append(subject.bbox)
    elif isinstance(subject, dict):
        pts = [(k[0], k[1]) for k in subject.values() if k is not None and k[2] >= c_min]
    elif subject is not None and len(subject) == 4 and all(isinstance(v, (int, float)) for v in subject):
        boxes.append(tuple(float(v) for v in subject))  # type: ignore[arg-type]
    elif subject is not None:
        pts = [(k[0], k[1]) for k in subject if k is not None and k[2] >= c_min]
    for b in boxes:
        pts += [(b[0], b[1]), (b[2], b[3])]
    if not pts:
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _blur_region(img: np.ndarray, box: Box | None, cfg: PrivacyCfg) -> None:
    """Blur ``box`` (expanded) in place; blur everything when ``box`` is None."""
    k = cfg.kernel if cfg.kernel % 2 else cfg.kernel + 1
    h, w = img.shape[:2]
    if box is None:
        img[:] = cv2.GaussianBlur(img, (k, k), cfg.sigma)
        return
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * cfg.expand_frac, (y2 - y1) * cfg.expand_frac
    c1, r1 = max(int(np.floor(x1 - dx)), 0), max(int(np.floor(y1 - dy)), 0)
    c2, r2 = min(int(np.ceil(x2 + dx)) + 1, w), min(int(np.ceil(y2 + dy)) + 1, h)
    if c2 <= c1 or r2 <= r1:
        return
    img[r1:r2, c1:c2] = cv2.GaussianBlur(img[r1:r2, c1:c2], (k, k), cfg.sigma)


def blur_person(image: np.ndarray, keypoints_or_bbox: Any, cfg: PrivacyCfg | None = None,
                c_min: float = 0.3) -> np.ndarray:
    """Return a copy of ``image`` with the person's (expanded) box blurred.

    If no valid keypoints or box are given, the whole frame is blurred.
    """
    out = image.copy()
    _blur_region(out, person_box(keypoints_or_bbox, c_min), cfg or PrivacyCfg())
    return out


def blur_people(image: np.ndarray, persons: Iterable[Any], cfg: PrivacyCfg | None = None,
                c_min: float = 0.3) -> np.ndarray:
    """Copy of ``image`` with every person blurred; the whole frame if none can be located."""
    cfg = cfg or PrivacyCfg()
    out = image.copy()
    boxes = [b for b in (person_box(p, c_min) for p in persons) if b is not None]
    if boxes:
        for b in boxes:
            _blur_region(out, b, cfg)
    else:
        _blur_region(out, None, cfg)
    return out


def save_blurred(image: np.ndarray, path: str | Path, persons: Iterable[Any], cfg: PrivacyCfg | None = None,
                 c_min: float = 0.3) -> Path:
    """Blur every person in ``persons`` and write only the blurred image.

    With no locatable person the whole frame is blurred before writing.
    """
    out = blur_people(image, persons, cfg, c_min)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".jpg", out)
    if not ok:
        raise OSError(f"could not encode image for {p}")
    p.write_bytes(buf.tobytes())
    return p
