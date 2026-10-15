"""Per-frame glue: person selection -> keypoint cleaning -> features."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from logic.config import Config
from logic.features import Features, FeatureTracker
from logic.filters import CleanFrame, KeypointCleaner
from logic.io import Frame, TrackState, select_person

logger = logging.getLogger(__name__)


@dataclass
class Step:
    """One processed frame: cleaned keypoints and their features."""

    clean: CleanFrame
    features: Features


class FramePipeline:
    """Streaming front half of the decision layer for one followed person.

    Output lags input by up to ``max_gap_frames`` frames (gap repair needs the
    next valid sample); call ``flush`` at the end of a stream.
    """

    def __init__(self, cfg: Config) -> None:
        self.track = TrackState(c_min=cfg.filters.c_min, lost_grace_s=cfg.fall.lost_person_grace_s)
        self.cleaner = KeypointCleaner(cfg.filters)
        self.tracker = FeatureTracker(cfg.features)
        self._last_t: float | None = None

    def push(self, frame: Frame) -> list[Step]:
        """Process one input frame; return the frames that are now final."""
        if self._last_t is not None and frame.t <= self._last_t:
            logger.warning("t=%.3f: timestamp not increasing (last %.3f); frame skipped", frame.t, self._last_t)
            return []
        self._last_t = frame.t
        person = select_person(frame, self.track)
        return [Step(cf, self.tracker.update(cf)) for cf in self.cleaner.push(frame.t, person)]

    def flush(self) -> list[Step]:
        """Finish the stream."""
        return [Step(cf, self.tracker.update(cf)) for cf in self.cleaner.flush()]
