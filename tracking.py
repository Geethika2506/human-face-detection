"""Associate faces across frames and smooth identity labels.

Tracks are matched by bounding-box IoU. A display name is shown only after the
same person id is known on that track for ``smooth_window`` consecutive
updates. Unknown, uncertain, a different person id, or a lost track clears the
name immediately so it cannot follow someone else.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Observation:
    bbox: tuple[int, int, int, int]
    status: str
    person_id: Optional[str] = None
    display_name: Optional[str] = None
    match_score: Optional[float] = None


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    history: deque = field(default_factory=deque)
    missed: int = 0
    label: str = "Unknown"
    score_text: str = ""

    def note(self, observation: Observation, smooth_window: int) -> None:
        previous_id = _last_known_id(self.history)
        identity_changed = (
            observation.status == "known"
            and previous_id is not None
            and observation.person_id != previous_id
        )
        if observation.status != "known" or identity_changed:
            self.history.clear()
        self.history.append(observation)
        self.bbox = observation.bbox
        self.missed = 0
        self.label, self.score_text = label_for(self.history, smooth_window)


class Tracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        smooth_window: int = 3,
        max_missed: int = 8,
    ):
        self.iou_threshold = iou_threshold
        self.smooth_window = smooth_window
        self.max_missed = max_missed
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, observations: list[Observation]) -> list[Track]:
        pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self.tracks):
            for obs_index, observation in enumerate(observations):
                overlap = iou(track.bbox, observation.bbox)
                if overlap >= self.iou_threshold:
                    pairs.append((overlap, track_index, obs_index))
        pairs.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_observations: set[int] = set()
        for _overlap, track_index, obs_index in pairs:
            if track_index in matched_tracks or obs_index in matched_observations:
                continue
            matched_tracks.add(track_index)
            matched_observations.add(obs_index)
            self.tracks[track_index].note(observations[obs_index], self.smooth_window)

        for track_index, track in enumerate(self.tracks):
            if track_index not in matched_tracks:
                track.missed += 1
        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]

        for obs_index, observation in enumerate(observations):
            if obs_index in matched_observations:
                continue
            track = Track(
                track_id=self._next_id,
                bbox=observation.bbox,
                history=deque(maxlen=self.smooth_window),
            )
            self._next_id += 1
            track.note(observation, self.smooth_window)
            self.tracks.append(track)
        return list(self.tracks)

    def reset(self) -> None:
        self.tracks.clear()


def label_for(history: deque, smooth_window: int) -> tuple[str, str]:
    if not history:
        return "Unknown", ""
    latest = history[-1]
    score_text = "" if latest.match_score is None else f"score {latest.match_score:.3f}"
    if latest.status == "uncertain":
        return "Uncertain", score_text
    if latest.status != "known":
        return "Unknown", score_text
    recent = list(history)[-smooth_window:]
    confirmed = len(recent) >= smooth_window and all(
        item.status == "known" and item.person_id == latest.person_id for item in recent
    )
    if not confirmed:
        return "Unknown", score_text
    return latest.display_name or "Unknown", score_text


def iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _last_known_id(history: deque) -> Optional[str]:
    for item in reversed(history):
        if item.status == "known" and item.person_id:
            return item.person_id
    return None
