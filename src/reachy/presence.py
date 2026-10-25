"""Person tracks and gaze choice built from per-frame person detections.

Pure logic (no camera, no model) so it can be unit-tested. Boxes are normalized
image coordinates in [0, 1], x to the right and y down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

# Track lifecycle
CONFIRM_S = 0.8  # a detection must persist this long before it counts as a person
CONFIRM_HITS = 3
LOST_S = 4.0  # distant detections flicker, so keep tracks alive through short gaps
MATCH_MAX_CENTER_DIST = 0.25  # the head moves, so boxes can jump between frames
# Chairs and coats get weak "person" scores; without a face seen, require a strong steady score.
STRONG_PERSON_SCORE = 0.6
PET_STRONG_SCORE = 0.55  # cats and dogs have no face check, but the detector rarely invents them
SCORE_SMOOTHING = 0.3

# Camera: Reachy Mini wide-angle lens, focal length / image width (K: fx≈550 px at 1280 px).
FOCAL_OVER_WIDTH = 0.43


@dataclass(frozen=True)
class Box:
    """Normalized bounding box."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    def iou(self, other: "Box") -> float:
        ix = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        iy = max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))
        inter = ix * iy
        union = self.width * self.height + other.width * other.height - inter
        return inter / union if union > 0 else 0.0

    def head_point(self, aspect: float = 16 / 9) -> tuple[float, float]:
        """Estimate where the head is: near the top, lower for upper-body (wide) boxes.

        `aspect` is the image width/height in pixels, since x and y are normalized separately.
        A box cut by the bottom of the image (someone close) says nothing about the head through
        its height, so only its width (the shoulders) is used then.
        """
        head_size = 0.5 * self.width * aspect  # in y units
        if self.y1 < 0.97:
            head_size = min(head_size, 0.3 * self.height)
        return self.center_x, self.y0 + head_size / 2


@dataclass
class Track:
    """One person followed across frames."""

    id: int
    box: Box
    first_seen: float
    last_seen: float
    hits: int = 1
    confirmed_at: Optional[float] = None
    name: Optional[str] = None  # set by face recognition (may be the "unknown" label)
    name_score: float = 0.0
    id_attempts: int = 0
    last_id_attempt: float = 0.0
    score: float = 0.0  # smoothed detector confidence
    face_seen: bool = False  # a face was found in the head region at least once
    observations: list = field(default_factory=list)  # recent (name, similarity) face looks, newest last

    @property
    def confirmed(self) -> bool:
        return self.confirmed_at is not None



class PersonTracker:
    """Associate per-frame boxes of one kind (people, cats...) into stable tracks."""

    def __init__(self, strong_score: float = STRONG_PERSON_SCORE, first_id: int = 1) -> None:
        self._tracks: list[Track] = []
        self._next_id = first_id
        self._strong_score = strong_score

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def confirmed(self) -> list[Track]:
        return [t for t in self._tracks if t.confirmed]

    def is_verified(self, track: Track) -> bool:
        """Confirmed and believable: a face was seen or the detector is consistently sure."""
        return track.confirmed and (track.face_seen or track.score >= self._strong_score)

    def verified(self) -> list[Track]:
        return [t for t in self._tracks if self.is_verified(t)]

    def update(self, boxes: list[Box], now: float, scores: Optional[list[float]] = None) -> list[Track]:
        """Fold one frame of detections in and return the confirmed tracks."""
        scores = scores if scores is not None else [1.0] * len(boxes)
        unmatched = list(range(len(boxes)))
        pairs = []
        for t_index, track in enumerate(self._tracks):
            for b_index, box in enumerate(boxes):
                iou = track.box.iou(box)
                dist = math.hypot(track.box.center_x - box.center_x, track.box.center_y - box.center_y)
                if iou > 0.1 or dist < MATCH_MAX_CENTER_DIST:
                    pairs.append((iou - dist, t_index, b_index))
        used_tracks: set[int] = set()
        for _, t_index, b_index in sorted(pairs, reverse=True):
            if t_index in used_tracks or b_index not in unmatched:
                continue
            used_tracks.add(t_index)
            unmatched.remove(b_index)
            track = self._tracks[t_index]
            track.box = boxes[b_index]
            track.score += SCORE_SMOOTHING * (scores[b_index] - track.score)
            track.last_seen = now
            track.hits += 1
            if not track.confirmed and track.hits >= CONFIRM_HITS and now - track.first_seen >= CONFIRM_S:
                track.confirmed_at = now

        for b_index in unmatched:
            self._tracks.append(
                Track(id=self._next_id, box=boxes[b_index], first_seen=now, last_seen=now, score=scores[b_index])
            )
            self._next_id += 1

        self._tracks = [
            t for t in self._tracks
            if now - t.last_seen < (LOST_S if t.confirmed else 1.0)
        ]
        return self.confirmed()


def bearing_to_image_x(target_yaw: float, head_yaw: float) -> float:
    """Map a body-frame yaw (positive = robot's left) to a normalized image x for the current head yaw."""
    delta = float(target_yaw - head_yaw)
    delta = max(-1.4, min(1.4, delta))
    return 0.5 - FOCAL_OVER_WIDTH * math.tan(delta)


class GazeSelector:
    """Decide which person to look at: whoever is talking, else the nearest person."""

    def __init__(self, newcomer_s: float = 3.0, nearer_ratio: float = 1.25) -> None:
        self._newcomer_s = newcomer_s
        self._nearer_ratio = nearer_ratio  # how much taller another box must be to switch to it
        self._target_id: Optional[int] = None

    @property
    def target_id(self) -> Optional[int]:
        return self._target_id

    def choose(
        self,
        tracks: list[Track],
        now: float,
        speaker_name: Optional[str] = None,
        speech_x: Optional[float] = None,
    ) -> Optional[Track]:
        """Return the track to look at.

        Priority: whoever is talking (by voice direction, then by recognized name), a newcomer for a
        brief greeting look, then the nearest person (tallest box), kept until someone is clearly nearer.
        """
        if not tracks:
            self._target_id = None
            return None
        if len(tracks) == 1:
            return self._set(tracks[0])

        if speech_x is not None:
            nearest = min(tracks, key=lambda t: abs(t.box.center_x - speech_x))
            if abs(nearest.box.center_x - speech_x) < 0.35:
                return self._set(nearest)
        if speaker_name:
            named = [t for t in tracks if t.name == speaker_name]
            if len(named) == 1:
                return self._set(named[0])

        newcomers = [t for t in tracks if t.confirmed_at is not None and now - t.confirmed_at < self._newcomer_s]
        if newcomers:
            return self._set(max(newcomers, key=lambda t: t.confirmed_at or 0.0))

        closest = max(tracks, key=lambda t: t.box.height)
        current = next((t for t in tracks if t.id == self._target_id), None)
        if current is not None and closest.box.height < current.box.height * self._nearer_ratio:
            return current
        return self._set(closest)

    def _set(self, track: Track) -> Track:
        self._target_id = track.id
        return track


@dataclass(frozen=True)
class EnrollChoice:
    """Which person an enrollment is about, or why it can't be decided."""

    track: Optional[Track] = None
    error: str = ""  # "nobody", "no_other", "ambiguous" or "bad_number"


def choose_enroll_track(
    people: list[Track],
    want_speaker: bool,
    speaker_track_id: Optional[int],
    is_known: Callable[[Track], bool],
    track_id: Optional[int] = None,
) -> EnrollChoice:
    """Pick the person to enroll among the people in view.

    `want_speaker` enrolls whoever is talking ("eu sou a Ana"); otherwise someone else ("ela é a Ana"),
    which excludes the speaker and people already recognized. `track_id` is an explicit choice.
    """
    if not people:
        return EnrollChoice(error="nobody")
    if track_id is not None:
        chosen = next((t for t in people if t.id == track_id), None)
        return EnrollChoice(chosen, "" if chosen else "bad_number")
    speaker = next((t for t in people if t.id == speaker_track_id), None)
    if want_speaker:
        if speaker is not None:
            return EnrollChoice(speaker)
        if len(people) == 1:
            return EnrollChoice(people[0])
        return EnrollChoice(error="ambiguous")
    others = [t for t in people if t is not speaker and not is_known(t)]
    if len(others) == 1:
        return EnrollChoice(others[0])
    return EnrollChoice(error="ambiguous" if others else "no_other")
