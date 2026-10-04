"""Per-turn speaker identification: mic ring buffer and face/voice fusion rules."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from bridge.runtime.common import resample_audio_chunk
from bridge.runtime.wakeword import _to_mono

VOICE_RATE = 16000

VOICE_MATCH = 0.50  # accept a voice-only identification
VOICE_STRONG = 0.60  # voice may overrule the face in front of the camera
VOICE_DISAGREE = 0.40  # below this the voice does not belong to the profile
FACE_LEARN = 0.50  # face confidence needed to teach the voice profile
LEARN_CONFLICT = 0.55  # skip learning when another voice profile matches this well
MATURE_SAMPLES = 3  # voice profiles with fewer samples are still unreliable


class MicRingBuffer:
    """Keep the last few seconds of mic audio (16 kHz mono) with monotonic timestamps."""

    def __init__(self, max_seconds: float = 8.0) -> None:
        self._max_seconds = max_seconds
        self._chunks: deque[tuple[float, np.ndarray]] = deque()  # (end_ts, samples)
        self._lock = threading.Lock()

    def append(self, sample_rate: int, sample: np.ndarray, now: float) -> None:
        """Store one mic chunk that ended at `now`."""
        mono = resample_audio_chunk(_to_mono(sample), sample_rate, VOICE_RATE)
        with self._lock:
            self._chunks.append((now, mono))
            while self._chunks and now - self._chunks[0][0] > self._max_seconds:
                self._chunks.popleft()

    def window(self, start_ts: float, end_ts: float) -> np.ndarray:
        """Return audio recorded between `start_ts` and `end_ts`."""
        with self._lock:
            parts = [
                chunk for ts, chunk in self._chunks
                if start_ts <= ts - chunk.size / VOICE_RATE and ts <= end_ts
            ]
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)


@dataclass(frozen=True)
class SpeakerDecision:
    """Outcome of one turn's identification."""

    name: Optional[str]  # None keeps the current speaker
    source: str
    learn_voice_as: Optional[str] = None


def fuse_speaker(
    face_name: Optional[str],
    face_score: float,
    voice_scores: Optional[dict[str, float]],
    voice_counts: dict[str, int],
    unknown: str,
) -> SpeakerDecision:
    """Combine face and voice evidence into one speaker decision.

    `face_name` is None without camera, or `unknown` with score 0.0 when no face is in view.
    `voice_scores` is None when the utterance was too short to embed.
    """
    best_voice, best_voice_score = None, 0.0
    if voice_scores:
        best_voice = max(voice_scores, key=voice_scores.__getitem__)
        best_voice_score = voice_scores[best_voice]

    if face_name and face_name != unknown:
        if voice_scores is None:
            return SpeakerDecision(face_name, "face")
        own = voice_scores.get(face_name)
        others = {n: s for n, s in voice_scores.items() if n != face_name}
        other = max(others, key=others.__getitem__) if others else None
        other_score = others[other] if other else 0.0
        # Someone off-camera is talking while the person in view stays quiet.
        if (
            other
            and other_score >= VOICE_STRONG
            and voice_counts.get(other, 0) >= MATURE_SAMPLES
            and (own is None or own < VOICE_DISAGREE)
        ):
            return SpeakerDecision(other, "voice")
        mature = voice_counts.get(face_name, 0) >= MATURE_SAMPLES
        learn = (
            face_score >= FACE_LEARN
            and other_score < LEARN_CONFLICT
            and not (mature and own is not None and own < VOICE_DISAGREE)
        )
        return SpeakerDecision(face_name, "face", face_name if learn else None)

    if best_voice and best_voice_score >= VOICE_MATCH:
        return SpeakerDecision(best_voice, "voice")
    if face_name == unknown and face_score > 0.0:
        return SpeakerDecision(unknown, "face")
    return SpeakerDecision(None, "none")
