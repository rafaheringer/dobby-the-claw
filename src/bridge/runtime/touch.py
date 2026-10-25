"""Detect petting and taps on Dobby's head from the mic array level.

Petting right on the array saturates it (100 ms windows near -10 dBFS, clipped samples) while speech
at a normal distance stays below about -22 dBFS; taps are short transients. Speech close to the head
reaches the same level, so a voice activity detector masks it. Pure logic: the
orchestrator feeds mic chunks and acts on the returned events.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

from bridge.runtime.common import resample_audio_chunk
from bridge.runtime.wakeword import _to_mono

_WINDOW_S = 0.1
_PET_SPAN_S = 1.5  # loud windows are counted within this span
_PET_MIN_WINDOWS = 3
_PET_SUSTAIN_WINDOWS = 3  # this many consecutive loud windows: a rubbing hand, never a tap
_PET_SETTLE_S = 1.0  # sparse loud windows are petting once they spread this long without a burst
_TAP_BURST_WINDOWS = 5  # this many loud windows within the span, none sustained: tapping
_PET_END_S = 2.0  # petting ends after this long without a loud window
_TAP_HOLD_S = 1.0  # taps are emitted after this long without a hit (petting may still show up)
_TAP_AFTER_PET_S = 3.0  # stray hits right after petting are its tail, not taps
_SPEECH_PROB = 0.5  # VAD probability above which a window is someone talking, not a touch
_SPEECH_MASK_S = 0.5  # windows this long after speech are masked too (syllable gaps, plosives)


class TouchKind(str, Enum):
    """Kinds of touch events."""

    PET_START = "pet_start"
    PET_END = "pet_end"
    TAP = "tap"


@dataclass(frozen=True)
class TouchEvent:
    """One detected touch; `duration_s` is set on PET_END, `interrupted` when deafness ended it."""

    kind: TouchKind
    at: float
    duration_s: float = 0.0
    interrupted: bool = False


class SpeechVad:
    """Streaming Silero VAD (bundled with faster-whisper): speech probability of mic audio."""

    _RATE = 16000
    _FRAME = 512
    _CONTEXT = 64

    def __init__(self) -> None:
        """Load the model (a few MB, CPU)."""
        from faster_whisper.vad import get_vad_model

        self._session = get_vad_model().session
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(self._CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._last = 0.0

    def __call__(self, audio: np.ndarray, sample_rate: int) -> float:
        """Feed mono audio; highest probability over the frames it completes (else the last one)."""
        audio = np.asarray(audio, dtype=np.float32)
        if sample_rate != self._RATE:
            audio = resample_audio_chunk(audio, sample_rate, self._RATE)
        self._pending = np.concatenate([self._pending, audio])
        best: float | None = None
        while len(self._pending) >= self._FRAME:
            frame, self._pending = self._pending[: self._FRAME], self._pending[self._FRAME :]
            batch = np.concatenate([self._context, frame])[None, :]
            out, self._h, self._c = self._session.run(
                None, {"input": batch, "h": self._h, "c": self._c}
            )
            self._context = frame[-self._CONTEXT :]
            prob = float(np.asarray(out).reshape(-1)[0])
            best = prob if best is None else max(best, prob)
        if best is not None:
            self._last = best
        return self._last


class TouchDetector:
    """Turn mic chunks into petting/tap events using 100 ms window level and peak."""

    def __init__(
        self,
        pet_dbfs: float = -18.0,
        tap_peak: float = 0.5,
        vad: Callable[[np.ndarray, int], float] | None = None,
    ) -> None:
        """Store thresholds: window RMS (dBFS) for petting, sample peak for taps; `vad` (speech
        probability of a window) masks someone talking close to the head."""
        self._pet_rms = 10 ** (pet_dbfs / 20)
        self._tap_peak = tap_peak
        self._vad = vad
        self._speech_until = float("-inf")
        self._buffer = np.zeros(0, dtype=np.float32)
        self._reset()

    @property
    def petting(self) -> bool:
        """Whether a petting episode is in progress."""
        return self._pet_started_at is not None

    def _reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._loud: deque[float] = deque()
        self._run = 0  # consecutive loud windows
        self._last_loud = float("-inf")
        self._pet_started_at: float | None = None
        self._pet_ended_at = float("-inf")
        self._last_hit: float | None = None  # pending tap(s)
        self._tapping = False

    def observe(
        self,
        sample_rate: int,
        sample: np.ndarray,
        now: float,
        deaf: bool = False,
        muted: bool = False,
    ) -> list[TouchEvent]:
        """Feed one mic chunk ending at `now`; return the events it completes.

        `deaf` (Dobby speaking) ends any episode and drops the audio; `muted` (a reaction sound
        playing) only drops the audio: an ongoing petting episode is assumed to continue.
        """
        if deaf:
            events = []
            if self._pet_started_at is not None:
                events.append(self._pet_end(now, interrupted=True))
            self._reset()
            return events
        if muted:
            self._buffer = np.zeros(0, dtype=np.float32)
            self._last_hit = None
            self._run = 0
            if self._pet_started_at is not None:
                self._last_loud = now
            return []
        self._buffer = np.concatenate([self._buffer, _to_mono(sample)])
        size = max(1, int(sample_rate * _WINDOW_S))
        count = len(self._buffer) // size
        events: list[TouchEvent] = []
        for i in range(count):
            window = self._buffer[i * size : (i + 1) * size]
            end = now - (len(self._buffer) - (i + 1) * size) / sample_rate
            events.extend(self._window(window, end, sample_rate))
        self._buffer = self._buffer[count * size :]
        return events

    def _window(self, window: np.ndarray, at: float, sample_rate: int) -> list[TouchEvent]:
        # Petting: sustained loud windows, or sparse ones spreading over a second. Taps: one or two
        # loud windows each with silence in between, ~5/s in a burst (or a lone hit).
        events: list[TouchEvent] = []
        if self._vad is not None and self._vad(window, sample_rate) >= _SPEECH_PROB:
            self._speech_until = at + _SPEECH_MASK_S
        talking = at < self._speech_until
        loud = not talking and float(np.sqrt(np.mean(window**2))) > self._pet_rms
        hit = loud or (not talking and float(np.max(np.abs(window))) >= self._tap_peak)
        self._run = self._run + 1 if loud else 0
        if loud:
            self._last_loud = at
            self._loud.append(at)
        while self._loud and at - self._loud[0] > _PET_SPAN_S:
            self._loud.popleft()

        if self._pet_started_at is not None:
            if at - self._last_loud >= _PET_END_S:
                events.append(self._pet_end(at))
                self._pet_started_at = None
                self._pet_ended_at = at
                self._loud.clear()
            return events

        if hit and at - self._pet_ended_at >= _TAP_AFTER_PET_S:
            self._last_hit = at
        if len(self._loud) >= _TAP_BURST_WINDOWS:
            self._tapping = True
        settled = len(self._loud) >= _PET_MIN_WINDOWS and at - self._loud[0] >= _PET_SETTLE_S
        if self._run >= _PET_SUSTAIN_WINDOWS or (settled and not self._tapping):
            self._pet_started_at = self._loud[0]
            self._last_hit = None
            self._tapping = False
            events.append(TouchEvent(TouchKind.PET_START, at))
        elif self._last_hit is not None and at - self._last_hit >= _TAP_HOLD_S:
            events.append(TouchEvent(TouchKind.TAP, at))
            self._last_hit = None
            self._tapping = False
            self._loud.clear()
        elif self._tapping and at - self._last_loud >= _TAP_HOLD_S:
            self._tapping = False
        return events

    def _pet_end(self, at: float, interrupted: bool = False) -> TouchEvent:
        assert self._pet_started_at is not None
        duration = at - self._pet_started_at
        return TouchEvent(TouchKind.PET_END, at, duration_s=duration, interrupted=interrupted)


class TouchSpeechGate:
    """Decide when a touch earns a spoken reaction (first one, then once per cooldown)."""

    def __init__(self, cooldown_s: float = 300.0) -> None:
        """Store the minimum gap between spoken reactions of the same kind."""
        self._cooldown_s = cooldown_s
        self._last: dict[TouchKind, float] = {}

    def notice(self, event: TouchEvent, now: float) -> str | None:
        """Return the notice text for the model when `event` should be spoken about, else None."""
        if event.interrupted:
            return None
        if event.kind is TouchKind.PET_END:
            text = f"[toque] carinho de {round(event.duration_s)} s terminou"
        elif event.kind is TouchKind.TAP:
            text = "[toque] levou batidinhas na cabeça"
        else:
            return None
        last = self._last.get(event.kind)
        if last is not None and now - last < self._cooldown_s:
            return None
        self._last[event.kind] = now
        return text
