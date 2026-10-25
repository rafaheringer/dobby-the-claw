"""Detect petting and taps on Dobby's head from the mic array level.

Petting right on the array saturates it (100 ms windows near -10 dBFS, clipped samples) while speech
at a normal distance stays below about -22 dBFS; taps are short transients. Speech close to the head
reaches the same level, so a voice activity detector masks it; loud hiss (mostly above 3 kHz) is not
a touch either: rubbing or knocking on the shell stays under 1 kHz. Contact is also far louder than
the room: a window counts only when it stands well above the recent noise floor (a TV or the shower
keep the room near -25 dBFS, and doors or footsteps rise only 7-13 dB above it). Pure logic: the
orchestrator feeds mic chunks and acts on the returned events.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
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
_PET_END_S = 2.0  # petting ends after this long without a loud window
_TAP_HOLD_S = 1.0  # taps are emitted after this long without a hit (petting may still show up)
_TAP_MIN_KNOCKS = 3  # taps come in series; one or two knocks are a chime or a bump on the table
_TAP_MAX_GAP_S = 0.6  # knocks further apart are separate series (measured 0.2-0.5 s)
_TAP_CADENCE = 2.0  # every gap within this factor of the median gap: steady, unlike typing
_KNOCK_FRAME_S = 0.005
_KNOCK_MAX_WIDTH_S = 0.025  # a knock stays within 10 dB of its peak ~20 ms; a stroke ~35 ms
_TAP_SHARP_SHARE = 0.75  # taps: nearly every peak of the series is a knock (strokes: ~40%)
_TAP_AFTER_PET_S = 3.0  # stray hits right after petting are its tail, not taps
_SPEECH_PROB = 0.5  # VAD probability above which a window is someone talking, not a touch
_SPEECH_MASK_S = 0.5  # windows this long after speech are masked too (syllable gaps, plosives)
_HISS_HZ = 3000.0
_HISS_SHARE = 0.4  # touch windows keep ~1% of their energy above _HISS_HZ, hiss bursts ~60%
_FLOOR_WINDOWS = 30  # noise floor: a low percentile of the last 3 s of windows
_FLOOR_PERCENTILE = 20
_ABOVE_FLOOR_DB = 15.0  # touches rise 25+ dB above a quiet room


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
        tap_dbfs: float = -15.0,
        vad: Callable[[np.ndarray, int], float] | None = None,
    ) -> None:
        """Store thresholds: window RMS (dBFS) for petting and for each knock of a tap series; `vad`
        (speech probability of a window) masks someone talking close to the head."""
        self._pet_rms = 10 ** (pet_dbfs / 20)
        self._tap_rms = 10 ** (tap_dbfs / 20)
        self._floor: deque[float] = deque(maxlen=_FLOOR_WINDOWS)
        self._vad = vad
        self._speech_until = float("-inf")
        self._buffer = np.zeros(0, dtype=np.float32)
        self._previous = np.zeros(0, dtype=np.float32)  # last window, context for knock shape
        self._reset()

    @property
    def petting(self) -> bool:
        """Whether a petting episode is in progress."""
        return self._pet_started_at is not None

    def _reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._loud: deque[float] = deque()
        self._strong: deque[tuple[float, bool]] = deque()  # (time, sharp) of windows above knock level
        self._run = 0  # consecutive loud windows
        self._last_loud = float("-inf")
        self._pet_started_at: float | None = None
        self._pet_ended_at = float("-inf")
        self._knocks: list[tuple[float, bool]] = []  # (onset, sharp) of the pending tap series
        self._was_hit = False

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
            self._knocks = []
            self._was_hit = False
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
        # Petting: sustained loud windows, or strong strokes spreading over a second. Taps: a steady
        # series of knocks (sharp peaks, silence in between, ~4/s).
        events: list[TouchEvent] = []
        if self._vad is not None and self._vad(window, sample_rate) >= _SPEECH_PROB:
            self._speech_until = at + _SPEECH_MASK_S
        talking = at < self._speech_until
        rms = float(np.sqrt(np.mean(window**2)))
        floor = float(np.percentile(self._floor, _FLOOR_PERCENTILE)) if self._floor else 0.0
        self._floor.append(rms)
        loud = rms > max(self._pet_rms, floor * 10 ** (_ABOVE_FLOOR_DB / 20))
        if talking or (loud and _hiss_share(window, sample_rate) > _HISS_SHARE):
            loud = False
        hit = loud and rms > self._tap_rms
        knock = hit and _peak_width_s(self._previous, window, sample_rate) <= _KNOCK_MAX_WIDTH_S
        self._previous = window
        self._run = self._run + 1 if loud else 0
        if loud:
            self._last_loud = at
            self._loud.append(at)
        if hit:
            self._strong.append((at, knock))
        while self._loud and at - self._loud[0] > _PET_SPAN_S:
            self._loud.popleft()
        while self._strong and at - self._strong[0][0] > _PET_SPAN_S:
            self._strong.popleft()

        if self._pet_started_at is not None:
            if at - self._last_loud >= _PET_END_S:
                events.append(self._pet_end(at))
                self._pet_started_at = None
                self._pet_ended_at = at
                self._loud.clear()
                self._strong.clear()
            return events

        onset, self._was_hit = hit and not self._was_hit, hit
        if onset and at - self._pet_ended_at >= _TAP_AFTER_PET_S:
            if self._knocks and at - self._knocks[-1][0] > _TAP_MAX_GAP_S:
                self._knocks = []
            self._knocks.append((at, knock))
        tapping = _steady_series(self._knocks)
        # Sparse petting needs strong strokes: footsteps or a door barely clear the petting level,
        # and a run of sharp knocks is tapping (or a bump), not a hand.
        settled = (
            len(self._strong) >= _PET_MIN_WINDOWS
            and at - self._strong[0][0] >= _PET_SETTLE_S
            and _sharp_share(self._strong) < _TAP_SHARP_SHARE
        )
        if self._run >= _PET_SUSTAIN_WINDOWS or (settled and not tapping):
            self._pet_started_at = self._loud[0]
            self._knocks = []
            events.append(TouchEvent(TouchKind.PET_START, at))
        elif self._knocks and at - self._knocks[-1][0] >= _TAP_HOLD_S:
            if _steady_series(self._knocks):
                events.append(TouchEvent(TouchKind.TAP, at))
            self._knocks = []
            self._loud.clear()
            self._strong.clear()
        return events

    def _pet_end(self, at: float, interrupted: bool = False) -> TouchEvent:
        assert self._pet_started_at is not None
        duration = at - self._pet_started_at
        return TouchEvent(TouchKind.PET_END, at, duration_s=duration, interrupted=interrupted)


def _hiss_share(window: np.ndarray, sample_rate: int) -> float:
    """Fraction of the window's energy above _HISS_HZ."""
    power = np.abs(np.fft.rfft(window * np.hanning(len(window)))) ** 2
    freqs = np.fft.rfftfreq(len(window), 1.0 / sample_rate)
    total = float(power.sum())
    return float(power[freqs > _HISS_HZ].sum()) / total if total > 0 else 0.0


def _peak_width_s(previous: np.ndarray, window: np.ndarray, sample_rate: int) -> float:
    """How long the loudest 5 ms frame of `window` stays within 10 dB (frames run on into `previous`)."""
    size = max(1, int(sample_rate * _KNOCK_FRAME_S))
    audio = np.concatenate([previous, window])
    frames = audio[: len(audio) // size * size].reshape(-1, size)
    energy = 10 * np.log10(np.maximum(np.mean(frames**2, axis=1), 1e-12))
    first = len(previous) // size
    peak = first + int(np.argmax(energy[first:]))
    above = energy > energy[peak] - 10
    start = end = peak
    while start > 0 and above[start - 1]:
        start -= 1
    while end + 1 < len(above) and above[end + 1]:
        end += 1
    return (end - start + 1) * _KNOCK_FRAME_S


def _sharp_share(peaks: Iterable[tuple[float, bool]]) -> float:
    flags = [sharp for _, sharp in peaks]
    return sum(flags) / len(flags)


def _steady_series(knocks: list[tuple[float, bool]]) -> bool:
    if len(knocks) < _TAP_MIN_KNOCKS:
        return False
    if _sharp_share(knocks) < _TAP_SHARP_SHARE:
        return False  # strokes of a petting hand, not knocks
    gaps = np.diff([at for at, _ in knocks])
    median = float(np.median(gaps))
    return bool(np.all(gaps <= median * _TAP_CADENCE) and np.all(gaps >= median / _TAP_CADENCE))


class TouchSpeechGate:
    """Decide when a touch earns a spoken reaction (first one, then once per cooldown)."""

    def __init__(self, cooldown_s: float = 300.0) -> None:
        """Store the minimum gap between spoken reactions of the same kind."""
        self._cooldown_s = cooldown_s
        self._last: dict[TouchKind, float] = {}
        self._count: dict[TouchKind, int] = {}  # touches of each kind, spoken about or not

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
        count = self._count[event.kind] = self._count.get(event.kind, 0) + 1
        text += f" ({count}ª vez)"  # a hook for the model to vary what it says
        last = self._last.get(event.kind)
        if last is not None and now - last < self._cooldown_s:
            return None
        self._last[event.kind] = now
        return text
