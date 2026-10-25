"""Detect petting and taps on Dobby's head from the mic array level.

Petting right on the array saturates it (100 ms windows near -10 dBFS, clipped samples) while speech
at a normal distance stays below about -22 dBFS; taps are short transients. Pure logic: the
orchestrator feeds mic chunks and acts on the returned events.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from bridge.runtime.wakeword import _to_mono

_WINDOW_S = 0.1
_PET_SPAN_S = 1.5  # loud windows needed within this span to start petting
_PET_MIN_WINDOWS = 3
_PET_END_S = 2.0  # petting ends after this long without a loud window
_TAP_HOLD_S = 1.0  # a tap waits this long for petting to show up before it is emitted


class TouchKind(str, Enum):
    """Kinds of touch events."""

    PET_START = "pet_start"
    PET_END = "pet_end"
    TAP = "tap"


@dataclass(frozen=True)
class TouchEvent:
    """One detected touch; `duration_s` is set on PET_END."""

    kind: TouchKind
    at: float
    duration_s: float = 0.0


class TouchDetector:
    """Turn mic chunks into petting/tap events using 100 ms window level and peak."""

    def __init__(self, pet_dbfs: float = -18.0, tap_peak: float = 0.5) -> None:
        """Store thresholds: window RMS (dBFS) for petting, sample peak for taps."""
        self._pet_rms = 10 ** (pet_dbfs / 20)
        self._tap_peak = tap_peak
        self._buffer = np.zeros(0, dtype=np.float32)
        self._reset()

    @property
    def petting(self) -> bool:
        """Whether a petting episode is in progress."""
        return self._pet_started_at is not None

    def _reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._loud: deque[float] = deque()
        self._last_loud = float("-inf")
        self._pet_started_at: float | None = None
        self._tap_at: float | None = None

    def observe(
        self, sample_rate: int, sample: np.ndarray, now: float, deaf: bool = False
    ) -> list[TouchEvent]:
        """Feed one mic chunk ending at `now`; return the events it completes."""
        if deaf:
            events = []
            if self._pet_started_at is not None:
                events.append(self._pet_end(now))
            self._reset()
            return events
        self._buffer = np.concatenate([self._buffer, _to_mono(sample)])
        size = max(1, int(sample_rate * _WINDOW_S))
        count = len(self._buffer) // size
        events: list[TouchEvent] = []
        for i in range(count):
            window = self._buffer[i * size : (i + 1) * size]
            end = now - (len(self._buffer) - (i + 1) * size) / sample_rate
            events.extend(self._window(window, end))
        self._buffer = self._buffer[count * size :]
        return events

    def _window(self, window: np.ndarray, at: float) -> list[TouchEvent]:
        events: list[TouchEvent] = []
        loud = float(np.sqrt(np.mean(window**2))) > self._pet_rms
        if loud:
            self._last_loud = at
            self._loud.append(at)
        while self._loud and at - self._loud[0] > _PET_SPAN_S:
            self._loud.popleft()

        if self._pet_started_at is None:
            if len(self._loud) >= _PET_MIN_WINDOWS:
                self._pet_started_at = self._loud[0]
                self._tap_at = None
                events.append(TouchEvent(TouchKind.PET_START, at))
            elif not loud and float(np.max(np.abs(window))) >= self._tap_peak:
                if self._tap_at is None and at - self._last_loud > _PET_SPAN_S:
                    self._tap_at = at
        elif at - self._last_loud >= _PET_END_S:
            events.append(self._pet_end(at))
            self._pet_started_at = None
            self._loud.clear()

        if self._tap_at is not None:
            if self._last_loud >= self._tap_at:
                self._tap_at = None  # petting showed up: the transient was part of it
            elif at - self._tap_at >= _TAP_HOLD_S:
                events.append(TouchEvent(TouchKind.TAP, at))
                self._tap_at = None
        return events

    def _pet_end(self, at: float) -> TouchEvent:
        assert self._pet_started_at is not None
        return TouchEvent(TouchKind.PET_END, at, duration_s=at - self._pet_started_at)


class TouchSpeechGate:
    """Decide when a touch earns a spoken reaction (first one, then once per cooldown)."""

    def __init__(self, cooldown_s: float = 300.0) -> None:
        """Store the minimum gap between spoken reactions of the same kind."""
        self._cooldown_s = cooldown_s
        self._last: dict[TouchKind, float] = {}

    def notice(self, event: TouchEvent, now: float) -> str | None:
        """Return the notice text for the model when `event` should be spoken about, else None."""
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
