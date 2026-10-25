"""Offline wakeword detection: transcribe each burst of speech and look for the robot's name.

The Reachy microphone's DSP gates silence to exact zeros and strips most energy above 3 kHz, which
small keyword models (openWakeWord, sherpa-onnx KWS) could not recognize. Whisper still hears
"Dobby" in that audio, and the gating makes speech bursts easy to cut out, so while Dobby sleeps
every burst is transcribed on a worker thread and the robot wakes when the text names it.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
import unicodedata
from collections import deque
from typing import Optional

import numpy as np

from bridge.runtime.common import resample_audio_chunk

logger = logging.getLogger(__name__)

_RATE = 16000
_FRAME = 320  # 20 ms
_VOICE_PEAK = 0.003  # a frame louder than this is speech (the mic outputs exact zeros in silence)
_PAD_S = 0.6  # audio kept before and after a burst; less context makes Whisper miss the word
_END_SILENCE_S = 0.5
_MAX_BURST_S = 3.0
_MIN_VOICED_S = 0.08
_TARGET_PEAK = 0.5
_MAX_GAIN = 20.0
# Spellings Whisper uses for the name, after accents are stripped.
_NAME_VARIANTS = {"dobby": ("dobby", "dobi", "doby", "dobbie", "dobie", "dobbi")}


def _to_mono(sample: np.ndarray) -> np.ndarray:
    """Downmix (frames, channels) or (channels, frames) audio to a 1-D float32 array."""
    arr = np.asarray(sample, dtype=np.float32)
    if arr.ndim == 2:
        axis = 0 if (arr.shape[0] <= 8 and arr.shape[0] <= arr.shape[1]) else 1
        arr = arr.mean(axis=axis)
    return arr.reshape(-1)


def _plain(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


class OfflineWakewordDetector:
    """Wake on the robot's name, transcribed locally with faster-whisper."""

    def __init__(self, aliases: tuple[str, ...], enabled: bool = True, model_size: str = "base", cooldown_s: float = 2.0) -> None:
        self.enabled = bool(enabled)
        names = [a.strip().lower() for a in aliases if a.strip()] or ["dobby"]
        self._words = {v for name in names for v in _NAME_VARIANTS.get(name, (name,))}
        self._prompt = ", ".join(n.capitalize() for n in names)  # biases Whisper toward the names
        self._model_size = model_size
        self.cooldown_s = cooldown_s
        self.last_rms = 0.0
        self.muted_until = 0.0  # monotonic time before which a heard name never wakes the robot
        self.last_heard = ""  # last transcript, for the web UI
        self.last_heard_ts = 0.0

        self._model = None
        self._model_lock = threading.Lock()
        self._pre = deque(maxlen=int(_PAD_S * _RATE / _FRAME))
        self._burst: list[np.ndarray] = []
        self._voiced = 0
        self._silent = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._jobs: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=2)
        self._woke = threading.Event()
        self._last_trigger_ts = 0.0
        if self.enabled:
            threading.Thread(target=self._worker, name="wakeword", daemon=True).start()

    # ------------------------------------------------------------------ model

    def _get_model(self):
        with self._model_lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                logging.getLogger("faster_whisper").setLevel(logging.WARNING)

                try:
                    self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8", local_files_only=True)
                except Exception:
                    self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
                logger.info("Offline wakeword: Whisper %s ready", self._model_size)
            return self._model

    def transcribe(self, audio: np.ndarray) -> str:
        """Text Whisper hears in a 16 kHz mono clip (level-normalized like live bursts)."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ""
        gain = min(_MAX_GAIN, _TARGET_PEAK / max(float(np.abs(audio).max()), 1e-4))
        segments, _ = self._get_model().transcribe(
            np.clip(audio * gain, -1.0, 1.0),
            language="pt",
            beam_size=1,
            initial_prompt=self._prompt,
            condition_on_previous_text=False,
            without_timestamps=True,
            max_new_tokens=12,  # the name comes in the first words; long decodes only add latency
        )
        return " ".join(s.text for s in segments if s.no_speech_prob < 0.6).strip()

    def hear_clip(self, audio: np.ndarray) -> Optional[str]:
        """Transcribe only the voiced part of a recorded clip, as the live path does; None if silent."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        frames = audio[: audio.size // _FRAME * _FRAME].reshape(-1, _FRAME)
        voiced = np.where(np.abs(frames).max(axis=1) >= _VOICE_PEAK)[0] if frames.size else np.zeros(0, int)
        if voiced.size * _FRAME < _MIN_VOICED_S * _RATE:
            return None
        pad = int(_PAD_S * _RATE)
        start = max(0, voiced[0] * _FRAME - pad)
        end = (voiced[-1] + 1) * _FRAME
        clip = np.concatenate((audio[start:end], np.zeros(pad, np.float32)))  # same trailing silence as live bursts
        return self.transcribe(clip[: int(_MAX_BURST_S * _RATE) + pad])

    def names_robot(self, text: str) -> bool:
        return any(word in self._words for word in re.findall(r"[a-z]+", _plain(text)))

    # ------------------------------------------------------------------ live audio

    def observe_sample(self, sample_rate: int, sample: np.ndarray) -> None:
        """Track the mic level (shown in the web UI)."""
        if sample is not None:
            self.last_rms = float(np.sqrt(np.mean(np.square(_to_mono(sample)))))

    def reset(self) -> None:
        """Forget partial bursts and any pending wake."""
        self._pre.clear()
        self._burst, self._voiced, self._silent = [], 0, 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._woke.clear()

    def process_sample(self, sample_rate: int, sample: np.ndarray) -> bool:
        """Feed mic audio while asleep; True once a transcribed burst named the robot."""
        if not self.enabled or sample is None:
            return False
        mono = _to_mono(sample)
        if sample_rate != _RATE:
            mono = resample_audio_chunk(mono, sample_rate, _RATE)
        audio = np.concatenate((self._pending, mono))
        whole = audio.size // _FRAME * _FRAME
        self._pending = audio[whole:]
        for frame in audio[:whole].reshape(-1, _FRAME):
            self._feed_frame(frame)
        if self._woke.is_set():
            self._woke.clear()
            return True
        return False

    def _feed_frame(self, frame: np.ndarray) -> None:
        voiced = float(np.abs(frame).max()) >= _VOICE_PEAK
        if not self._burst:
            if voiced:
                self._burst = list(self._pre) + [frame]
                self._voiced, self._silent = 1, 0
            else:
                self._pre.append(frame)
            return
        self._burst.append(frame)
        if voiced:
            self._voiced += 1
            self._silent = 0
        else:
            self._silent += 1
        ended = self._silent * _FRAME >= _END_SILENCE_S * _RATE
        if ended or len(self._burst) * _FRAME >= _MAX_BURST_S * _RATE:
            if self._voiced * _FRAME >= _MIN_VOICED_S * _RATE:
                pad = max(0, int(_PAD_S * _RATE / _FRAME) - self._silent)  # end with ~_PAD_S of silence
                clip = np.concatenate(self._burst + [np.zeros(_FRAME, np.float32)] * pad)
                try:
                    self._jobs.put_nowait(clip)
                except queue.Full:
                    logger.debug("Offline wakeword: busy, dropped a speech burst")
            self._burst, self._voiced, self._silent = [], 0, 0
            self._pre.clear()

    def _worker(self) -> None:
        try:
            self._get_model()  # load up front so the first wake is not slower
        except Exception as exc:
            logger.warning("Offline wakeword: Whisper unavailable (%s)", exc)
            return
        while True:
            clip = self._jobs.get()
            started = time.monotonic()
            try:
                text = self.transcribe(clip)
            except Exception as exc:
                logger.warning("Offline wakeword transcription failed: %s", exc)
                continue
            if not text:
                continue
            self.last_heard, self.last_heard_ts = text, time.monotonic()
            hit = self.names_robot(text)
            logger.info("Offline wakeword heard %r (%.1fs of audio, %.2fs)%s", text, clip.size / _RATE, time.monotonic() - started, " → wake" if hit else "")
            now = time.monotonic()
            if hit and now >= self.muted_until and now - self._last_trigger_ts >= self.cooldown_s:
                self._last_trigger_ts = now
                self._woke.set()

    def heard_recently(self, window_s: float = 8.0) -> Optional[str]:
        return self.last_heard if self.last_heard and time.monotonic() - self.last_heard_ts <= window_s else None
