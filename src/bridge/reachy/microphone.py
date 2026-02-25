"""Continuous microphone listener with VAD-based turn-taking.

This module keeps listening without wake word and detects speech start/end.
It supports barge-in signaling through a speech-start callback.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import wave
from collections import deque
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class MicrophoneListener:
    """Capture Reachy microphone audio and emit utterances when speech ends."""

    def __init__(
        self,
        reachy_mini,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_utterance: Optional[Callable[[bytes], None]] = None,
        vad_on_db: float = -35.0,
        vad_off_db: float = -45.0,
        vad_attack_ms: int = 80,
        vad_release_ms: int = 500,
        pre_roll_ms: int = 250,
        min_utterance_ms: int = 300,
    ) -> None:
        """Initialize microphone listener and VAD parameters."""
        self.reachy_mini = reachy_mini
        self.on_speech_start = on_speech_start
        self.on_utterance = on_utterance

        self.vad_on_db = vad_on_db
        self.vad_off_db = vad_off_db
        self.vad_attack_ms = vad_attack_ms
        self.vad_release_ms = vad_release_ms
        self.pre_roll_ms = pre_roll_ms
        self.min_utterance_ms = min_utterance_ms

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start background audio capture and VAD loop."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop listener loop and audio recording stream."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self.reachy_mini.media.stop_recording()
        except Exception:
            pass

    def _run(self) -> None:
        """Continuously capture audio chunks and perform VAD segmentation."""
        sample_rate = int(self.reachy_mini.media.get_input_audio_samplerate())
        logger.info("Microphone listener started at %s Hz", sample_rate)
        self.reachy_mini.media.start_recording()

        speech_active = False
        speech_above = 0
        speech_below = 0
        utterance_chunks: list[NDArray[np.float32]] = []
        utterance_samples = 0

        pre_roll_samples = int(sample_rate * (self.pre_roll_ms / 1000.0))
        pre_roll_chunks: deque[NDArray[np.float32]] = deque()
        pre_roll_total = 0

        noise_floor_db = -60.0
        empty_reads = 0
        last_status_log = time.monotonic()

        while not self._stop_event.is_set():
            sample = self.reachy_mini.media.get_audio_sample()
            if sample is None:
                empty_reads += 1
                now = time.monotonic()
                if now - last_status_log > 5.0:
                    logger.warning("No microphone sample received yet (reads without data: %s)", empty_reads)
                    last_status_log = now
                time.sleep(0.01)
                continue

            empty_reads = 0

            mono = self._to_float32_mono(sample)
            if mono.size == 0:
                continue

            db = self._rms_dbfs(mono)
            noise_floor_db = 0.98 * noise_floor_db + 0.02 * db
            vad_on_db = max(self.vad_on_db, noise_floor_db + 10.0)
            vad_off_db = max(self.vad_off_db, noise_floor_db + 6.0)
            chunk_ms = int((mono.size / sample_rate) * 1000)
            attack_frames = max(1, self.vad_attack_ms // max(chunk_ms, 1))
            release_frames = max(1, self.vad_release_ms // max(chunk_ms, 1))

            pre_roll_chunks.append(mono)
            pre_roll_total += mono.size
            while pre_roll_total > pre_roll_samples and pre_roll_chunks:
                pre_roll_total -= pre_roll_chunks[0].size
                pre_roll_chunks.popleft()

            if db >= vad_on_db:
                speech_above += 1
                speech_below = 0
                if not speech_active and speech_above >= attack_frames:
                    speech_active = True
                    utterance_chunks = list(pre_roll_chunks)
                    utterance_samples = sum(chunk.size for chunk in utterance_chunks)
                    logger.info("Speech start detected (db=%.1f, threshold=%.1f)", db, vad_on_db)
                    if self.on_speech_start is not None:
                        self.on_speech_start()
            elif db <= vad_off_db:
                speech_below += 1
                speech_above = 0
            else:
                speech_above = 0
                speech_below = 0

            if speech_active:
                utterance_chunks.append(mono)
                utterance_samples += mono.size

                if speech_below >= release_frames:
                    duration_ms = int((utterance_samples / sample_rate) * 1000)
                    logger.info("Speech end detected, duration=%sms", duration_ms)
                    if duration_ms >= self.min_utterance_ms and self.on_utterance is not None:
                        wav_bytes = self._to_wav_bytes(utterance_chunks, sample_rate)
                        self.on_utterance(wav_bytes)
                    else:
                        logger.debug("Discarded short utterance (%sms)", duration_ms)

                    speech_active = False
                    speech_above = 0
                    speech_below = 0
                    utterance_chunks = []
                    utterance_samples = 0

            now = time.monotonic()
            if now - last_status_log > 5.0:
                logger.info(
                    "Mic alive: db=%.1f noise=%.1f on=%.1f off=%.1f speaking=%s",
                    db,
                    noise_floor_db,
                    vad_on_db,
                    vad_off_db,
                    speech_active,
                )
                last_status_log = now

    def _to_wav_bytes(self, chunks: list[NDArray[np.float32]], sample_rate: int) -> bytes:
        """Encode float32 mono chunks into PCM16 WAV bytes."""
        if not chunks:
            return b""
        audio = np.concatenate(chunks)
        audio = np.clip(audio, -1.0, 1.0)
        pcm16 = (audio * 32767.0).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())
        return buffer.getvalue()

    def _to_float32_mono(self, sample: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert arbitrary captured sample shape into mono float32."""
        arr = np.asarray(sample)
        if arr.ndim == 0:
            return np.zeros(0, dtype=np.float32)
        if arr.ndim == 1:
            return arr.astype(np.float32, copy=False)
        if arr.ndim == 2:
            if arr.shape[0] <= 8 and arr.shape[0] <= arr.shape[1]:
                return np.mean(arr, axis=0).astype(np.float32, copy=False)
            return np.mean(arr, axis=1).astype(np.float32, copy=False)
        return np.mean(arr.reshape(arr.shape[0], -1), axis=0).astype(np.float32, copy=False)

    def _rms_dbfs(self, x: NDArray[np.float32]) -> float:
        """Compute RMS level in dBFS from mono float32 samples."""
        x = x.astype(np.float32, copy=False)
        rms = np.sqrt(np.mean(x * x, dtype=np.float32) + 1e-12, dtype=np.float32)
        return float(20.0 * np.log10(float(rms) + 1e-12))
