"""Offline wakeword detection helpers.

This module prefers openWakeWord when available and falls back to
simple local speech-energy detection for MVP operation when custom
wakeword models are not configured yet.
"""

from __future__ import annotations

import logging
import importlib
import time

import numpy as np

from bridge.runtime.common import resample_audio_chunk

logger = logging.getLogger(__name__)


class OfflineWakewordDetector:
    """Detect wake events from local microphone audio without cloud dependency."""

    def __init__(
        self,
        aliases: tuple[str, ...],
        threshold: float,
        enabled: bool = True,
        fallback_on_speech: bool = True,
        cooldown_s: float = 2.0,
        speech_rms_threshold: float = 0.03,
        auto_calibration_enabled: bool = True,
        calibration_seconds: float = 6.0,
        calibration_multiplier: float = 2.8,
    ) -> None:
        self.enabled = bool(enabled)
        self.aliases = tuple(a.strip().lower() for a in aliases if a.strip())
        self.threshold = float(threshold)
        self.fallback_on_speech = bool(fallback_on_speech)
        self.cooldown_s = max(0.2, float(cooldown_s))
        self.speech_rms_threshold = float(speech_rms_threshold)
        self.auto_calibration_enabled = bool(auto_calibration_enabled)
        self.calibration_seconds = max(1.0, float(calibration_seconds))
        self.calibration_multiplier = max(1.2, float(calibration_multiplier))

        self._target_rate = 16000
        self._frame_samples = int(0.08 * self._target_rate)  # 80 ms
        self._buffer = np.array([], dtype=np.int16)
        self._last_trigger_ts = 0.0
        self._speech_hits = 0

        self._model = None
        self._alias_model_names: set[str] = set()
        self._openwakeword_available = False
        self._calibration_deadline_ts: float | None = None
        self._calibration_count = 0
        self._calibration_sum_rms = 0.0
        self._calibration_sum_sq = 0.0
        self._calibration_finalized = False
        self._calibration_min_frames = 30

        if self.enabled:
            self._try_init_openwakeword()

    def start_auto_calibration(self) -> None:
        """Start non-blocking ambient calibration window.

        The calibration runs opportunistically as samples are observed,
        and does not block audio/LLM pipelines.
        """
        if not self.enabled or not self.auto_calibration_enabled:
            return
        self._calibration_deadline_ts = time.monotonic() + self.calibration_seconds
        self._calibration_count = 0
        self._calibration_sum_rms = 0.0
        self._calibration_sum_sq = 0.0
        self._calibration_finalized = False
        logger.info(
            "Offline wakeword calibration started for %.1fs (current fallback_rms=%.4f)",
            self.calibration_seconds,
            self.speech_rms_threshold,
        )

    def _try_init_openwakeword(self) -> None:
        try:
            model_module = importlib.import_module("openwakeword.model")
            model_cls = getattr(model_module, "Model")
            self._model = model_cls()
            self._openwakeword_available = True
            logger.info("Offline wakeword: openWakeWord initialized")
        except Exception as exc:
            self._model = None
            self._openwakeword_available = False
            logger.warning("Offline wakeword: openWakeWord unavailable (%s)", exc)

    def reset(self) -> None:
        self._buffer = np.array([], dtype=np.int16)
        self._speech_hits = 0

    def process_sample(self, sample_rate: int, sample: np.ndarray) -> bool:
        """Feed one float32 audio sample and return True when wake is detected."""
        if not self.enabled:
            return False

        if sample is None:
            return False

        mono = np.asarray(sample, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return False

        if sample_rate != self._target_rate:
            mono = resample_audio_chunk(mono, sample_rate, self._target_rate)

        self.observe_sample(self._target_rate, mono)

        pcm16 = np.clip(mono, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        if pcm16.size == 0:
            return False

        self._buffer = np.concatenate((self._buffer, pcm16))

        while self._buffer.size >= self._frame_samples:
            frame = self._buffer[: self._frame_samples]
            self._buffer = self._buffer[self._frame_samples :]

            if self._frame_is_wake(frame):
                return True

        return False

    def observe_sample(self, sample_rate: int, sample: np.ndarray) -> None:
        """Observe audio for calibration only (no wake decision)."""
        if not self.enabled or not self.auto_calibration_enabled:
            return
        if self._calibration_deadline_ts is None or self._calibration_finalized:
            return

        mono = np.asarray(sample, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return

        if sample_rate != self._target_rate:
            mono = resample_audio_chunk(mono, sample_rate, self._target_rate)

        idx = 0
        while idx + self._frame_samples <= mono.size:
            frame = mono[idx : idx + self._frame_samples]
            idx += self._frame_samples

            rms = float(np.sqrt(np.mean(frame * frame)))
            if not np.isfinite(rms):
                continue

            self._calibration_count += 1
            self._calibration_sum_rms += rms
            self._calibration_sum_sq += (rms * rms)

        if time.monotonic() >= self._calibration_deadline_ts:
            self._finalize_calibration()

    def _finalize_calibration(self) -> None:
        if self._calibration_finalized:
            return
        self._calibration_finalized = True

        if self._calibration_count < self._calibration_min_frames:
            logger.info(
                "Offline wakeword calibration skipped (insufficient frames=%s), keeping fallback_rms=%.4f",
                self._calibration_count,
                self.speech_rms_threshold,
            )
            return

        mean_rms = self._calibration_sum_rms / float(self._calibration_count)
        var_rms = max(
            0.0,
            (self._calibration_sum_sq / float(self._calibration_count)) - (mean_rms * mean_rms),
        )
        std_rms = float(np.sqrt(var_rms))

        candidate = max(
            0.008,
            (mean_rms * self.calibration_multiplier) + (0.5 * std_rms),
        )
        calibrated = min(0.25, candidate)
        self.speech_rms_threshold = calibrated

        logger.info(
            "Offline wakeword calibration finished: frames=%s mean=%.4f std=%.4f new_fallback_rms=%.4f",
            self._calibration_count,
            mean_rms,
            std_rms,
            self.speech_rms_threshold,
        )

    def _frame_is_wake(self, frame: np.ndarray) -> bool:
        now = time.monotonic()
        if (now - self._last_trigger_ts) < self.cooldown_s:
            return False

        if self._openwakeword_available and self._model is not None:
            try:
                prediction = self._model.predict(frame)
                if isinstance(prediction, dict):
                    if not self._alias_model_names and self.aliases:
                        lowered = {str(name).lower() for name in prediction.keys()}
                        matched = {
                            name
                            for name in lowered
                            for alias in self.aliases
                            if alias in name
                        }
                        self._alias_model_names = matched
                        if not matched:
                            logger.warning(
                                "Offline wakeword: no model names match aliases=%s; using fallback speech trigger until custom models are added",
                                self.aliases,
                            )

                    candidate_names = self._alias_model_names or {
                        str(name).lower() for name in prediction.keys()
                    }
                    for name, score in prediction.items():
                        n = str(name).lower()
                        if n not in candidate_names:
                            continue
                        if float(score) >= self.threshold:
                            self._last_trigger_ts = now
                            self._speech_hits = 0
                            logger.info(
                                "Offline wakeword detected via openWakeWord model=%s score=%.3f",
                                name,
                                float(score),
                            )
                            return True
            except Exception as exc:
                logger.debug("Offline wakeword model prediction failed: %s", exc)

        if self.fallback_on_speech:
            f = frame.astype(np.float32) / 32767.0
            rms = float(np.sqrt(np.mean(f * f))) if f.size > 0 else 0.0
            if rms >= self.speech_rms_threshold:
                self._speech_hits += 1
            else:
                self._speech_hits = max(0, self._speech_hits - 1)

            if self._speech_hits >= 3:
                self._last_trigger_ts = now
                self._speech_hits = 0
                logger.info(
                    "Offline wakeword fallback triggered by local speech energy (rms=%.4f)",
                    rms,
                )
                return True

        return False
