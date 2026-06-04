"""Media IO adapter backed by Reachy SDK media APIs."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bridge.runtime.ports import MediaIOPort


logger = logging.getLogger(__name__)


class ReachyDirectAlsaMediaIO(MediaIOPort):
    """Direct ALSA media adapter for Linux runtimes where SDK GStreamer audio is unavailable."""

    def __init__(self, device_name: str = "default") -> None:
        self._device_name = device_name
        self._sample_rate = 16000
        self._channels = 2
        self._frames_per_chunk = 1024
        self._record_proc: subprocess.Popen[bytes] | None = None
        self._play_proc: subprocess.Popen[bytes] | None = None

    def get_output_audio_samplerate(self) -> int:
        return self._sample_rate

    def get_input_audio_samplerate(self) -> int:
        return self._sample_rate

    def get_audio_sample(self) -> NDArray[np.float32] | None:
        if self._record_proc is None or self._record_proc.stdout is None:
            return None

        byte_count = self._frames_per_chunk * self._channels * 2
        data = self._record_proc.stdout.read(byte_count)
        if not data or len(data) < byte_count:
            return None

        samples = np.frombuffer(data, dtype=np.int16)
        if samples.size == 0:
            return None

        stereo = samples.reshape(-1, self._channels).astype(np.float32)
        mono = stereo.mean(axis=1) / 32768.0
        return mono.astype(np.float32)

    def start_playing(self) -> None:
        if self._play_proc is not None and self._play_proc.poll() is None:
            return
        self._play_proc = subprocess.Popen(
            [
                "aplay",
                "-D",
                self._device_name,
                "-q",
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-r",
                str(self._sample_rate),
                "-c",
                str(self._channels),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop_playing(self) -> None:
        proc = self._play_proc
        self._play_proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            proc.kill()

    def push_audio_sample(self, sample: NDArray[np.float32]) -> None:
        if self._play_proc is None or self._play_proc.stdin is None:
            self.start_playing()
        if self._play_proc is None or self._play_proc.stdin is None:
            return

        mono = np.asarray(sample, dtype=np.float32).reshape(-1)
        clipped = np.clip(mono, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        stereo = np.repeat(pcm16[:, None], self._channels, axis=1)
        try:
            self._play_proc.stdin.write(stereo.tobytes())
            self._play_proc.stdin.flush()
        except BrokenPipeError:
            logger.warning("Direct ALSA playback pipe broke; restarting aplay")
            self.stop_playing()

    def start_recording(self) -> None:
        if self._record_proc is not None and self._record_proc.poll() is None:
            return
        self._record_proc = subprocess.Popen(
            [
                "arecord",
                "-D",
                self._device_name,
                "-q",
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-r",
                str(self._sample_rate),
                "-c",
                str(self._channels),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def stop_recording(self) -> None:
        proc = self._record_proc
        self._record_proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            proc.kill()


class ReachySdkMediaIO(MediaIOPort):
    """Expose Reachy SDK media as runtime media port."""

    def __init__(self, reachy_sdk_instance: Any) -> None:
        """Bind media controls from the provided Reachy SDK instance."""
        self._media = reachy_sdk_instance.media

    def get_output_audio_samplerate(self) -> int:
        """Return Reachy speaker output sample rate."""
        return int(self._media.get_output_audio_samplerate())

    def get_input_audio_samplerate(self) -> int:
        """Return Reachy microphone input sample rate."""
        return int(self._media.get_input_audio_samplerate())

    def get_audio_sample(self) -> NDArray[np.float32] | None:
        """Fetch one audio sample chunk from Reachy microphone stream."""
        sample = self._media.get_audio_sample()
        return sample

    def start_playing(self) -> None:
        """Start Reachy audio playback pipeline."""
        self._media.start_playing()

    def stop_playing(self) -> None:
        """Stop Reachy audio playback pipeline."""
        self._media.stop_playing()

    def push_audio_sample(self, sample: NDArray[np.float32]) -> None:
        """Push one audio sample chunk to Reachy speakers."""
        self._media.push_audio_sample(sample)

    def start_recording(self) -> None:
        """Start Reachy microphone recording."""
        self._media.start_recording()

    def stop_recording(self) -> None:
        """Stop Reachy microphone recording."""
        self._media.stop_recording()


def build_reachy_media_io(reachy_sdk_instance: Any) -> MediaIOPort:
    """Select the most reliable local media backend for the current runtime."""
    use_direct_alsa = os.getenv("REACHY_DIRECT_ALSA_AUDIO", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if (
        use_direct_alsa
        and platform.system() == "Linux"
        and shutil.which("arecord")
        and shutil.which("aplay")
    ):
        logger.info("Using direct ALSA media adapter for bridge audio")
        return ReachyDirectAlsaMediaIO(device_name="default")
    return ReachySdkMediaIO(reachy_sdk_instance)
