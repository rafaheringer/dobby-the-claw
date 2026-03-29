"""Media IO adapter backed by Reachy SDK media APIs."""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from bridge.runtime.ports import MediaIOPort


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
