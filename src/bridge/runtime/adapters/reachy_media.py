"""Media IO adapter backed by Reachy SDK media APIs."""

from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from bridge.runtime.ports import MediaIOPort


class ReachySdkMediaIO(MediaIOPort):
    """Expose Reachy SDK media as runtime media port."""

    def __init__(self, reachy_sdk_instance: Any) -> None:
        self._media = reachy_sdk_instance.media

    def get_output_audio_samplerate(self) -> int:
        return int(self._media.get_output_audio_samplerate())

    def get_input_audio_samplerate(self) -> int:
        return int(self._media.get_input_audio_samplerate())

    def get_audio_sample(self) -> NDArray[np.float32] | None:
        sample = self._media.get_audio_sample()
        return sample

    def start_playing(self) -> None:
        self._media.start_playing()

    def stop_playing(self) -> None:
        self._media.stop_playing()

    def push_audio_sample(self, sample: NDArray[np.float32]) -> None:
        self._media.push_audio_sample(sample)

    def start_recording(self) -> None:
        self._media.start_recording()

    def stop_recording(self) -> None:
        self._media.stop_recording()
