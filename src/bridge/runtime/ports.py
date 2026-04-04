"""Runtime ports for conversation/session abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ConversationCallbacks:
    """Callback bundle consumed by conversation session adapters."""

    on_speech_start: Optional[Callable[[], None]] = None
    on_user_transcript: Optional[Callable[[str], None]] = None
    on_assistant_text: Optional[Callable[[str], None]] = None
    on_assistant_audio_chunk: Optional[Callable[[NDArray[np.float32]], None]] = None
    on_assistant_audio_done: Optional[Callable[[], None]] = None
    on_error: Optional[Callable[[str], None]] = None


class ConversationSessionPort(Protocol):
    """Abstract conversation session used by runtime orchestration."""

    def start(self) -> None:
        """Start the session and begin processing realtime events."""
        ...

    def wait_until_ready(self, timeout_s: float = 8.0) -> bool:
        """Block until session is connected and ready or timeout expires."""
        ...

    def stop(self) -> None:
        """Stop the session and release network/audio resources."""
        ...

    def feed_audio(self, sample_rate: int, sample: NDArray[np.float32]) -> None:
        """Send one input audio sample chunk to the session."""
        ...

    def send_text(self, text: str) -> bool:
        """Send one text turn and report whether enqueueing succeeded."""
        ...


class ConversationSessionFactoryPort(Protocol):
    """Factory that creates runtime conversation session ports."""

    def create(
        self,
        *,
        api_key: str,
        identity_prompt: str,
        tool_specs: list[dict[str, Any]],
        on_tool_call: Callable[[str, dict[str, Any]], Any],
        callbacks: ConversationCallbacks,
    ) -> ConversationSessionPort:
        """Create a configured conversation session adapter."""
        ...


class RobotActionsPort(Protocol):
    """Abstract robot action interface used by runtime orchestration."""

    def gesture_listening(self) -> None:
        """Trigger robot listening gesture."""
        ...

    def gesture_think(self) -> None:
        """Trigger robot thinking gesture."""
        ...

    def gesture_delegating(self) -> None:
        """Trigger robot delegating gesture."""
        ...

    def wake_up(self) -> None:
        """Wake robot from sleep posture."""
        ...

    def goto_sleep(self) -> None:
        """Send robot to sleep posture."""
        ...

    def disable_motors(self) -> None:
        """Cut torque on all motors to save power."""
        ...

    def enable_motors(self) -> None:
        """Re-energize all motors."""
        ...


class ToolRuntimePort(Protocol):
    """Abstract tool runtime interface used by orchestration."""

    def names(self) -> list[str]:
        """Return registered tool names."""
        ...

    def openai_specs(self) -> list[dict[str, Any]]:
        """Return OpenAI function-call specifications for tools."""
        ...

    def runtime_guardrails(self) -> list[str]:
        """Return runtime guardrail statements contributed by tools."""
        ...

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute one tool by name with validated arguments."""
        ...


class MediaIOPort(Protocol):
    """Abstract audio input/output interface used by runtime orchestration."""

    def get_output_audio_samplerate(self) -> int:
        """Return speaker output sample rate."""
        ...

    def get_input_audio_samplerate(self) -> int:
        """Return microphone input sample rate."""
        ...

    def get_audio_sample(self) -> NDArray[np.float32] | None:
        """Read one microphone audio sample chunk when available."""
        ...

    def start_playing(self) -> None:
        """Start speaker playback mode."""
        ...

    def stop_playing(self) -> None:
        """Stop speaker playback mode."""
        ...

    def push_audio_sample(self, sample: NDArray[np.float32]) -> None:
        """Write one audio sample chunk to speaker output."""
        ...

    def start_recording(self) -> None:
        """Start microphone recording mode."""
        ...

    def stop_recording(self) -> None:
        """Stop microphone recording mode."""
        ...
