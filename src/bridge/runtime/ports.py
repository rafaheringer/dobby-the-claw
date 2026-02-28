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

    def start(self) -> None: ...

    def wait_until_ready(self, timeout_s: float = 8.0) -> bool: ...

    def stop(self) -> None: ...

    def feed_audio(self, sample_rate: int, sample: NDArray[np.float32]) -> None: ...

    def send_text(self, text: str) -> bool: ...


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
    ) -> ConversationSessionPort: ...


class RobotActionsPort(Protocol):
    """Abstract robot action interface used by runtime orchestration."""

    def gesture_listening(self) -> None: ...

    def gesture_think(self) -> None: ...

    def wake_up(self) -> None: ...

    def goto_sleep(self) -> None: ...


class ToolRuntimePort(Protocol):
    """Abstract tool runtime interface used by orchestration."""

    def names(self) -> list[str]: ...

    def openai_specs(self) -> list[dict[str, Any]]: ...

    def execute(self, name: str, arguments: dict[str, Any]) -> Any: ...
