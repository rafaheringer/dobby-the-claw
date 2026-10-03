"""Default conversation session adapter backed by OpenAI Realtime."""

from __future__ import annotations

from typing import Any, Callable

from bridge.config import BridgeConfig

from bridge.runtime.audio_support import build_realtime_session, resolve_gemini_api_key
from bridge.runtime.ports import ConversationCallbacks, ConversationSessionFactoryPort, ConversationSessionPort


class OpenAIRealtimeSessionFactory(ConversationSessionFactoryPort):
    """Build conversation sessions using the current OpenAI Realtime client."""

    def __init__(self, config: BridgeConfig) -> None:
        """Store bridge configuration for creating realtime sessions."""
        self._config = config

    def create(
        self,
        *,
        api_key: str,
        identity_prompt: str,
        tool_specs: list[dict[str, Any]],
        on_tool_call: Callable[[str, dict[str, Any]], Any],
        callbacks: ConversationCallbacks,
    ) -> ConversationSessionPort:
        """Create a configured OpenAI Realtime session adapter instance."""
        return build_realtime_session(
            api_key=api_key,
            config=self._config,
            identity_prompt=identity_prompt,
            tool_specs=tool_specs,
            on_tool_call=on_tool_call,
            on_speech_start=callbacks.on_speech_start,
            on_user_transcript=callbacks.on_user_transcript,
            on_assistant_text=callbacks.on_assistant_text,
            on_assistant_audio_chunk=callbacks.on_assistant_audio_chunk,
            on_assistant_audio_done=callbacks.on_assistant_audio_done,
            on_error=callbacks.on_error,
        )


class GeminiLiveSessionFactory(ConversationSessionFactoryPort):
    """Build conversation sessions using Gemini Live."""

    def __init__(self, config: BridgeConfig) -> None:
        """Store bridge configuration for creating Gemini sessions."""
        self._config = config

    def create(
        self,
        *,
        api_key: str,
        identity_prompt: str,
        tool_specs: list[dict[str, Any]],
        on_tool_call: Callable[[str, dict[str, Any]], Any],
        callbacks: ConversationCallbacks,
    ) -> ConversationSessionPort:
        """Create a configured Gemini Live session."""
        from reachy.gemini_live_client import GeminiLiveSession

        gemini_key = resolve_gemini_api_key()
        if not gemini_key:
            raise RuntimeError("Missing GEMINI_API_KEY for CONVERSATION_PROVIDER=gemini")
        return GeminiLiveSession(
            api_key=gemini_key,
            model=self._config.gemini_live_model,
            instructions=identity_prompt,
            thinking_level=self._config.gemini_thinking_level,
            voice=self._config.gemini_voice,
            voice_style=self._config.gemini_voice_style,
            vad_silence_ms=max(200, self._config.realtime_vad_silence_ms),
            vad_prefix_padding_ms=self._config.realtime_vad_prefix_padding_ms,
            tool_specs=tool_specs,
            on_tool_call=on_tool_call,
            on_speech_start=callbacks.on_speech_start,
            on_user_transcript=callbacks.on_user_transcript,
            on_assistant_text=callbacks.on_assistant_text,
            on_assistant_audio_chunk=callbacks.on_assistant_audio_chunk,
            on_assistant_audio_done=callbacks.on_assistant_audio_done,
            on_error=callbacks.on_error,
        )


def build_conversation_factory(config: BridgeConfig) -> ConversationSessionFactoryPort:
    """Pick the conversation provider configured via CONVERSATION_PROVIDER."""
    if config.conversation_provider == "gemini":
        return GeminiLiveSessionFactory(config)
    return OpenAIRealtimeSessionFactory(config)
