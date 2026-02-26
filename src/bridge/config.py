from dataclasses import dataclass
import os


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class BridgeConfig:
    llm_api_base: str
    llm_api_key_env: str
    realtime_model: str
    realtime_transcription_model: str
    realtime_vad_silence_ms: int
    realtime_vad_prefix_padding_ms: int
    stt_language: str
    reachy_bridge_url: str
    vision_debug_window: bool
    vision_debug_log_interval_s: float
    camera_tool_enabled: bool


    @staticmethod
    def from_env() -> "BridgeConfig":
        return BridgeConfig(
            llm_api_base=os.getenv("LLM_API_BASE", "https://api.openai.com/v1"),
            llm_api_key_env=os.getenv("LLM_API_KEY_ENV", "OPENAI_API_KEY"),
            realtime_model=os.getenv("REALTIME_MODEL", "gpt-realtime"),
            realtime_transcription_model=os.getenv(
                "REALTIME_TRANSCRIPTION_MODEL", "gpt-4o-transcribe"
            ),
            realtime_vad_silence_ms=int(os.getenv("REALTIME_VAD_SILENCE_MS", "250")),
            realtime_vad_prefix_padding_ms=int(
                os.getenv("REALTIME_VAD_PREFIX_PADDING_MS", "200")
            ),
            stt_language=os.getenv("STT_LANGUAGE", "pt"),
            reachy_bridge_url=os.getenv("REACHY_BRIDGE_URL", "http://reachy-bridge:8001"),
            vision_debug_window=_env_flag("VISION_DEBUG_WINDOW", False),
            vision_debug_log_interval_s=float(os.getenv("VISION_DEBUG_LOG_INTERVAL_S", "1.0")),
            camera_tool_enabled=_env_flag("CAMERA_TOOL_ENABLED", True),
        )
