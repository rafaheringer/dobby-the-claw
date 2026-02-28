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
    head_tracking_enabled: bool
    idle_sleep_timeout_s: float
    offline_wakeword_enabled: bool
    offline_wakeword_aliases: tuple[str, ...]
    offline_wakeword_threshold: float
    offline_wakeword_fallback_on_speech: bool
    offline_wakeword_auto_calibration_enabled: bool
    offline_wakeword_calibration_seconds: float
    offline_wakeword_calibration_multiplier: float
    offline_wakeword_fallback_speech_rms_threshold: float


    @staticmethod
    def from_env() -> "BridgeConfig":
        aliases_raw = os.getenv("OFFLINE_WAKEWORD_ALIASES", "reachy,dobby")
        aliases = tuple(
            alias.strip().lower()
            for alias in aliases_raw.split(",")
            if alias.strip()
        )
        if not aliases:
            aliases = ("reachy", "dobby")

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
            head_tracking_enabled=_env_flag("HEAD_TRACKING_ENABLED", True),
            idle_sleep_timeout_s=float(os.getenv("IDLE_SLEEP_TIMEOUT_S", "300")),
            offline_wakeword_enabled=_env_flag("OFFLINE_WAKEWORD_ENABLED", True),
            offline_wakeword_aliases=aliases,
            offline_wakeword_threshold=float(os.getenv("OFFLINE_WAKEWORD_THRESHOLD", "0.5")),
            offline_wakeword_fallback_on_speech=_env_flag(
                "OFFLINE_WAKEWORD_FALLBACK_ON_SPEECH", True
            ),
            offline_wakeword_auto_calibration_enabled=_env_flag(
                "OFFLINE_WAKEWORD_AUTO_CALIBRATION_ENABLED", True
            ),
            offline_wakeword_calibration_seconds=float(
                os.getenv("OFFLINE_WAKEWORD_CALIBRATION_SECONDS", "6.0")
            ),
            offline_wakeword_calibration_multiplier=float(
                os.getenv("OFFLINE_WAKEWORD_CALIBRATION_MULTIPLIER", "2.8")
            ),
            offline_wakeword_fallback_speech_rms_threshold=float(
                os.getenv("OFFLINE_WAKEWORD_FALLBACK_SPEECH_RMS_THRESHOLD", "0.03")
            ),
        )
