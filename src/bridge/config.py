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
    reachy_output_volume: int
    stt_language: str
    reachy_bridge_url: str
    vision_debug_window: bool
    vision_debug_log_interval_s: float
    antenna_finger_tracking_enabled: bool
    antenna_finger_max_angle_deg: float
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
    openclaw_enabled: bool
    openclaw_ws_url: str
    openclaw_bearer_token: str
    openclaw_timeout_s: float
    home_assistant_enabled: bool
    home_assistant_ws_url: str
    home_assistant_token: str
    home_assistant_timeout_s: float
    home_assistant_sensitive_domains: tuple[str, ...]


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
            reachy_output_volume=int(os.getenv("REACHY_OUTPUT_VOLUME", "-1")),
            stt_language=os.getenv("STT_LANGUAGE", "pt"),
            reachy_bridge_url=os.getenv("REACHY_BRIDGE_URL", "http://reachy-bridge:8001"),
            vision_debug_window=_env_flag("VISION_DEBUG_WINDOW", False),
            vision_debug_log_interval_s=float(os.getenv("VISION_DEBUG_LOG_INTERVAL_S", "1.0")),
            antenna_finger_tracking_enabled=_env_flag("ANTENNA_FINGER_TRACKING_ENABLED", True),
            antenna_finger_max_angle_deg=float(os.getenv("ANTENNA_FINGER_MAX_ANGLE_DEG", "180.0")),
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
            openclaw_enabled=_env_flag("OPENCLAW_ENABLED", True),
            openclaw_ws_url=os.getenv("OPENCLAW_WS_URL", "ws://127.0.0.1:18789").strip(),
            openclaw_bearer_token=os.getenv("OPENCLAW_BEARER_TOKEN", "").strip(),
            openclaw_timeout_s=float(os.getenv("OPENCLAW_TIMEOUT_S", "45")),
            home_assistant_enabled=_env_flag("HOME_ASSISTANT_ENABLED", False),
            home_assistant_ws_url=os.getenv(
                "HOME_ASSISTANT_WS_URL", "ws://127.0.0.1:8123/api/websocket"
            ).strip(),
            home_assistant_token=os.getenv("HOME_ASSISTANT_TOKEN", "").strip(),
            home_assistant_timeout_s=float(os.getenv("HOME_ASSISTANT_TIMEOUT_S", "15")),
            home_assistant_sensitive_domains=tuple(
                domain.strip() for domain in os.getenv(
                    "HOME_ASSISTANT_SENSITIVE_DOMAINS",
                    "alarm_control_panel,lock,cover,security_system,button"
                ).split(",") if domain.strip()
            ),
        )
