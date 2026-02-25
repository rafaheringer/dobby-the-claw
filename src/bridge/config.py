from dataclasses import dataclass
import os


@dataclass(frozen=True)
class BridgeConfig:
    openclaw_api_url: str
    openclaw_intent_path: str
    openclaw_timeout_s: int
    llm_mode: str
    llm_api_base: str
    llm_model: str
    llm_api_key_env: str
    stt_api_base: str
    stt_api_key_env: str
    stt_model: str
    stt_language: str
    mic_vad_on_db: float
    mic_vad_off_db: float
    mic_vad_attack_ms: int
    mic_vad_release_ms: int
    mic_pre_roll_ms: int
    mic_min_utterance_ms: int
    tts_api_base: str
    tts_api_key_env: str
    tts_model: str
    tts_voice: str
    tts_format: str
    tts_output: str
    reachy_bridge_url: str
    stt_provider: str
    tts_provider: str
    wake_word: str
    listening_timeout_s: int
    confirmation_timeout_s: int


    @staticmethod
    def from_env() -> "BridgeConfig":
        openclaw_api_url = os.getenv("OPENCLOW_API_URL") or os.getenv(
            "OPENCLAW_API_URL", "http://openclaw:8000"
        )
        return BridgeConfig(
            openclaw_api_url=openclaw_api_url,
            openclaw_intent_path=os.getenv("OPENCLAW_INTENT_PATH", "/v1/intent"),
            openclaw_timeout_s=int(os.getenv("OPENCLAW_TIMEOUT_S", "20")),
            llm_mode=os.getenv("BRIDGE_LLM_MODE", "openclaw"),
            llm_api_base=os.getenv("LLM_API_BASE", "https://api.openai.com/v1"),
            llm_model=os.getenv("LLM_MODEL", "gpt-5.2"),
            llm_api_key_env=os.getenv("LLM_API_KEY_ENV", "OPENAI_API_KEY"),
            stt_api_base=os.getenv("STT_API_BASE")
            or os.getenv("LLM_API_BASE", "https://api.openai.com/v1"),
            stt_api_key_env=os.getenv("STT_API_KEY_ENV")
            or os.getenv("LLM_API_KEY_ENV", "OPENAI_API_KEY"),
            stt_model=os.getenv("STT_MODEL", "gpt-4o-mini-transcribe"),
            stt_language=os.getenv("STT_LANGUAGE", "pt"),
            mic_vad_on_db=float(os.getenv("MIC_VAD_ON_DB", "-35")),
            mic_vad_off_db=float(os.getenv("MIC_VAD_OFF_DB", "-45")),
            mic_vad_attack_ms=int(os.getenv("MIC_VAD_ATTACK_MS", "80")),
            mic_vad_release_ms=int(os.getenv("MIC_VAD_RELEASE_MS", "500")),
            mic_pre_roll_ms=int(os.getenv("MIC_PRE_ROLL_MS", "250")),
            mic_min_utterance_ms=int(os.getenv("MIC_MIN_UTTERANCE_MS", "300")),
            tts_api_base=os.getenv("TTS_API_BASE")
            or os.getenv("LLM_API_BASE", "https://api.openai.com/v1"),
            tts_api_key_env=os.getenv("TTS_API_KEY_ENV")
            or os.getenv("LLM_API_KEY_ENV", "OPENAI_API_KEY"),
            tts_model=os.getenv("TTS_MODEL", "gpt-4o-mini-tts"),
            tts_voice=os.getenv("TTS_VOICE", "alloy"),
            tts_format=os.getenv("TTS_FORMAT", "wav"),
            tts_output=os.getenv("TTS_OUTPUT", "reachy"),
            reachy_bridge_url=os.getenv("REACHY_BRIDGE_URL", "http://reachy-bridge:8001"),
            stt_provider=os.getenv("STT_PROVIDER", "openai"),
            tts_provider=os.getenv("TTS_PROVIDER", "openai"),
            wake_word=os.getenv("WAKE_WORD", "Reachy"),
            listening_timeout_s=int(os.getenv("LISTENING_TIMEOUT_S", "10")),
            confirmation_timeout_s=int(os.getenv("CONFIRMATION_TIMEOUT_S", "15")),
        )
