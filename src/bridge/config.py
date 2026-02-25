from dataclasses import dataclass
import os


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
        )
