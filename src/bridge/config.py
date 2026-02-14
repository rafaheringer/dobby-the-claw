from dataclasses import dataclass
import os


@dataclass(frozen=True)
class BridgeConfig:
    openclaw_api_url: str
    openclaw_intent_path: str
    openclaw_timeout_s: int
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
            reachy_bridge_url=os.getenv("REACHY_BRIDGE_URL", "http://reachy-bridge:8001"),
            stt_provider=os.getenv("STT_PROVIDER", "openai"),
            tts_provider=os.getenv("TTS_PROVIDER", "openai"),
            wake_word=os.getenv("WAKE_WORD", "Reachy"),
            listening_timeout_s=int(os.getenv("LISTENING_TIMEOUT_S", "10")),
            confirmation_timeout_s=int(os.getenv("CONFIRMATION_TIMEOUT_S", "15")),
        )
