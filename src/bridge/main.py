import logging
import time

from bridge.config import BridgeConfig
from bridge.openclaw_client import OpenClawClient
from bridge.reachy_client import ReachyClient
from bridge.state_machine import StateMachine
from bridge.voice import VoicePipeline


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = BridgeConfig.from_env()

    logging.info("Bridge starting")
    logging.info("OpenClaw API: %s", config.openclaw_api_url)
    logging.info("Reachy Bridge API: %s", config.reachy_bridge_url)

    state_machine = StateMachine()
    openclaw = OpenClawClient(config.openclaw_api_url)
    reachy = ReachyClient(config.reachy_bridge_url)
    voice = VoicePipeline(config.stt_provider, config.tts_provider, config.wake_word)

    _ = (state_machine, openclaw, reachy, voice)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
