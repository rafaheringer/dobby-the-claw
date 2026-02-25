import argparse
from queue import Empty, Queue
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

from bridge.config import BridgeConfig
from bridge.llm_client import DirectLLMClient
from bridge.openclaw_client import OpenClawClient
from bridge.reachy.camera_worker import CameraWorker
from bridge.reachy.client import ReachyClient
from bridge.reachy.microphone import MicrophoneListener
from bridge.reachy.motion import MotionManager
from bridge.reachy.stt_client import OpenAISttClient
from bridge.reachy.voice import VoicePipeline
from bridge.state_machine import StateMachine
from bridge.state_machine import Event


def main() -> None:
    parser = argparse.ArgumentParser(description="Dobby bridge")
    parser.add_argument(
        "--mode",
        choices=["idle", "cli", "mic"],
        default=os.getenv("BRIDGE_MODE", "mic"),
    )
    parser.add_argument(
        "--llm-mode",
        choices=["openclaw", "direct"],
        default=None,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = BridgeConfig.from_env()

    logging.info("Bridge starting")
    logging.info("OpenClaw API: %s", config.openclaw_api_url)
    logging.info("Reachy Bridge API: %s", config.reachy_bridge_url)
    logging.info("Run mode: %s", args.mode)

    state_machine = StateMachine()
    llm_mode = args.llm_mode or config.llm_mode
    openclaw: Optional[OpenClawClient] = None
    direct_llm: Optional[DirectLLMClient] = None
    if llm_mode == "openclaw":
        openclaw = OpenClawClient(
            config.openclaw_api_url,
            intent_path=config.openclaw_intent_path,
            timeout_s=config.openclaw_timeout_s,
        )
    else:
        api_key = os.getenv(config.llm_api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing API key in env var {config.llm_api_key_env}"
            )
        direct_llm = DirectLLMClient(
            api_base=config.llm_api_base,
            api_key=api_key,
            model=config.llm_model,
            timeout_s=config.openclaw_timeout_s,
        )
    reachy = ReachyClient(config.reachy_bridge_url)
    reachy_sdk_instance = None
    camera_worker = None
    motion_manager = None
    if config.reachy_bridge_url.strip().lower().startswith("sdk"):
        try:
            reachy_sdk_instance = reachy.get_sdk_instance()
            camera_worker = CameraWorker(reachy_sdk_instance)
            camera_worker.start()
            motion_manager = MotionManager(reachy_sdk_instance, camera_worker=camera_worker)
            motion_manager.start()
        except Exception:
            reachy_sdk_instance = None
            camera_worker = None
            motion_manager = None

    voice = VoicePipeline(
        config.stt_provider,
        config.tts_provider,
        config.wake_word,
        config.tts_api_base,
        config.tts_api_key_env,
        config.tts_model,
        config.tts_voice,
        config.tts_format,
        config.tts_output,
        reachy_sdk_instance,
        motion_manager,
    )

    if args.mode == "cli":
        logging.info("Interactive CLI input is disabled. Switching to microphone mode.")
        args.mode = "mic"

    if args.mode == "mic":
        _run_mic_loop(
            state_machine,
            openclaw,
            direct_llm,
            reachy,
            voice,
            motion_manager,
            camera_worker,
            config,
            reachy_sdk_instance,
        )
        return

    _ = (state_machine, openclaw, reachy, voice, motion_manager, camera_worker)

    while True:
        time.sleep(1)


def _run_mic_loop(
    state_machine: StateMachine,
    openclaw: Optional[OpenClawClient],
    direct_llm: Optional[DirectLLMClient],
    reachy: ReachyClient,
    voice: VoicePipeline,
    motion_manager: Optional[MotionManager],
    camera_worker: Optional[CameraWorker],
    config: BridgeConfig,
    reachy_sdk_instance,
) -> None:
    """Run continuous microphone interaction loop without wake word."""
    if reachy_sdk_instance is None:
        raise RuntimeError("Microphone mode requires REACHY_BRIDGE_URL=sdk")

    stt_api_key = os.getenv(config.stt_api_key_env, "").strip()
    if not stt_api_key:
        raise RuntimeError(f"Missing STT API key in env var {config.stt_api_key_env}")

    stt = OpenAISttClient(
        api_base=config.stt_api_base,
        api_key=stt_api_key,
        model=config.stt_model,
        timeout_s=config.openclaw_timeout_s,
    )
    logging.info("STT model: %s", config.stt_model)
    logging.info(
        "MIC VAD config on/off=%.1f/%.1f attack/release=%sms/%sms pre-roll=%sms min-utt=%sms",
        config.mic_vad_on_db,
        config.mic_vad_off_db,
        config.mic_vad_attack_ms,
        config.mic_vad_release_ms,
        config.mic_pre_roll_ms,
        config.mic_min_utterance_ms,
    )

    session_id = str(uuid.uuid4())
    utterance_queue: "Queue[bytes]" = Queue()

    def _on_speech_start() -> None:
        if voice.is_speaking():
            voice.interrupt()

    def _on_utterance(wav_bytes: bytes) -> None:
        utterance_queue.put(wav_bytes)

    microphone = MicrophoneListener(
        reachy_mini=reachy_sdk_instance,
        on_speech_start=_on_speech_start,
        on_utterance=_on_utterance,
        vad_on_db=config.mic_vad_on_db,
        vad_off_db=config.mic_vad_off_db,
        vad_attack_ms=config.mic_vad_attack_ms,
        vad_release_ms=config.mic_vad_release_ms,
        pre_roll_ms=config.mic_pre_roll_ms,
        min_utterance_ms=config.mic_min_utterance_ms,
    )

    logging.info("MIC mode active (always listening, no wake word). Press Ctrl+C to exit.")
    if motion_manager is not None:
        motion_manager.set_state(state_machine.state)

    microphone.start()
    try:
        while True:
            try:
                wav_bytes = utterance_queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                user_text = stt.transcribe(wav_bytes, language=config.stt_language)
            except Exception as exc:
                logging.warning("STT failed: %s", exc)
                continue

            user_text = user_text.strip()
            if not user_text:
                continue

            logging.info("User said: %s", user_text)

            _apply_event(state_machine, Event.WAKE_WORD, motion_manager)
            _ = reachy.execute_action({"type": "gesture.listening"})
            _apply_event(state_machine, Event.STT_RECEIVED, motion_manager)
            _ = reachy.execute_action({"type": "gesture.think"})

            if direct_llm is not None:
                try:
                    reply = direct_llm.generate_reply(user_text, session_id)
                except Exception as exc:
                    _apply_event(state_machine, Event.MODEL_ERROR, motion_manager)
                    voice.speak(f"Erro ao chamar LLM: {exc}")
                    _apply_event(state_machine, Event.RESPONSE_READY, motion_manager)
                    continue

                _apply_event(state_machine, Event.RESPONSE_READY, motion_manager)
                voice.speak(reply)
                _apply_event(state_machine, Event.RESPONSE_READY, motion_manager)
                continue

            if openclaw is None:
                _apply_event(state_machine, Event.MODEL_ERROR, motion_manager)
                voice.speak("OpenClaw nao configurado")
                _apply_event(state_machine, Event.RESPONSE_READY, motion_manager)
                continue

            try:
                response = openclaw.send_text(user_text, session_id)
            except Exception as exc:
                _apply_event(state_machine, Event.MODEL_ERROR, motion_manager)
                voice.speak(f"Erro ao chamar OpenClaw: {exc}")
                _apply_event(state_machine, Event.RESPONSE_READY, motion_manager)
                continue

            msg_type = response.get("type")
            payload = response.get("payload", {})
            if msg_type != "openclaw.intent":
                _apply_event(state_machine, Event.MODEL_ERROR, motion_manager)
                voice.speak(payload.get("message", "Resposta inesperada do OpenClaw"))
                _apply_event(state_machine, Event.RESPONSE_READY, motion_manager)
                continue

            summary = payload.get("summary", "")
            actions = payload.get("actions", [])
            _apply_event(state_machine, Event.RESPONSE_READY, motion_manager)
            _execute_actions(actions, reachy, voice, summary)
            _apply_event(state_machine, Event.RESPONSE_READY, motion_manager)
    finally:
        microphone.stop()
        if motion_manager is not None:
            motion_manager.stop()
        if camera_worker is not None:
            camera_worker.stop()


def _apply_event(
    state_machine: StateMachine,
    event: Event,
    motion_manager: Optional[MotionManager],
):
    state = state_machine.transition(event)
    if motion_manager is not None:
        motion_manager.set_state(state)
    return state


def _execute_actions(
    actions: List[Dict[str, Any]],
    reachy: ReachyClient,
    voice: VoicePipeline,
    summary: str,
) -> None:
    if not actions:
        if summary:
            voice.speak(summary)
        return

    for action in actions:
        action_type = (action.get("type") or action.get("action") or "").strip().lower()
        if action_type.startswith("gesture."):
            _ = reachy.execute_action(action)
            continue
        if action_type == "speech.say":
            text = str(action.get("text", "")).strip()
            if text:
                voice.speak(text)
            continue


if __name__ == "__main__":
    main()
