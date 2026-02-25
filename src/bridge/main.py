import argparse
from queue import Empty, Queue
import logging
import os
from pathlib import Path
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from bridge.config import BridgeConfig
from bridge.llm_client import DirectLLMClient
from bridge.openclaw_client import OpenClawClient
from bridge.reachy.camera_worker import CameraWorker
from bridge.reachy.client import ReachyClient
from bridge.reachy.microphone import MicrophoneListener
from bridge.reachy.motion import MotionManager
from bridge.reachy.realtime_client import OpenAIRealtimeSession
from bridge.reachy.stt_client import OpenAISttClient
from bridge.reachy.voice import VoicePipeline
from bridge.state_machine import StateMachine
from bridge.state_machine import Event


def main() -> None:
    parser = argparse.ArgumentParser(description="Dobby bridge")
    parser.add_argument(
        "--mode",
        choices=["idle", "cli", "mic", "realtime"],
        default=os.getenv("BRIDGE_MODE", "mic"),
    )
    parser.add_argument(
        "--llm-mode",
        choices=["openclaw", "direct"],
        default=None,
    )
    args = parser.parse_args()

    log_level_name = os.getenv("BRIDGE_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
    )
    _configure_third_party_loggers(log_level)
    config = BridgeConfig.from_env()

    logging.info("Bridge starting")
    logging.info("Log level: %s", log_level_name)
    logging.info("OpenClaw API: %s", config.openclaw_api_url)
    logging.info("Reachy Bridge API: %s", config.reachy_bridge_url)
    logging.info("Run mode: %s", args.mode)

    state_machine = StateMachine()
    identity_prompt = _load_identity_prompt()
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
            system_prompt=identity_prompt,
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

    if args.mode == "realtime":
        _run_realtime_loop(
            state_machine,
            direct_llm,
            reachy,
            voice,
            motion_manager,
            camera_worker,
            config,
            reachy_sdk_instance,
            identity_prompt,
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
        prompt=config.stt_prompt,
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
                wav_bytes = utterance_queue.get(timeout=0.05)
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
                    reply = direct_llm.generate_reply(
                        user_text,
                        session_id,
                        language=config.llm_language,
                    )
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


def _run_realtime_loop(
    state_machine: StateMachine,
    direct_llm: Optional[DirectLLMClient],
    reachy: ReachyClient,
    voice: VoicePipeline,
    motion_manager: Optional[MotionManager],
    camera_worker: Optional[CameraWorker],
    config: BridgeConfig,
    reachy_sdk_instance,
    identity_prompt: str,
) -> None:
    """Run low-latency realtime mode using OpenAI Realtime API."""
    if direct_llm is not None:
        logging.info("Realtime mode uses OpenAI Realtime directly (ignoring direct responses mode).")
    if reachy_sdk_instance is None:
        raise RuntimeError("Realtime mode requires REACHY_BRIDGE_URL=sdk")

    api_key = os.getenv(config.llm_api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {config.llm_api_key_env}")

    if motion_manager is not None:
        motion_manager.set_state(state_machine.state)

    assistant_queue: "Queue[str]" = Queue()
    audio_queue: "Queue[tuple[str, Any]]" = Queue()
    playback_started = False
    last_audio_chunk_ts = 0.0
    loop_start = time.monotonic()
    last_health_log = loop_start
    audio_chunks_total = 0
    responses_streamed = 0
    responses_fallback_tts = 0
    output_sample_rate = int(reachy_sdk_instance.media.get_output_audio_samplerate())
    realtime_output_rate = 24000

    def _elapsed_ms() -> int:
        return int((time.monotonic() - loop_start) * 1000)

    def _on_speech_start() -> None:
        logging.debug("[%dms] Callback speech_start", _elapsed_ms())
        try:
            if voice.is_speaking():
                voice.interrupt()
        except Exception as exc:
            logging.warning("[%dms] voice.interrupt failed: %s", _elapsed_ms(), exc)
        audio_queue.put(("force_stop", None))
        try:
            _apply_event(state_machine, Event.WAKE_WORD, motion_manager)
            _ = reachy.execute_action({"type": "gesture.listening"})
        except Exception as exc:
            logging.warning("[%dms] gesture.listening failed: %s", _elapsed_ms(), exc)

    def _on_user_text(text: str) -> None:
        logging.info("[%dms] User said: %s", _elapsed_ms(), text)
        try:
            _apply_event(state_machine, Event.STT_RECEIVED, motion_manager)
            _ = reachy.execute_action({"type": "gesture.think"})
        except Exception as exc:
            logging.warning("[%dms] gesture.think failed: %s", _elapsed_ms(), exc)

    def _on_assistant_text(text: str) -> None:
        logging.debug("[%dms] Assistant text queued len=%s", _elapsed_ms(), len(text))
        assistant_queue.put(text)

    def _on_assistant_audio_chunk(chunk) -> None:
        audio_queue.put(("chunk", chunk))

    def _on_assistant_audio_done() -> None:
        logging.debug("[%dms] Assistant audio done queued", _elapsed_ms())
        audio_queue.put(("done", None))

    def _on_error(message: str) -> None:
        logging.warning("[%dms] Realtime error: %s", _elapsed_ms(), message)

    realtime = OpenAIRealtimeSession(
        api_key=api_key,
        api_base=config.llm_api_base,
        model=config.realtime_model,
        instructions=identity_prompt,
        language=config.stt_language,
        transcription_model=config.realtime_transcription_model,
        vad_silence_ms=config.realtime_vad_silence_ms,
        vad_prefix_padding_ms=config.realtime_vad_prefix_padding_ms,
        on_speech_start=_on_speech_start,
        on_user_transcript=_on_user_text,
        on_assistant_text=_on_assistant_text,
        on_assistant_audio_chunk=_on_assistant_audio_chunk,
        on_assistant_audio_done=_on_assistant_audio_done,
        on_error=_on_error,
    )

    logging.info(
        "Realtime mode active model=%s transcribe=%s silence=%sms padding=%sms out_sr=%s",
        config.realtime_model,
        config.realtime_transcription_model,
        config.realtime_vad_silence_ms,
        config.realtime_vad_prefix_padding_ms,
        output_sample_rate,
    )

    realtime.start()
    if not realtime.wait_until_ready(timeout_s=8.0):
        realtime.stop()
        raise RuntimeError("Failed to start OpenAI Realtime session")
    logging.info("[%dms] Realtime session connected", _elapsed_ms())

    reachy_sdk_instance.media.start_recording()
    logging.info("[%dms] Reachy microphone recording started", _elapsed_ms())
    try:
        while True:
            sample = reachy_sdk_instance.media.get_audio_sample()
            if sample is not None:
                sample_rate = int(reachy_sdk_instance.media.get_input_audio_samplerate())
                realtime.feed_audio(sample_rate, sample)

            while True:
                try:
                    kind, payload = audio_queue.get_nowait()
                except Empty:
                    break

                if kind == "chunk":
                    if not playback_started:
                        logging.info("[%dms] Playback started", _elapsed_ms())
                        reachy_sdk_instance.media.start_playing()
                        playback_started = True
                    chunk = payload
                    if output_sample_rate != realtime_output_rate:
                        chunk = _resample_audio_chunk(chunk, realtime_output_rate, output_sample_rate)
                    reachy_sdk_instance.media.push_audio_sample(chunk)
                    last_audio_chunk_ts = time.monotonic()
                    audio_chunks_total += 1
                elif kind == "done":
                    responses_streamed += 1
                    logging.debug("[%dms] Streamed response completed", _elapsed_ms())
                elif kind == "force_stop":
                    if playback_started:
                        logging.info("[%dms] Playback stopped reason=force_stop", _elapsed_ms())
                        reachy_sdk_instance.media.stop_playing()
                        playback_started = False

            try:
                assistant_text = assistant_queue.get_nowait().strip()
            except Empty:
                assistant_text = ""

            if assistant_text:
                if (time.monotonic() - last_audio_chunk_ts) > 0.8:
                    logging.info("[%dms] Fallback TTS response", _elapsed_ms())
                    _apply_event(state_machine, Event.RESPONSE_READY, motion_manager)
                    voice.speak(assistant_text)
                    _apply_event(state_machine, Event.RESPONSE_READY, motion_manager)
                    responses_fallback_tts += 1

            now = time.monotonic()
            if (now - last_health_log) > 5.0:
                logging.debug(
                    "[%dms] Health ready=%s playback=%s audio_q=%s text_q=%s chunks=%s streamed=%s fallback=%s",
                    _elapsed_ms(),
                    realtime.wait_until_ready(timeout_s=0.0),
                    playback_started,
                    audio_queue.qsize(),
                    assistant_queue.qsize(),
                    audio_chunks_total,
                    responses_streamed,
                    responses_fallback_tts,
                )
                last_health_log = now

            time.sleep(0.01)
    finally:
        if playback_started:
            try:
                reachy_sdk_instance.media.stop_playing()
            except Exception:
                pass
        realtime.stop()
        try:
            reachy_sdk_instance.media.stop_recording()
        except Exception:
            pass
        if motion_manager is not None:
            motion_manager.stop()
        if camera_worker is not None:
            camera_worker.stop()


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


def _resample_audio_chunk(chunk: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample mono float32 chunk from src_rate to dst_rate."""
    if src_rate == dst_rate:
        return chunk
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.size < 2:
        return chunk
    duration = chunk.size / float(src_rate)
    target_size = max(1, int(duration * dst_rate))
    source_x = np.linspace(0.0, 1.0, num=chunk.size, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, num=target_size, dtype=np.float32)
    return np.interp(target_x, source_x, chunk).astype(np.float32)


def _load_identity_prompt() -> str:
    """Load robot identity instructions from prompts/identity.txt."""
    src_root = Path(__file__).resolve().parents[1]
    identity_path = src_root / "prompts" / "identity.txt"
    try:
        content = identity_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Failed to load identity prompt at {identity_path}: {exc}") from exc

    if not content:
        raise RuntimeError(f"Identity prompt is empty: {identity_path}")

    return content


def _configure_third_party_loggers(app_log_level: int) -> None:
    """Keep app logs verbose while preventing third-party transport log spam."""
    noisy_loggers = {
        "websockets": logging.WARNING,
        "websockets.client": logging.WARNING,
        "websockets.protocol": logging.WARNING,
        "httpcore": logging.WARNING,
        "httpx": logging.WARNING,
        "openai": logging.INFO if app_log_level <= logging.DEBUG else logging.WARNING,
        "asyncio": logging.INFO if app_log_level <= logging.DEBUG else logging.WARNING,
    }
    for logger_name, logger_level in noisy_loggers.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logger_level)
        logger.propagate = True


if __name__ == "__main__":
    main()
