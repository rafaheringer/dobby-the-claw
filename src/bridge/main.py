import argparse
import logging
import os
from pathlib import Path
import time
from queue import Empty, Queue
from typing import Any, Optional

import numpy as np

from bridge.config import BridgeConfig
from bridge.reachy.camera_worker import CameraWorker
from bridge.reachy.client import ReachyClient
from bridge.reachy.motion import MotionManager
from bridge.reachy.realtime_client import OpenAIRealtimeSession
from bridge.state_machine import StateMachine
from bridge.state_machine import Event
from bridge.tools import CameraSnapshotTool, ToolRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Dobby bridge")
    parser.add_argument(
        "--mode",
        choices=["idle", "realtime"],
        default=os.getenv("BRIDGE_MODE", "realtime"),
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
    logging.info("Reachy Bridge API: %s", config.reachy_bridge_url)
    logging.info("Run mode: %s", args.mode)
    logging.info(
        "Vision debug window=%s log_interval=%.1fs",
        config.vision_debug_window,
        config.vision_debug_log_interval_s,
    )

    state_machine = StateMachine()
    identity_prompt = _load_identity_prompt()

    reachy = ReachyClient(config.reachy_bridge_url)
    reachy_sdk_instance = None
    camera_worker = None
    motion_manager = None
    if config.reachy_bridge_url.strip().lower().startswith("sdk"):
        try:
            reachy_sdk_instance = reachy.get_sdk_instance()
            camera_worker = CameraWorker(
                reachy_sdk_instance,
                debug_visual_window=config.vision_debug_window,
                debug_log_interval_s=config.vision_debug_log_interval_s,
            )
            camera_worker.start()
            motion_manager = MotionManager(reachy_sdk_instance, camera_worker=camera_worker)
            motion_manager.start()
        except Exception:
            reachy_sdk_instance = None
            camera_worker = None
            motion_manager = None

    if args.mode == "realtime":
        _run_realtime_loop(
            state_machine,
            reachy,
            motion_manager,
            camera_worker,
            config,
            reachy_sdk_instance,
            identity_prompt,
        )
        return

    _ = (state_machine, reachy, motion_manager, camera_worker)

    while True:
        time.sleep(1)


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
    reachy: ReachyClient,
    motion_manager: Optional[MotionManager],
    camera_worker: Optional[CameraWorker],
    config: BridgeConfig,
    reachy_sdk_instance,
    identity_prompt: str,
) -> None:
    """Run low-latency realtime mode using OpenAI Realtime API."""
    if reachy_sdk_instance is None:
        raise RuntimeError("Realtime mode requires REACHY_BRIDGE_URL=sdk")

    api_key = os.getenv(config.llm_api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {config.llm_api_key_env}")

    if motion_manager is not None:
        motion_manager.set_state(state_machine.state)

    audio_queue: "Queue[tuple[str, Any]]" = Queue()
    playback_started = False
    loop_start = time.monotonic()
    last_health_log = loop_start
    audio_chunks_total = 0
    responses_streamed = 0
    output_sample_rate = int(reachy_sdk_instance.media.get_output_audio_samplerate())
    realtime_output_rate = 24000

    tool_registry = ToolRegistry()
    if config.camera_tool_enabled and camera_worker is not None:
        tool_registry.register(CameraSnapshotTool(camera_worker))
    tool_specs = tool_registry.openai_specs()

    def _elapsed_ms() -> int:
        return int((time.monotonic() - loop_start) * 1000)

    def _on_speech_start() -> None:
        logging.debug("[%dms] Callback speech_start", _elapsed_ms())
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
        logging.info("[%dms] Assistant text: %s", _elapsed_ms(), text)

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
        tool_specs=tool_specs,
        on_tool_call=tool_registry.execute,
    )

    logging.info(
        "Realtime mode active model=%s transcribe=%s silence=%sms padding=%sms out_sr=%s tools=%s",
        config.realtime_model,
        config.realtime_transcription_model,
        config.realtime_vad_silence_ms,
        config.realtime_vad_prefix_padding_ms,
        output_sample_rate,
        tool_registry.names(),
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
                    audio_chunks_total += 1
                elif kind == "done":
                    responses_streamed += 1
                    logging.debug("[%dms] Streamed response completed", _elapsed_ms())
                    _apply_event(state_machine, Event.RESPONSE_READY, motion_manager)
                elif kind == "force_stop":
                    if playback_started:
                        logging.info("[%dms] Playback stopped reason=force_stop", _elapsed_ms())
                        reachy_sdk_instance.media.stop_playing()
                        playback_started = False

            now = time.monotonic()
            if (now - last_health_log) > 5.0:
                vision_info = None
                if camera_worker is not None:
                    try:
                        vision_info = camera_worker.get_tracking_debug_snapshot()
                    except Exception:
                        vision_info = None
                logging.debug(
                    "[%dms] Health ready=%s playback=%s audio_q=%s chunks=%s streamed=%s face=%s eye=%s",
                    _elapsed_ms(),
                    realtime.wait_until_ready(timeout_s=0.0),
                    playback_started,
                    audio_queue.qsize(),
                    audio_chunks_total,
                    responses_streamed,
                    (vision_info or {}).get("face_detected_recently"),
                    (vision_info or {}).get("eye_center"),
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
