"""Terminal text chat mode runtime loop."""

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import Any, Optional

from bridge.config import BridgeConfig
from bridge.reachy.camera_worker import CameraWorker
from bridge.reachy.client import ReachyClient
from bridge.reachy.motion import MotionManager
from bridge.reachy.realtime_client import OpenAIRealtimeSession
from bridge.state_machine import Event, StateMachine
from bridge.tools import ToolRegistry

from bridge.runtime.audio_support import (
    build_tool_registry,
    maybe_log_health,
    process_audio_queue,
    require_sdk_instance,
    resolve_llm_api_key,
    shutdown_runtime,
)
from bridge.runtime.common import apply_event


def run_chat_loop(
    state_machine: StateMachine,
    reachy: ReachyClient,
    motion_manager: Optional[MotionManager],
    camera_worker: Optional[CameraWorker],
    config: BridgeConfig,
    reachy_sdk_instance: Any,
    identity_prompt: str,
) -> None:
    """Run terminal chat mode with realtime model and Reachy audio output."""
    require_sdk_instance("Chat", reachy_sdk_instance)
    api_key = resolve_llm_api_key(config)

    if motion_manager is not None:
        motion_manager.set_state(state_machine.state)

    audio_queue: "Queue[tuple[str, Any]]" = Queue()
    text_queue: "Queue[str]" = Queue()
    input_stop = threading.Event()
    playback_started = False
    loop_start = time.monotonic()
    last_health_log = loop_start
    audio_chunks_total = 0
    responses_streamed = 0
    text_turns = 0
    output_sample_rate = int(reachy_sdk_instance.media.get_output_audio_samplerate())
    realtime_output_rate = 24000

    tool_registry: ToolRegistry = build_tool_registry(config, camera_worker)
    tool_specs = tool_registry.openai_specs()

    def _elapsed_ms() -> int:
        return int((time.monotonic() - loop_start) * 1000)

    def _on_user_text(text: str) -> None:
        logging.info("[%dms] User typed: %s", _elapsed_ms(), text)
        try:
            apply_event(state_machine, Event.STT_RECEIVED, motion_manager)
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
        on_user_transcript=_on_user_text,
        on_assistant_text=_on_assistant_text,
        on_assistant_audio_chunk=_on_assistant_audio_chunk,
        on_assistant_audio_done=_on_assistant_audio_done,
        on_error=_on_error,
        tool_specs=tool_specs,
        on_tool_call=tool_registry.execute,
    )

    logging.info(
        "Chat mode active model=%s transcribe=%s out_sr=%s tools=%s",
        config.realtime_model,
        config.realtime_transcription_model,
        output_sample_rate,
        tool_registry.names(),
    )

    realtime.start()
    if not realtime.wait_until_ready(timeout_s=8.0):
        realtime.stop()
        raise RuntimeError("Failed to start OpenAI Realtime session")
    logging.info("[%dms] Realtime session connected", _elapsed_ms())
    logging.info("Type your message and press Enter. Use /quit to exit chat mode.")

    def _input_worker() -> None:
        while not input_stop.is_set():
            try:
                line = input("You> ")
            except EOFError:
                text_queue.put("/quit")
                return
            except KeyboardInterrupt:
                text_queue.put("/quit")
                return
            text_queue.put(line)

    threading.Thread(target=_input_worker, daemon=True).start()

    try:
        while True:
            playback_started, audio_chunks_total, responses_streamed = process_audio_queue(
                audio_queue=audio_queue,
                reachy_sdk_instance=reachy_sdk_instance,
                output_sample_rate=output_sample_rate,
                realtime_output_rate=realtime_output_rate,
                playback_started=playback_started,
                audio_chunks_total=audio_chunks_total,
                responses_streamed=responses_streamed,
                elapsed_ms=_elapsed_ms,
                state_machine=state_machine,
                motion_manager=motion_manager,
            )

            while True:
                try:
                    user_text = text_queue.get_nowait()
                except Empty:
                    break

                text = user_text.strip()
                if not text:
                    continue
                if text.lower() in {"/quit", "/exit"}:
                    logging.info("[%dms] Chat mode exit requested", _elapsed_ms())
                    return

                audio_queue.put(("force_stop", None))
                try:
                    apply_event(state_machine, Event.WAKE_WORD, motion_manager)
                    _ = reachy.execute_action({"type": "gesture.listening"})
                except Exception as exc:
                    logging.warning("[%dms] gesture.listening failed: %s", _elapsed_ms(), exc)

                _on_user_text(text)
                sent = realtime.send_text(text)
                if not sent:
                    logging.warning("[%dms] Failed to queue chat turn", _elapsed_ms())
                else:
                    text_turns += 1

            now = time.monotonic()
            last_health_log = maybe_log_health(
                now=now,
                last_health_log=last_health_log,
                elapsed_ms=_elapsed_ms,
                realtime=realtime,
                playback_started=playback_started,
                audio_queue=audio_queue,
                audio_chunks_total=audio_chunks_total,
                responses_streamed=responses_streamed,
                camera_worker=camera_worker,
                text_queue=text_queue,
                text_turns=text_turns,
            )

            time.sleep(0.01)
    finally:
        input_stop.set()
        shutdown_runtime(
            playback_started=playback_started,
            reachy_sdk_instance=reachy_sdk_instance,
            realtime=realtime,
            motion_manager=motion_manager,
            camera_worker=camera_worker,
            stop_recording=False,
        )
