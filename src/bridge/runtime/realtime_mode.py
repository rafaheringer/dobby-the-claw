"""Realtime voice mode runtime loop."""

from __future__ import annotations

import logging
import time
from queue import Queue
from typing import Any, Optional

from bridge.config import BridgeConfig
from bridge.reachy.camera_worker import CameraWorker
from bridge.reachy.client import ReachyClient
from bridge.reachy.motion import MotionManager
from bridge.reachy.realtime_client import OpenAIRealtimeSession
from bridge.state_machine import Event, State, StateMachine
from bridge.tools import ToolRegistry

from bridge.runtime.audio_support import (
    build_realtime_session,
    build_tool_registry,
    build_wakeword_detector,
    maybe_log_health,
    process_audio_queue,
    require_sdk_instance,
    resolve_llm_api_key,
)
from bridge.runtime.common import apply_event


def run_realtime_loop(
    state_machine: StateMachine,
    reachy: ReachyClient,
    motion_manager: Optional[MotionManager],
    camera_worker: Optional[CameraWorker],
    config: BridgeConfig,
    reachy_sdk_instance: Any,
    identity_prompt: str,
    idle_sleep_timeout_s: float,
) -> None:
    """Run low-latency audio interaction mode with the OpenAI Realtime API."""
    require_sdk_instance("Realtime", reachy_sdk_instance)
    api_key = resolve_llm_api_key(config)

    if motion_manager is not None:
        motion_manager.set_state(state_machine.state)

    audio_queue: "Queue[tuple[str, Any]]" = Queue()
    playback_started = False
    loop_start = time.monotonic()
    last_health_log = loop_start
    audio_chunks_total = 0
    responses_streamed = 0
    recording_started = False
    sleeping = False
    realtime: OpenAIRealtimeSession | None = None
    last_user_activity = time.monotonic()
    output_sample_rate = int(reachy_sdk_instance.media.get_output_audio_samplerate())
    realtime_output_rate = 24000
    input_sample_rate = int(reachy_sdk_instance.media.get_input_audio_samplerate())

    tool_registry: ToolRegistry = build_tool_registry(config, camera_worker)
    tool_specs = tool_registry.openai_specs()
    wakeword = build_wakeword_detector(config)
    idle_sleep_enabled = idle_sleep_timeout_s > 0.0 and config.offline_wakeword_enabled
    if idle_sleep_timeout_s > 0.0 and not config.offline_wakeword_enabled:
        logging.warning("Idle sleep timeout configured, but OFFLINE_WAKEWORD_ENABLED=false; idle sleep disabled")

    def _elapsed_ms() -> int:
        return int((time.monotonic() - loop_start) * 1000)

    def _mark_user_activity() -> None:
        nonlocal last_user_activity
        last_user_activity = time.monotonic()

    def _on_speech_start() -> None:
        logging.debug("[%dms] Callback speech_start", _elapsed_ms())
        _mark_user_activity()
        audio_queue.put(("force_stop", None))
        try:
            apply_event(state_machine, Event.WAKE_WORD, motion_manager)
            _ = reachy.execute_action({"type": "gesture.listening"})
        except Exception as exc:
            logging.warning("[%dms] gesture.listening failed: %s", _elapsed_ms(), exc)

    def _on_user_text(text: str) -> None:
        _mark_user_activity()
        logging.info("[%dms] User said: %s", _elapsed_ms(), text)
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

    def _start_active_session() -> None:
        nonlocal realtime, recording_started, sleeping, playback_started
        realtime = build_realtime_session(
            api_key=api_key,
            config=config,
            identity_prompt=identity_prompt,
            tool_specs=tool_specs,
            on_tool_call=tool_registry.execute,
            on_speech_start=_on_speech_start,
            on_user_transcript=_on_user_text,
            on_assistant_text=_on_assistant_text,
            on_assistant_audio_chunk=_on_assistant_audio_chunk,
            on_assistant_audio_done=_on_assistant_audio_done,
            on_error=_on_error,
        )
        realtime.start()
        if not realtime.wait_until_ready(timeout_s=8.0):
            realtime.stop()
            realtime = None
            raise RuntimeError("Failed to start OpenAI Realtime session")
        logging.info("[%dms] Realtime session connected", _elapsed_ms())

        if not recording_started:
            reachy_sdk_instance.media.start_recording()
            recording_started = True
            logging.info("[%dms] Reachy microphone recording started", _elapsed_ms())

        playback_started = False
        sleeping = False
        wakeword.reset()
        wakeword.start_auto_calibration()

    def _enter_sleep_mode() -> None:
        nonlocal realtime, playback_started, recording_started, sleeping
        if sleeping:
            return

        logging.info("[%dms] Entering sleep mode after %.1fs idle", _elapsed_ms(), idle_sleep_timeout_s)

        if playback_started:
            try:
                reachy_sdk_instance.media.stop_playing()
            except Exception:
                pass
            playback_started = False

        if realtime is not None:
            realtime.stop()
            realtime = None

        if recording_started:
            try:
                reachy_sdk_instance.media.stop_recording()
            except Exception:
                pass
            recording_started = False

        if motion_manager is not None:
            motion_manager.stop()
        if camera_worker is not None:
            camera_worker.stop()

        try:
            reachy.goto_sleep()
        except Exception as exc:
            logging.warning("[%dms] Failed to send goto_sleep: %s", _elapsed_ms(), exc)

        try:
            reachy_sdk_instance.media.start_recording()
            recording_started = True
        except Exception as exc:
            logging.warning("[%dms] Failed to start low-power microphone recording: %s", _elapsed_ms(), exc)

        state_machine.state = State.IDLE
        sleeping = True
        wakeword.reset()

    def _wake_from_sleep_mode() -> None:
        nonlocal recording_started, sleeping
        logging.info("[%dms] Offline wakeword detected, waking up", _elapsed_ms())

        if recording_started:
            try:
                reachy_sdk_instance.media.stop_recording()
            except Exception:
                pass
            recording_started = False

        try:
            reachy.wake_up()
        except Exception as exc:
            logging.warning("[%dms] Failed to send wake_up: %s", _elapsed_ms(), exc)

        if camera_worker is not None:
            camera_worker.start()
        if motion_manager is not None:
            motion_manager.start()

        _start_active_session()
        _mark_user_activity()
        sleeping = False

    logging.info(
        "Realtime mode active model=%s transcribe=%s silence=%sms padding=%sms out_sr=%s tools=%s idle_sleep=%s",
        config.realtime_model,
        config.realtime_transcription_model,
        config.realtime_vad_silence_ms,
        config.realtime_vad_prefix_padding_ms,
        output_sample_rate,
        tool_registry.names(),
        idle_sleep_enabled,
    )

    _start_active_session()
    try:
        while True:
            sample = reachy_sdk_instance.media.get_audio_sample()
            if sample is not None:
                wakeword.observe_sample(input_sample_rate, sample)

            if sleeping:
                if sample is not None and wakeword.process_sample(input_sample_rate, sample):
                    _wake_from_sleep_mode()
                time.sleep(0.01)
                continue

            if sample is not None and realtime is not None:
                realtime.feed_audio(input_sample_rate, sample)

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

            now = time.monotonic()
            if realtime is not None:
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
                )

            if idle_sleep_enabled and (now - last_user_activity) >= idle_sleep_timeout_s:
                _enter_sleep_mode()

            time.sleep(0.01)
    finally:
        if playback_started:
            try:
                reachy_sdk_instance.media.stop_playing()
            except Exception:
                pass
        if realtime is not None:
            realtime.stop()
        if recording_started:
            try:
                reachy_sdk_instance.media.stop_recording()
            except Exception:
                pass
        if motion_manager is not None:
            motion_manager.stop()
        if camera_worker is not None:
            camera_worker.stop()
