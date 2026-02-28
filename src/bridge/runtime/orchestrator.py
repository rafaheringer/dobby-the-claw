"""Shared runtime orchestrator for chat and realtime modes."""

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
from bridge.state_machine import Event, StateMachine
from bridge.tools import ToolRegistry

from bridge.runtime.audio_support import (
    build_tool_registry,
    build_wakeword_detector,
    maybe_log_health,
    process_audio_queue,
    require_sdk_instance,
    resolve_llm_api_key,
)
from bridge.runtime.common import apply_event
from bridge.runtime.ports import (
    ConversationCallbacks,
    ConversationSessionFactoryPort,
    ConversationSessionPort,
    RobotActionsPort,
)
from bridge.runtime.reachy_actions_adapter import ReachyRobotActions
from bridge.runtime.realtime_session_adapter import OpenAIRealtimeSessionFactory


class RuntimeOrchestrator:
    """Coordinate shared runtime behavior across chat and realtime modes."""

    def __init__(
        self,
        *,
        mode_name: str,
        state_machine: StateMachine,
        reachy: ReachyClient,
        motion_manager: Optional[MotionManager],
        camera_worker: Optional[CameraWorker],
        config: BridgeConfig,
        reachy_sdk_instance: Any,
        identity_prompt: str,
        idle_sleep_timeout_s: float,
        interactive_text: bool,
        active_mode_uses_mic_recording: bool,
        conversation_factory: ConversationSessionFactoryPort | None = None,
        robot_actions: RobotActionsPort | None = None,
    ) -> None:
        self.mode_name = mode_name
        self.state_machine = state_machine
        self.motion_manager = motion_manager
        self.camera_worker = camera_worker
        self.config = config
        self.reachy_sdk_instance = reachy_sdk_instance
        self.identity_prompt = identity_prompt
        self.idle_sleep_timeout_s = idle_sleep_timeout_s
        self.interactive_text = interactive_text
        self.active_mode_uses_mic_recording = active_mode_uses_mic_recording
        self.conversation_factory = conversation_factory or OpenAIRealtimeSessionFactory(config)
        self.robot_actions = robot_actions or ReachyRobotActions(reachy)

        require_sdk_instance(mode_name, reachy_sdk_instance)
        self.api_key = resolve_llm_api_key(config)

        if self.motion_manager is not None:
            self.motion_manager.set_state(self.state_machine.state)

        self.audio_queue: "Queue[tuple[str, Any]]" = Queue()
        self.text_queue: "Queue[str]" = Queue()
        self.input_stop = threading.Event()

        self.playback_started = False
        self.loop_start = time.monotonic()
        self.last_health_log = self.loop_start
        self.audio_chunks_total = 0
        self.responses_streamed = 0
        self.text_turns = 0
        self.recording_started = False
        self.sleeping = False
        self.realtime: ConversationSessionPort | None = None
        self.last_user_activity = time.monotonic()

        self.output_sample_rate = int(reachy_sdk_instance.media.get_output_audio_samplerate())
        self.realtime_output_rate = 24000
        self.input_sample_rate = int(reachy_sdk_instance.media.get_input_audio_samplerate())

        self.tool_registry: ToolRegistry = build_tool_registry(config, camera_worker)
        self.tool_specs = self.tool_registry.openai_specs()
        self.wakeword = build_wakeword_detector(config)
        self.idle_sleep_enabled = idle_sleep_timeout_s > 0.0 and config.offline_wakeword_enabled

        if idle_sleep_timeout_s > 0.0 and not config.offline_wakeword_enabled:
            logging.warning("Idle sleep timeout configured, but OFFLINE_WAKEWORD_ENABLED=false; idle sleep disabled")

    def run(self) -> None:
        """Run the orchestrator event loop until interrupted/exit."""
        self._log_mode_start()
        self._start_active_session()

        if self.interactive_text:
            logging.info("Type your message and press Enter. Use /quit to exit chat mode.")
            threading.Thread(target=self._input_worker, daemon=True).start()

        try:
            while True:
                sample = self.reachy_sdk_instance.media.get_audio_sample()
                if sample is not None:
                    self.wakeword.observe_sample(self.input_sample_rate, sample)

                if self.sleeping:
                    if sample is not None and self.wakeword.process_sample(self.input_sample_rate, sample):
                        self._wake_from_sleep_mode()
                    time.sleep(0.01)
                    continue

                if sample is not None and self.realtime is not None and self.active_mode_uses_mic_recording:
                    self.realtime.feed_audio(self.input_sample_rate, sample)

                self.playback_started, self.audio_chunks_total, self.responses_streamed = process_audio_queue(
                    audio_queue=self.audio_queue,
                    reachy_sdk_instance=self.reachy_sdk_instance,
                    output_sample_rate=self.output_sample_rate,
                    realtime_output_rate=self.realtime_output_rate,
                    playback_started=self.playback_started,
                    audio_chunks_total=self.audio_chunks_total,
                    responses_streamed=self.responses_streamed,
                    elapsed_ms=self._elapsed_ms,
                    state_machine=self.state_machine,
                    motion_manager=self.motion_manager,
                )

                if self.interactive_text:
                    should_exit = self._drain_text_queue()
                    if should_exit:
                        return

                now = time.monotonic()
                if self.realtime is not None:
                    self.last_health_log = maybe_log_health(
                        now=now,
                        last_health_log=self.last_health_log,
                        elapsed_ms=self._elapsed_ms,
                        realtime=self.realtime,
                        playback_started=self.playback_started,
                        audio_queue=self.audio_queue,
                        audio_chunks_total=self.audio_chunks_total,
                        responses_streamed=self.responses_streamed,
                        camera_worker=self.camera_worker,
                        text_queue=self.text_queue if self.interactive_text else None,
                        text_turns=self.text_turns if self.interactive_text else None,
                    )

                if self.idle_sleep_enabled and (now - self.last_user_activity) >= self.idle_sleep_timeout_s:
                    self._enter_sleep_mode()

                time.sleep(0.01)
        finally:
            self.input_stop.set()
            if self.playback_started:
                try:
                    self.reachy_sdk_instance.media.stop_playing()
                except Exception:
                    pass
            if self.realtime is not None:
                self.realtime.stop()
            if self.recording_started:
                try:
                    self.reachy_sdk_instance.media.stop_recording()
                except Exception:
                    pass
            if self.motion_manager is not None:
                self.motion_manager.stop()
            if self.camera_worker is not None:
                self.camera_worker.stop()

    def _elapsed_ms(self) -> int:
        return int((time.monotonic() - self.loop_start) * 1000)

    def _mark_user_activity(self) -> None:
        self.last_user_activity = time.monotonic()

    def _on_speech_start(self) -> None:
        logging.debug("[%dms] Callback speech_start", self._elapsed_ms())
        self._mark_user_activity()
        self.audio_queue.put(("force_stop", None))
        try:
            apply_event(self.state_machine, Event.WAKE_WORD, self.motion_manager)
            self.robot_actions.gesture_listening()
        except Exception as exc:
            logging.warning("[%dms] gesture.listening failed: %s", self._elapsed_ms(), exc)

    def _on_user_text(self, text: str, *, source: str) -> None:
        self._mark_user_activity()
        logging.info("[%dms] User %s: %s", self._elapsed_ms(), source, text)
        try:
            apply_event(self.state_machine, Event.STT_RECEIVED, self.motion_manager)
            self.robot_actions.gesture_think()
        except Exception as exc:
            logging.warning("[%dms] gesture.think failed: %s", self._elapsed_ms(), exc)

    def _on_assistant_text(self, text: str) -> None:
        logging.info("[%dms] Assistant text: %s", self._elapsed_ms(), text)

    def _on_assistant_audio_chunk(self, chunk) -> None:
        self.audio_queue.put(("chunk", chunk))

    def _on_assistant_audio_done(self) -> None:
        logging.debug("[%dms] Assistant audio done queued", self._elapsed_ms())
        self.audio_queue.put(("done", None))

    def _on_error(self, message: str) -> None:
        logging.warning("[%dms] Realtime error: %s", self._elapsed_ms(), message)

    def _start_active_session(self) -> None:
        callbacks = ConversationCallbacks(
            on_speech_start=self._on_speech_start if self.active_mode_uses_mic_recording else None,
            on_user_transcript=(
                (lambda text: self._on_user_text(text, source="said"))
                if self.active_mode_uses_mic_recording
                else None
            ),
            on_assistant_text=self._on_assistant_text,
            on_assistant_audio_chunk=self._on_assistant_audio_chunk,
            on_assistant_audio_done=self._on_assistant_audio_done,
            on_error=self._on_error,
        )
        self.realtime = self.conversation_factory.create(
            api_key=self.api_key,
            identity_prompt=self.identity_prompt,
            tool_specs=self.tool_specs,
            on_tool_call=self.tool_registry.execute,
            callbacks=callbacks,
        )
        self.realtime.start()
        if not self.realtime.wait_until_ready(timeout_s=8.0):
            self.realtime.stop()
            self.realtime = None
            raise RuntimeError("Failed to start OpenAI Realtime session")

        logging.info("[%dms] Realtime session connected", self._elapsed_ms())

        if self.active_mode_uses_mic_recording and not self.recording_started:
            self.reachy_sdk_instance.media.start_recording()
            self.recording_started = True
            logging.info("[%dms] Reachy microphone recording started", self._elapsed_ms())

        self.playback_started = False
        self.sleeping = False
        self.wakeword.reset()
        self.wakeword.start_auto_calibration()

    def _enter_sleep_mode(self) -> None:
        if self.sleeping:
            return

        logging.info("[%dms] Entering sleep mode after %.1fs idle", self._elapsed_ms(), self.idle_sleep_timeout_s)

        if self.playback_started:
            try:
                self.reachy_sdk_instance.media.stop_playing()
            except Exception:
                pass
            self.playback_started = False

        if self.realtime is not None:
            self.realtime.stop()
            self.realtime = None

        if self.active_mode_uses_mic_recording and self.recording_started:
            try:
                self.reachy_sdk_instance.media.stop_recording()
            except Exception:
                pass
            self.recording_started = False

        if self.motion_manager is not None:
            self.motion_manager.stop()
        if self.camera_worker is not None:
            self.camera_worker.stop()

        try:
            self.robot_actions.goto_sleep()
        except Exception as exc:
            logging.warning("[%dms] Failed to send goto_sleep: %s", self._elapsed_ms(), exc)

        if not self.recording_started:
            try:
                self.reachy_sdk_instance.media.start_recording()
                self.recording_started = True
            except Exception as exc:
                logging.warning("[%dms] Failed to start low-power microphone recording: %s", self._elapsed_ms(), exc)

        apply_event(self.state_machine, Event.RESET, self.motion_manager)
        self.sleeping = True
        self.wakeword.reset()

    def _wake_from_sleep_mode(self) -> None:
        logging.info("[%dms] Offline wakeword detected, waking up", self._elapsed_ms())

        if self.recording_started:
            try:
                self.reachy_sdk_instance.media.stop_recording()
            except Exception:
                pass
            self.recording_started = False

        try:
            self.robot_actions.wake_up()
        except Exception as exc:
            logging.warning("[%dms] Failed to send wake_up: %s", self._elapsed_ms(), exc)

        if self.camera_worker is not None:
            self.camera_worker.start()
        if self.motion_manager is not None:
            self.motion_manager.start()

        self._start_active_session()
        self._mark_user_activity()
        self.sleeping = False

    def _drain_text_queue(self) -> bool:
        while True:
            try:
                user_text = self.text_queue.get_nowait()
            except Empty:
                return False

            text = user_text.strip()
            if not text:
                continue
            if text.lower() in {"/quit", "/exit"}:
                logging.info("[%dms] Chat mode exit requested", self._elapsed_ms())
                return True

            if self.sleeping:
                logging.info("[%dms] Ignoring text input while sleeping; say wakeword to wake", self._elapsed_ms())
                continue

            self.audio_queue.put(("force_stop", None))
            try:
                apply_event(self.state_machine, Event.WAKE_WORD, self.motion_manager)
                self.robot_actions.gesture_listening()
            except Exception as exc:
                logging.warning("[%dms] gesture.listening failed: %s", self._elapsed_ms(), exc)

            self._on_user_text(text, source="typed")
            sent = self.realtime.send_text(text) if self.realtime is not None else False
            if not sent:
                logging.warning("[%dms] Failed to queue chat turn", self._elapsed_ms())
                continue

            self.text_turns += 1

    def _input_worker(self) -> None:
        while not self.input_stop.is_set():
            try:
                line = input("You> ")
            except EOFError:
                self.text_queue.put("/quit")
                return
            except KeyboardInterrupt:
                self.text_queue.put("/quit")
                return
            self.text_queue.put(line)

    def _log_mode_start(self) -> None:
        if self.interactive_text:
            logging.info(
                "Chat mode active model=%s transcribe=%s out_sr=%s tools=%s idle_sleep=%s",
                self.config.realtime_model,
                self.config.realtime_transcription_model,
                self.output_sample_rate,
                self.tool_registry.names(),
                self.idle_sleep_enabled,
            )
            return

        logging.info(
            "Realtime mode active model=%s transcribe=%s silence=%sms padding=%sms out_sr=%s tools=%s idle_sleep=%s",
            self.config.realtime_model,
            self.config.realtime_transcription_model,
            self.config.realtime_vad_silence_ms,
            self.config.realtime_vad_prefix_padding_ms,
            self.output_sample_rate,
            self.tool_registry.names(),
            self.idle_sleep_enabled,
        )
