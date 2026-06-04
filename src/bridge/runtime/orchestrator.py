"""Shared runtime orchestrator for chat and realtime modes."""

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import Any, Optional

from bridge.config import BridgeConfig
from reachy.camera_worker import CameraWorker
from reachy.client import ReachyClient
from reachy.motion import MotionManager
from bridge.state_machine import Event, StateMachine

from bridge.runtime.adapters.notification_server import NotificationServer
from bridge.runtime.adapters.reachy_actions import ReachyRobotActions
from bridge.runtime.adapters.reachy_media import build_reachy_media_io
from bridge.runtime.adapters.realtime_session import OpenAIRealtimeSessionFactory
from bridge.runtime.adapters.tool_runtime import build_tool_runtime
from bridge.runtime.audio_support import (
    build_wakeword_detector,
    maybe_log_health,
    process_audio_queue,
    require_sdk_instance,
    resolve_llm_api_key,
)
from homeassistant.home_assistant_client import HomeAssistantWsClient
from bridge.runtime.common import apply_event
from bridge.runtime.ports import (
    ConversationCallbacks,
    ConversationSessionFactoryPort,
    ConversationSessionPort,
    MediaIOPort,
    RobotActionsPort,
    ToolRuntimePort,
)


_conv_log = logging.getLogger("bridge.conversation")


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
        wake_on_startup: bool = True,
        conversation_factory: ConversationSessionFactoryPort | None = None,
        robot_actions: RobotActionsPort | None = None,
        tool_runtime: ToolRuntimePort | None = None,
        media_io: MediaIOPort | None = None,
        ha_client: HomeAssistantWsClient | None = None,
        skip_stdin: bool = False,
    ) -> None:
        """Initialize shared runtime dependencies and orchestration state."""
        self.mode_name = mode_name
        self.state_machine = state_machine
        self.motion_manager = motion_manager
        self.camera_worker = camera_worker
        self.config = config
        self.identity_prompt = identity_prompt
        self.idle_sleep_timeout_s = idle_sleep_timeout_s
        self.interactive_text = interactive_text
        self.active_mode_uses_mic_recording = active_mode_uses_mic_recording
        self.wake_on_startup = wake_on_startup
        self.conversation_factory = conversation_factory or OpenAIRealtimeSessionFactory(config)
        self.robot_actions = robot_actions or ReachyRobotActions(reachy)

        if media_io is None:
            require_sdk_instance(mode_name, reachy_sdk_instance)
            self.media_io = build_reachy_media_io(reachy_sdk_instance)
        else:
            self.media_io = media_io

        self.api_key = resolve_llm_api_key(config)

        self._current_speaker: str | None = None
        self._session_messages: list[dict] = []

        self._speaker_memory = None
        if config.speaker_memory_enabled:
            import os as _os
            from bridge.speaker_memory import SpeakerMemory
            self._speaker_memory = SpeakerMemory(
                storage_dir=_os.path.expanduser(config.speaker_memory_dir),
                extraction_model=config.speaker_memory_model,
                api_key=self.api_key,
            )

        self._notification_queue: "Queue[str]" = Queue()

        self.tool_runtime = tool_runtime or build_tool_runtime(
            config,
            camera_worker,
            motion_manager,
            speaker_memory=self._speaker_memory,
            notification_enqueue=self._notification_queue.put,
        )

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
        self.sleep_after_response = threading.Event()
        self._sleep_drain_at: float = 0.0  # monotonic timestamp; >0 means sleep is pending
        self._play_start_ts: float = 0.0   # when start_playing() was called for current response
        self._queued_samples: int = 0      # samples queued since last speech start (at realtime_output_rate)
        self.realtime: ConversationSessionPort | None = None
        self.last_user_activity = time.monotonic()
        self.realtime_recovery_requested = False
        self.realtime_last_error_message = ""
        self.realtime_next_recovery_at = 0.0
        self.pending_recovery_notice: str | None = None

        self.output_sample_rate = self.media_io.get_output_audio_samplerate()
        self.realtime_output_rate = 24000
        self.input_sample_rate = self.media_io.get_input_audio_samplerate()

        self._ha_client = ha_client
        self._skip_stdin = skip_stdin
        self._notification_server: NotificationServer | None = None
        self.tool_specs = self.tool_runtime.openai_specs()
        self.identity_prompt_runtime = self._build_runtime_identity_prompt(identity_prompt)
        self.wakeword = build_wakeword_detector(config)
        self.idle_sleep_enabled = idle_sleep_timeout_s > 0.0 and config.offline_wakeword_enabled

        if idle_sleep_timeout_s > 0.0 and not config.offline_wakeword_enabled:
            logging.warning("Idle sleep timeout configured, but OFFLINE_WAKEWORD_ENABLED=false; idle sleep disabled")

    def run(self) -> None:
        """Run the orchestrator event loop until interrupted/exit."""
        self._log_mode_start()

        try:
            self.robot_actions.enable_motors()
        except Exception as exc:
            logging.warning("[%dms] Failed to enable motors on startup: %s", self._elapsed_ms(), exc)

        if self.wake_on_startup:
            try:
                self.robot_actions.wake_up()
            except Exception as exc:
                logging.warning("[%dms] Failed to wake_up on startup: %s", self._elapsed_ms(), exc)

        if self.camera_worker is not None and self.config.speaker_id_enabled:
            try:
                speaker = self.camera_worker.identify_current_speaker(wait_s=0.5)
                self._current_speaker = speaker
                if speaker:
                    logging.info("[%dms] Startup speaker ID: %s", self._elapsed_ms(), speaker)
                else:
                    logging.info("[%dms] Startup speaker ID: no face detected", self._elapsed_ms())
            except Exception as exc:
                logging.warning("[%dms] Startup speaker ID failed: %s", self._elapsed_ms(), exc)

        self._start_active_session()

        if self.interactive_text and not self._skip_stdin:
            logging.info("Type your message and press Enter. Use /quit to exit chat mode.")
            threading.Thread(target=self._input_worker, daemon=True).start()

        if self.config.notification_server_port > 0:
            self._notification_server = NotificationServer(
                "0.0.0.0",
                self.config.notification_server_port,
                notification_queue=self._notification_queue,
            )
            try:
                self._notification_server.start()
            except OSError as exc:
                logging.warning(
                    "[%dms] Notification server failed to start on port %d: %s",
                    self._elapsed_ms(), self.config.notification_server_port, exc,
                )
                self._notification_server = None

        try:
            while True:
                try:
                    sample = self.media_io.get_audio_sample()
                except Exception as exc:
                    logging.warning("[%dms] get_audio_sample error: %s", self._elapsed_ms(), exc)
                    sample = None
                if sample is not None:
                    self.wakeword.observe_sample(self.input_sample_rate, sample)

                if self.sleeping:
                    has_notification = not self._notification_queue.empty()
                    if sample is not None and self.wakeword.process_sample(self.input_sample_rate, sample):
                        self._wake_from_sleep_mode()
                    elif has_notification:
                        self._wake_from_sleep_mode(send_wake_notice=False)
                    else:
                        time.sleep(0.01)
                        continue

                if sample is not None and self.realtime is not None and self.active_mode_uses_mic_recording:
                    self.realtime.feed_audio(self.input_sample_rate, sample)

                previous_responses_streamed = self.responses_streamed
                prev_playback = self.playback_started
                self.playback_started, self.audio_chunks_total, self.responses_streamed = process_audio_queue(
                    audio_queue=self.audio_queue,
                    media_io=self.media_io,
                    output_sample_rate=self.output_sample_rate,
                    realtime_output_rate=self.realtime_output_rate,
                    playback_started=self.playback_started,
                    audio_chunks_total=self.audio_chunks_total,
                    responses_streamed=self.responses_streamed,
                    elapsed_ms=self._elapsed_ms,
                    state_machine=self.state_machine,
                    motion_manager=self.motion_manager,
                )
                if self.playback_started and not prev_playback:
                    self._play_start_ts = time.monotonic()

                if (
                    self.sleep_after_response.is_set()
                    and self.responses_streamed > previous_responses_streamed
                ):
                    self.sleep_after_response.clear()
                    # Calculate exact drain time from queued samples + 200 ms GStreamer buffer allowance
                    if self._play_start_ts > 0.0 and self._queued_samples > 0:
                        play_duration = self._queued_samples / self.realtime_output_rate
                        self._sleep_drain_at = self._play_start_ts + play_duration + 0.2
                    else:
                        self._sleep_drain_at = time.monotonic() + 1.0

                if self._sleep_drain_at > 0.0 and time.monotonic() >= self._sleep_drain_at:
                    self._sleep_drain_at = 0.0
                    self._enter_sleep_mode()
                    continue

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

                if self._should_recover_realtime(now):
                    self._recover_realtime_session(now)
                    time.sleep(0.01)
                    continue

                if self.idle_sleep_enabled and (now - self.last_user_activity) >= self.idle_sleep_timeout_s:
                    self._enter_sleep_mode()

                self._drain_notification_queue()

                time.sleep(0.01)
        finally:
            self.input_stop.set()
            if self.playback_started:
                try:
                    self.media_io.stop_playing()
                except Exception:
                    pass
            if self.realtime is not None:
                self.realtime.stop()
            if self.recording_started:
                try:
                    self.media_io.stop_recording()
                except Exception:
                    pass
            if self.motion_manager is not None:
                self.motion_manager.stop()
            if self.camera_worker is not None:
                self.camera_worker.stop()
            if self._notification_server is not None:
                self._notification_server.stop()
                self._notification_server = None

    def _elapsed_ms(self) -> int:
        """Return elapsed orchestrator runtime in milliseconds."""
        return int((time.monotonic() - self.loop_start) * 1000)

    def _mark_user_activity(self) -> None:
        """Record the latest user activity timestamp for idle-sleep logic."""
        self.last_user_activity = time.monotonic()

    def _on_speech_start(self) -> None:
        """Handle speech-start callback from conversation session."""
        logging.debug("[%dms] Callback speech_start", self._elapsed_ms())
        self._mark_user_activity()
        self._queued_samples = 0
        self._play_start_ts = 0.0
        self.audio_queue.put(("force_stop", None))
        try:
            apply_event(self.state_machine, Event.WAKE_WORD, self.motion_manager)
            self.robot_actions.gesture_listening()
        except Exception as exc:
            logging.warning("[%dms] gesture.listening failed: %s", self._elapsed_ms(), exc)
        if self.camera_worker is not None and self.config.speaker_id_enabled:
            threading.Thread(target=self._identify_speaker_for_turn, daemon=True).start()

    def _identify_speaker_for_turn(self) -> None:
        """Background: DOA-steer + face-ID so result is ready before transcript arrives."""
        try:
            self.camera_worker.identify_current_speaker(wait_s=0.4)  # type: ignore[union-attr]
        except Exception as exc:
            logging.debug("Speaker turn ID failed: %s", exc)

    def _on_user_text(self, text: str, *, source: str) -> None:
        """Handle recognized or typed user text and trigger think gesture."""
        self._mark_user_activity()
        logging.info("[%dms] User %s: %s", self._elapsed_ms(), source, text)
        try:
            apply_event(self.state_machine, Event.STT_RECEIVED, self.motion_manager)
            self.robot_actions.gesture_think()
        except Exception as exc:
            logging.warning("[%dms] gesture.think failed: %s", self._elapsed_ms(), exc)
        if source == "said":
            if text:
                self._session_messages.append({"role": "user", "content": text})
            self._maybe_inject_speaker_change()

    def _maybe_inject_speaker_change(self) -> None:
        """Notify the model when the identified speaker changed since the last turn."""
        if self.camera_worker is None or not self.config.speaker_id_enabled:
            return
        from reachy.face_recognizer import FaceRecognizer
        detected = self.camera_worker.get_current_speaker()
        if detected == self._current_speaker:
            return
        self._current_speaker = detected
        if detected and detected != FaceRecognizer.UNKNOWN:
            notice = (
                f"SYSTEM NOTICE (não é mensagem do usuário): o falante atual mudou. "
                f"A pessoa que acabou de falar foi identificada como: {detected}."
            )
        else:
            notice = (
                "SYSTEM NOTICE (não é mensagem do usuário): o falante atual mudou. "
                "A pessoa que acabou de falar não foi reconhecida (visitante)."
            )
        logging.info("[%dms] Speaker changed → %s", self._elapsed_ms(), detected or "visitante")
        if self.realtime is not None:
            self.realtime.send_text(notice)

    def _on_assistant_text(self, text: str) -> None:
        """Log assistant text responses emitted by realtime session."""
        logging.info("[%dms] Assistant text: %s", self._elapsed_ms(), text)
        _conv_log.info("Dobby ❯ %s", text)
        if text:
            self._session_messages.append({"role": "assistant", "content": text})

    def _on_assistant_audio_chunk(self, chunk) -> None:
        """Queue one streamed assistant audio chunk for playback."""
        try:
            self._queued_samples += int(getattr(chunk, "size", len(chunk)))
        except Exception:
            pass
        self.audio_queue.put(("chunk", chunk))

    def _on_assistant_audio_done(self) -> None:
        """Queue assistant audio completion marker."""
        logging.debug("[%dms] Assistant audio done queued", self._elapsed_ms())
        self.audio_queue.put(("done", None))

    def _on_error(self, message: str) -> None:
        """Handle realtime errors and schedule transparent recovery messaging."""
        logging.warning("[%dms] Realtime error: %s", self._elapsed_ms(), message)
        self.realtime_last_error_message = message.strip()
        self.realtime_recovery_requested = True
        self.pending_recovery_notice = (
            "SYSTEM NOTICE (not user message): tivemos uma falha técnica temporária na conexão "
            "durante a última resposta. Avise o usuário em português de forma breve e peça para repetir."
        )

    def _should_recover_realtime(self, now: float) -> bool:
        """Determine whether realtime session recovery should be attempted."""
        if self.sleeping or self.realtime is None:
            return False
        if now < self.realtime_next_recovery_at:
            return False
        if self.realtime_recovery_requested:
            return True
        try:
            return not self.realtime.wait_until_ready(timeout_s=0.0)
        except Exception:
            return True

    def _recover_realtime_session(self, now: float) -> None:
        """Recreate realtime session after connection or runtime failure."""
        self.realtime_next_recovery_at = now + 2.0
        logging.info("[%dms] Attempting realtime session recovery", self._elapsed_ms())

        if self.realtime is not None:
            try:
                self.realtime.stop()
            except Exception as exc:
                logging.warning("[%dms] Realtime stop during recovery failed: %s", self._elapsed_ms(), exc)
            self.realtime = None

        try:
            self._start_active_session()
        except Exception as exc:
            self.realtime_recovery_requested = True
            self.realtime_next_recovery_at = time.monotonic() + 5.0
            logging.warning("[%dms] Realtime recovery failed: %s", self._elapsed_ms(), exc)
            return

        self.realtime_recovery_requested = False
        if self.pending_recovery_notice and self.realtime is not None:
            sent = self.realtime.send_text(self.pending_recovery_notice)
            if sent:
                self.pending_recovery_notice = None
            else:
                logging.warning("[%dms] Failed to send post-recovery notice", self._elapsed_ms())

    def _build_ha_catalog_section(self) -> str:
        """Fetch HA entities and return a formatted catalog section, or empty string."""
        if self._ha_client is None:
            return ""
        try:
            entities = self._ha_client.get_catalog()
            if not entities:
                return ""
            lines = "\n".join(
                f"- {e['entity_id']} | {e['friendly_name']} | {e['state']}"
                for e in entities
            )
            logging.info("HA catalog: %d entities injected into session", len(entities))
            return (
                "\n\n## DISPOSITIVOS HOME ASSISTANT DISPONÍVEIS\n"
                f"{lines}\n"
                "(Use control_home_device para controlar. "
                "Use discover_home_devices para atributos detalhados ou estado atualizado.)"
            )
        except Exception as exc:
            logging.warning("HA catalog fetch failed: %s", exc)
            return ""

    def _build_runtime_identity_prompt(self, identity_prompt: str) -> str:
        """Append runtime tool guardrails to the base identity prompt."""
        tool_guardrails = [item.strip() for item in self.tool_runtime.runtime_guardrails() if item.strip()]
        if not tool_guardrails:
            return identity_prompt

        lines = "\n".join(f"- {item}" for item in tool_guardrails)
        runtime_guardrails = (
            "\n\n## RUNTIME TOOL GUARDRAILS (HIGHEST PRIORITY)\n"
            f"{lines}"
        )
        return f"{identity_prompt.rstrip()}{runtime_guardrails}\n"

    def _build_speaker_section(self) -> str:
        """Return a speaker context section to append to the identity prompt."""
        if not self._current_speaker:
            return ""
        from reachy.face_recognizer import FaceRecognizer
        if self._current_speaker == FaceRecognizer.UNKNOWN:
            return (
                "\n\n## FALANTE ATUAL\n"
                "A pessoa que ativou o wakeword não foi reconhecida (visitante)."
            )
        section = (
            "\n\n## FALANTE ATUAL\n"
            f"A pessoa que ativou o wakeword foi identificada como: {self._current_speaker}."
        )
        if self._speaker_memory and self._speaker_memory.available:
            memories = self._speaker_memory.load(self._current_speaker)
            if memories:
                section += (
                    f"\n\n## O QUE VOCÊ JÁ SABE SOBRE {self._current_speaker.upper()}\n"
                    f"{memories}"
                )
        return section

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute one tool call and apply delegation/sleep side effects."""
        is_openclaw_delegate = name == "delegate_task"
        if is_openclaw_delegate:
            logging.info("[%dms] Delegating task to Open Claw", self._elapsed_ms())
            try:
                apply_event(self.state_machine, Event.DELEGATION_STARTED, self.motion_manager)
                self.robot_actions.gesture_delegating()
            except Exception as exc:
                logging.warning("[%dms] gesture.delegating failed: %s", self._elapsed_ms(), exc)

        try:
            result = self.tool_runtime.execute(name, arguments)
            if name == "go_to_sleep":
                output = getattr(result, "output", None)
                if isinstance(output, dict) and output.get("ok"):
                    self._queue_sleep_request()
            if name == "enroll_speaker":
                output = getattr(result, "output", None)
                enrolled_name = str(arguments.get("name", "")).strip()
                if isinstance(output, dict) and output.get("ok") and enrolled_name:
                    self._current_speaker = enrolled_name
                    if self.camera_worker is not None:
                        with self.camera_worker._speaker_lock:
                            self.camera_worker._current_speaker = enrolled_name
            return result
        finally:
            if is_openclaw_delegate:
                apply_event(self.state_machine, Event.DELEGATION_DONE, self.motion_manager)

    def _start_active_session(self) -> None:
        """Create and start the active realtime session for the current mode."""
        ha_section = self._build_ha_catalog_section()
        speaker_section = self._build_speaker_section()
        identity = self.identity_prompt_runtime + ha_section + speaker_section

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
            identity_prompt=identity,
            tool_specs=self.tool_specs,
            on_tool_call=self._execute_tool,
            callbacks=callbacks,
        )
        self.realtime.start()
        if not self.realtime.wait_until_ready(timeout_s=35.0):
            self.realtime.stop()
            self.realtime = None
            raise RuntimeError("Failed to start OpenAI Realtime session")

        logging.info("[%dms] Realtime session connected", self._elapsed_ms())

        if self.active_mode_uses_mic_recording and not self.recording_started:
            self.media_io.start_recording()
            self.recording_started = True
            logging.info("[%dms] Reachy microphone recording started", self._elapsed_ms())

        self.playback_started = False
        self.sleeping = False
        self.wakeword.reset()
        self.wakeword.start_auto_calibration()
        self._mark_user_activity()

    def _enter_sleep_mode(self) -> None:
        """Stop active session and place runtime into wakeword-only sleep mode."""
        if self.sleeping:
            return

        logging.info("[%dms] Entering sleep mode after %.1fs idle", self._elapsed_ms(), self.idle_sleep_timeout_s)

        if self.playback_started:
            try:
                self.media_io.stop_playing()
            except Exception:
                pass
            self.playback_started = False

        if self.realtime is not None:
            self.realtime.stop()
            self.realtime = None
        self.realtime_recovery_requested = False
        self.realtime_next_recovery_at = 0.0

        if self.active_mode_uses_mic_recording and self.recording_started:
            try:
                self.media_io.stop_recording()
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

        try:
            self.robot_actions.disable_motors()
        except Exception as exc:
            logging.warning("[%dms] Failed to disable motors: %s", self._elapsed_ms(), exc)

        if not self.recording_started:
            try:
                self.media_io.start_recording()
                self.recording_started = True
            except Exception as exc:
                logging.warning("[%dms] Failed to start low-power microphone recording: %s", self._elapsed_ms(), exc)

        apply_event(self.state_machine, Event.RESET, self.motion_manager)
        self.sleeping = True
        self.wakeword.reset()

        if self._speaker_memory and self._current_speaker:
            from reachy.face_recognizer import FaceRecognizer
            if self._current_speaker != FaceRecognizer.UNKNOWN:
                self._speaker_memory.save_async(self._current_speaker, self._session_messages)
        self._session_messages = []

    def _queue_sleep_request(self) -> None:
        """Request entering sleep mode after the current response is completed."""
        if self.sleeping:
            logging.info("[%dms] Sleep request ignored; already sleeping", self._elapsed_ms())
            return
        if self.sleep_after_response.is_set():
            return
        logging.info("[%dms] Sleep requested by tool", self._elapsed_ms())
        self.sleep_after_response.set()

    def _wake_from_sleep_mode(self, *, send_wake_notice: bool = True) -> None:
        """Wake runtime from sleep mode and restore active realtime session."""
        self._sleep_drain_at = 0.0
        logging.info("[%dms] Offline wakeword detected, waking up", self._elapsed_ms())

        if self.recording_started:
            try:
                self.media_io.stop_recording()
            except Exception:
                pass
            self.recording_started = False

        try:
            self.robot_actions.enable_motors()
        except Exception as exc:
            logging.warning("[%dms] Failed to enable motors: %s", self._elapsed_ms(), exc)

        try:
            self.robot_actions.wake_up()
        except Exception as exc:
            logging.warning("[%dms] Failed to send wake_up: %s", self._elapsed_ms(), exc)

        if self.camera_worker is not None:
            self.camera_worker.start()
            speaker = self.camera_worker.identify_current_speaker(wait_s=0.7)
            if speaker:
                logging.info("[%dms] Speaker identified: %s", self._elapsed_ms(), speaker)
            self._current_speaker = speaker
        if self.motion_manager is not None:
            self.motion_manager.start()

        self._start_active_session()
        self._mark_user_activity()
        self.sleeping = False

        if send_wake_notice and self.realtime is not None:
            self.realtime.send_text(
                "SYSTEM NOTICE (não é mensagem do usuário): você acabou de ser acordado pelo wakeword. "
                "Diga algo breve em português para avisar que está acordado e pronto para ajudar. "
                "Seja fiel ao seu personagem."
            )

    def _drain_notification_queue(self) -> None:
        """Inject pending async notifications into the active realtime session."""
        if self._notification_queue.empty():
            return
        if self.realtime is None or not self.realtime.wait_until_ready(timeout_s=0.0):
            return
        if not self.audio_queue.empty():
            return
        while True:
            try:
                summary = self._notification_queue.get_nowait()
            except Empty:
                break
            notice = (
                "SYSTEM NOTICE (não é mensagem do usuário): lembrete agendado disparado. "
                f"Diga ao usuário: {summary}"
            )
            logging.info("[%dms] Delivering notification: %.80s", self._elapsed_ms(), summary)
            self.realtime.send_text(notice)
            self._mark_user_activity()

    def _drain_text_queue(self) -> bool:
        """Process queued chat text inputs and return True when exit requested."""
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
        """Read stdin lines in chat mode and enqueue them for processing."""
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
        """Log startup context for chat or realtime mode."""
        if self.interactive_text:
            logging.info(
                "Chat mode active model=%s transcribe=%s out_sr=%s tools=%s idle_sleep=%s",
                self.config.realtime_model,
                self.config.realtime_transcription_model,
                self.output_sample_rate,
                self.tool_runtime.names(),
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
            self.tool_runtime.names(),
            self.idle_sleep_enabled,
        )
