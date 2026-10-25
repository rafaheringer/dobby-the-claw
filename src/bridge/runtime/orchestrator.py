"""Shared runtime orchestrator for the realtime voice runtime."""

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import Any, Optional

import numpy as np

from bridge.config import BridgeConfig
from reachy.camera_worker import CameraWorker
from reachy.client import ReachyClient
from reachy.motion import MotionManager
from bridge.state_machine import Event, State, StateMachine

from bridge.runtime.adapters.notification_server import NotificationServer
from bridge.runtime.adapters.reachy_actions import ReachyRobotActions
from bridge.runtime.adapters.reachy_media import build_reachy_media_io
from bridge.runtime.adapters.realtime_session import build_conversation_factory
from bridge.runtime.adapters.tool_runtime import build_tool_runtime
from bridge.runtime.audio_support import (
    build_wakeword_detector,
    maybe_log_health,
    process_audio_queue,
    require_sdk_instance,
    resolve_gemini_api_key,
    resolve_llm_api_key,
)
from homeassistant.home_assistant_client import HomeAssistantWsClient
from bridge.runtime.bootstrap import load_prompt
from bridge.runtime.common import apply_event, resample_audio_chunk
from bridge.runtime.presence_notices import PresenceNotifier
from bridge.runtime.touch import TouchDetector, TouchKind, TouchSpeechGate
from bridge.runtime.speaker_id import MicRingBuffer, _to_mono, fuse_speaker
from bridge.runtime.ports import (
    ConversationCallbacks,
    ConversationSessionFactoryPort,
    ConversationSessionPort,
    MediaIOPort,
    RobotActionsPort,
    ToolRuntimePort,
)


_conv_log = logging.getLogger("bridge.conversation")


# Touch event -> (kind, phase) for RobotActionsPort.react_touch.
_TOUCH_REACTIONS = {
    TouchKind.PET_START: ("pet", "start"),
    TouchKind.PET_END: ("pet", "end"),
    TouchKind.TAP: ("tap", "start"),
}


class RuntimeOrchestrator:
    """Coordinate shared runtime behavior across chat and realtime modes."""

    def __init__(
        self,
        *,
        state_machine: StateMachine,
        reachy: ReachyClient,
        motion_manager: Optional[MotionManager],
        camera_worker: Optional[CameraWorker],
        config: BridgeConfig,
        reachy_sdk_instance: Any,
        identity_prompt: str,
        idle_sleep_timeout_s: float,
        wake_on_startup: bool = True,
        conversation_factory: ConversationSessionFactoryPort | None = None,
        robot_actions: RobotActionsPort | None = None,
        tool_runtime: ToolRuntimePort | None = None,
        media_io: MediaIOPort | None = None,
        ha_client: HomeAssistantWsClient | None = None,
        accept_text: bool = False,
    ) -> None:
        """Initialize shared runtime dependencies and orchestration state."""
        self.state_machine = state_machine
        self.motion_manager = motion_manager
        self.camera_worker = camera_worker
        self.config = config
        self.identity_prompt = identity_prompt
        self.idle_sleep_timeout_s = idle_sleep_timeout_s
        self.transitioning = ""  # "sleep" / "wake" while that change is under way
        self.accept_text = accept_text  # debug UI feeds typed turns through text_queue
        self.wake_on_startup = wake_on_startup
        self.conversation_factory = conversation_factory or build_conversation_factory(config)
        self.robot_actions = robot_actions or ReachyRobotActions(reachy, motion_manager)

        if media_io is None:
            require_sdk_instance(reachy_sdk_instance)
            self.media_io = build_reachy_media_io(reachy_sdk_instance)
        else:
            self.media_io = media_io

        self.api_key = resolve_llm_api_key(config)

        self._current_speaker: str | None = None
        self._session_messages: list[dict] = []

        self._speaker_memory = None
        memory_provider = "gemini" if config.conversation_provider == "gemini" else "openai"
        memory_key = resolve_gemini_api_key() if memory_provider == "gemini" else self.api_key
        if config.speaker_memory_enabled and not memory_key:
            logging.warning("Speaker memory has no API key for %s; disabled for this run", memory_provider)
        elif config.speaker_memory_enabled:
            import os as _os
            from bridge.speaker_memory import SpeakerMemory
            self._speaker_memory = SpeakerMemory(
                storage_dir=_os.path.expanduser(config.speaker_memory_dir),
                extraction_model=config.speaker_memory_model,
                api_key=memory_key,
                provider=memory_provider,
            )

        self._voice = None
        self._mic_tap: Optional[list] = None  # while a wakeword clip is recorded, every mic chunk is also kept here
        self._mic_buffer = MicRingBuffer()  # voice ID reads utterances from it; wakeword calibration records clips
        self._speaker_id_lock = threading.Lock()
        self._last_turn_voice_emb = None
        self._utterance_done = threading.Event()  # set when the user transcript for the turn arrives
        self._playback_cut_ts = 0.0  # when playback was last stopped by the user talking over Dobby
        if config.speaker_id_enabled and config.voice_id_enabled:
            import os as _os
            from reachy.voice_recognizer import VoiceRecognizer
            voice = VoiceRecognizer(
                _os.path.expanduser(config.speaker_id_profiles_dir),
                model_path=config.voice_id_model_path,
            )
            if voice.available:
                self._voice = voice

        self._notification_queue: "Queue[str]" = Queue()

        self._presence: PresenceNotifier | None = None
        self._last_presence_poll = 0.0
        if config.presence_enabled and camera_worker is not None:
            self._presence = PresenceNotifier(greet_cooldown_s=config.presence_greet_cooldown_s)

        self._touch: TouchDetector | None = None
        self._touch_gate = TouchSpeechGate(cooldown_s=config.touch_speak_cooldown_s)
        self._touch_deaf_until = 0.0
        if config.touch_enabled:
            self._touch = TouchDetector(pet_dbfs=config.touch_pet_dbfs, tap_peak=config.touch_tap_peak)

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

        self.playback_started = False
        self.loop_start = time.monotonic()
        self.last_health_log = self.loop_start
        self.audio_chunks_total = 0
        self.responses_streamed = 0
        self.recording_started = False
        self.sleeping = False
        self.sleep_after_response = threading.Event()
        self._sleep_requested_ts: float = 0.0
        self._last_audio_chunk_ts: float = 0.0
        self._last_tool_call_ts: float = 0.0
        self._playback_end_ts: float = 0.0  # estimated time when queued assistant audio finishes
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
                speaker, _ = self.camera_worker.identify_current_speaker(wait_s=0.5)
                self._current_speaker = speaker
                if speaker:
                    logging.info("[%dms] Startup speaker ID: %s", self._elapsed_ms(), speaker)
                else:
                    logging.info("[%dms] Startup speaker ID: no face detected", self._elapsed_ms())
            except Exception as exc:
                logging.warning("[%dms] Startup speaker ID failed: %s", self._elapsed_ms(), exc)

        self._start_active_session()

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
                _loop_t0 = time.monotonic()
                try:
                    sample = self.media_io.get_audio_sample()
                except Exception as exc:
                    logging.warning("[%dms] get_audio_sample error: %s", self._elapsed_ms(), exc)
                    sample = None
                _fetch_s = time.monotonic() - _loop_t0
                if sample is not None:
                    self.wakeword.observe_sample(self.input_sample_rate, sample)
                    if self._touch is not None:
                        self._observe_touch(sample, time.monotonic())
                    if self._mic_buffer is not None:
                        self._mic_buffer.append(self.input_sample_rate, sample, time.monotonic())
                    tap = self._mic_tap
                    if tap is not None:
                        tap.append(sample)

                _observe_s = time.monotonic() - _loop_t0 - _fetch_s
                if _fetch_s + _observe_s > 0.3:
                    logging.warning("Slow audio loop: fetch %.2fs observe %.2fs (%d samples)", _fetch_s, _observe_s, 0 if sample is None else len(sample))
                if self.sleeping:
                    has_notification = not self._notification_queue.empty()
                    if self.accept_text and self._take_wake_request():
                        self._wake_from_sleep_mode()
                    elif sample is not None and self.wakeword.process_sample(self.input_sample_rate, sample):
                        self._wake_from_sleep_mode()
                    elif has_notification:
                        self._wake_from_sleep_mode(send_wake_notice=False)
                    else:
                        time.sleep(0.01)
                        continue

                if sample is not None and self.realtime is not None:
                    self.realtime.feed_audio(self.input_sample_rate, sample)

                was_playing = self.playback_started
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
                if was_playing and not self.playback_started:
                    self._playback_cut_ts = time.monotonic()
                if self.sleep_after_response.is_set():
                    # The model may speak its farewell after the tool call, so wait for a quiet gap
                    # after the last audio chunk and for the queued audio to finish playing.
                    now = time.monotonic()
                    quiet_since = max(self._sleep_requested_ts, self._last_audio_chunk_ts)
                    if now - quiet_since >= 2.5 and now >= self._playback_end_ts + 0.8:
                        self.sleep_after_response.clear()
                        self._enter_sleep_mode()
                        continue

                if self.accept_text:
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
                    )

                if self._should_recover_realtime(now):
                    self._recover_realtime_session(now)
                    time.sleep(0.01)
                    continue

                if self.idle_sleep_enabled and (now - self.last_user_activity) >= self.idle_sleep_timeout_s:
                    self._enter_sleep_mode()

                self._reset_if_unanswered(now)
                self._drain_notification_queue()
                self._maybe_send_presence_notice(now)

                time.sleep(0.01)
        finally:
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

    def _reset_if_unanswered(self, now: float) -> None:
        """Leave the listening pose when the model chose not to answer (speech not meant for it)."""
        if self.state_machine.state not in (State.LISTENING, State.THINKING):
            return
        last_event = max(self.last_user_activity, self._last_audio_chunk_ts, self._last_tool_call_ts)
        if now - last_event < 12.0:
            return
        logging.info("[%dms] No reply to the last utterance; back to idle", self._elapsed_ms())
        apply_event(self.state_machine, Event.RESET, self.motion_manager)

    def _observe_touch(self, sample: Any, now: float) -> None:
        """Feed the touch detector and react to petting/taps (sound only while asleep)."""
        deaf = (
            self.playback_started
            or now < self._playback_end_ts + 0.8
            or now < self._touch_deaf_until
        )
        for event in self._touch.observe(self.input_sample_rate, sample, now, deaf=deaf):
            kind, phase = _TOUCH_REACTIONS[event.kind]
            logging.info(
                "[%dms] Touch: %s (%.1fs)", self._elapsed_ms(), event.kind.value, event.duration_s
            )
            sound_s = self.robot_actions.react_touch(kind, phase, asleep=self.sleeping)
            if sound_s > 0:
                self._touch_deaf_until = now + sound_s + 0.3
            if self.sleeping:
                continue  # not user activity: touches must not keep Dobby awake
            self.last_user_activity = now
            text = self._touch_gate.notice(event, now)
            if text is not None and self.realtime is not None:
                logging.info("[%dms] Touch notice: %s", self._elapsed_ms(), text)
                self.realtime.send_text(text, respond=True)

    def _maybe_send_presence_notice(self, now: float) -> None:
        """Tell the model who arrived, was recognized or left, only while the conversation is quiet."""
        if self._presence is None or self.camera_worker is None or self.realtime is None:
            return
        if now - self._last_presence_poll < 0.5:
            return
        self._last_presence_poll = now
        wall = time.time()  # camera tracks are stamped with wall-clock time
        self._presence.observe(
            [(name, since) for _, name, since in self.camera_worker.get_presence()],
            wall,
            pets=self.camera_worker.get_pets(),
        )
        notice = self._presence.pending(wall)
        if notice is None:
            return
        if notice.text:
            last_event = max(self.last_user_activity, self._last_audio_chunk_ts, self._last_tool_call_ts)
            busy = (
                self.state_machine.state in (State.LISTENING, State.THINKING, State.DELEGATING, State.CONFIRMING)
                or now - last_event < 3.0
                or now < self._playback_end_ts + 1.0
                or not self.audio_queue.empty()
                or self.sleep_after_response.is_set()
                or not self.realtime.wait_until_ready(timeout_s=0.0)
            )
            if busy:
                return
            logging.info("[%dms] Presence notice (respond=%s): %s", self._elapsed_ms(), notice.respond, notice.text)
            # Not user activity: presence alone must not keep Dobby awake.
            if not self.realtime.send_text(notice.text, respond=notice.respond):
                return
        self._presence.commit(notice, wall)

    def _mark_user_activity(self) -> None:
        """Record the latest user activity timestamp for idle-sleep logic."""
        self.last_user_activity = time.monotonic()

    def _on_speech_start(self) -> None:
        """Handle speech-start callback from conversation session."""
        logging.debug("[%dms] Callback speech_start", self._elapsed_ms())
        self._mark_user_activity()
        speech_ts = time.monotonic()
        # Mic audio before this point may still contain Dobby's own voice.
        playback_end = self._playback_end_ts
        self._playback_end_ts = 0.0
        self._utterance_done.clear()
        self.audio_queue.put(("force_stop", None))
        try:
            apply_event(self.state_machine, Event.WAKE_WORD, self.motion_manager)
            self.robot_actions.gesture_listening()
        except Exception as exc:
            logging.warning("[%dms] gesture.listening failed: %s", self._elapsed_ms(), exc)
        if self.config.speaker_id_enabled and (self.camera_worker is not None or self._voice is not None):
            threading.Thread(
                target=self._identify_speaker_for_turn, args=(speech_ts, playback_end), daemon=True
            ).start()

    def _identify_speaker_for_turn(self, speech_ts: float, playback_end: float) -> None:
        """Background: identify who is talking, by face right away and by voice once they finish."""
        if not self._speaker_id_lock.acquire(blocking=False):
            return  # interruption and transcript can both signal the same utterance
        try:
            from reachy.face_recognizer import FaceRecognizer
            face_name, face_score, face_certain = None, 0.0, True
            if self.camera_worker is not None:
                face_name, face_score = self.camera_worker.identify_current_speaker(wait_s=0.4, update=False)
                face_certain = self.camera_worker.speaker_face_certain
                logging.info(
                    "[%dms] Turn face: %s %.2f after %.1fs",
                    self._elapsed_ms(), face_name, face_score, time.monotonic() - speech_ts,
                )
                # Early pass: a known face is enough to warn the model before it answers.
                if face_name and face_name != FaceRecognizer.UNKNOWN:
                    self._set_turn_speaker(face_name)

            emb = None
            if self._voice is not None and self._mic_buffer is not None:
                # The echo canceller mutes the mic while Dobby talks, and speech start is only signalled
                # with the first transcript, so embed the whole utterance once it is over.
                self._utterance_done.wait(timeout=8.0)
                end = time.monotonic()
                start = max(speech_ts - 2.0, end - 7.5)
                if playback_end > 0.0:
                    start = max(start, min(playback_end + 0.3, speech_ts))
                if self._playback_cut_ts >= speech_ts - 0.5:
                    start = max(start, self._playback_cut_ts + 0.2)
                clip = self._mic_buffer.window(start, end)
                logging.info(
                    "[%dms] Turn voice clip: %.1fs, peak %.3f",
                    self._elapsed_ms(), clip.size / 16000.0, float(np.abs(clip).max()) if clip.size else 0.0,
                )
                emb = self._voice.embed(clip)
            self._last_turn_voice_emb = emb

            voice_scores, voice_counts = None, {}
            if emb is not None:
                voice_scores = self._voice.scores(emb)  # type: ignore[union-attr]
                voice_counts = {name: self._voice.sample_count(name) for name in voice_scores}  # type: ignore[union-attr]
            # With several people in view and no sign of who is talking, a face is no evidence of the speaker:
            # it is just the person the camera happened to pick (the old behavior answered everyone as the first).
            decision = fuse_speaker(
                face_name if face_certain else None, face_score if face_certain else 0.0,
                voice_scores, voice_counts, FaceRecognizer.UNKNOWN,
            )
            if decision.name is None and not face_certain and self.camera_worker is not None:
                self._note_unknown_speaker()
            logging.info(
                "[%dms] Turn speaker: %s via %s after %.1fs (face=%s %.2f, voice=%s)",
                self._elapsed_ms(), decision.name or "unchanged", decision.source, time.monotonic() - speech_ts,
                face_name, face_score,
                {n: round(v, 2) for n, v in voice_scores.items()} if voice_scores is not None else "n/a",
            )
            # With several faces and no clear voice direction the face may belong to someone silent.
            if decision.learn_voice_as and emb is not None and face_certain:
                self._voice.add_sample(decision.learn_voice_as, emb)  # type: ignore[union-attr]
            if decision.name is not None:
                self._set_turn_speaker(decision.name)
        except Exception as exc:
            logging.warning("Speaker turn ID failed: %s", exc)
        finally:
            self._speaker_id_lock.release()

    def _note_unknown_speaker(self) -> None:
        """Several people are in view and nothing says who is talking: tell the model not to assume."""
        names = [name or "pessoa não reconhecida" for _, name, _ in self.camera_worker.get_presence()]  # type: ignore[union-attr]
        if len(names) < 2:
            return
        key = ",".join(sorted(names))
        if key == getattr(self, "_ambiguous_notice_key", None):
            return
        self._ambiguous_notice_key = key
        self._current_speaker = None
        logging.info("[%dms] Speaker unclear among: %s", self._elapsed_ms(), ", ".join(names))
        if self.realtime is not None:
            self.realtime.send_text(
                "SYSTEM NOTICE (não é mensagem do usuário): há várias pessoas à vista "
                f"({', '.join(names)}) e não deu para saber quem está falando agora. Não presuma que é a mesma "
                "pessoa de antes; se o pedido depender de quem fala, pergunte quem é.",
                respond=False,
            )

    def _set_turn_speaker(self, name: str) -> None:
        """Make `name` the current speaker and tell the model right away when it changed."""
        from reachy.face_recognizer import FaceRecognizer
        if self.camera_worker is not None:
            self.camera_worker.set_current_speaker(name, hold_s=20.0)
        if self._presence is not None:
            self._presence.mark_greeted(None if name == FaceRecognizer.UNKNOWN else name, time.time())
        if name == self._current_speaker:
            return
        self._current_speaker = name
        if name != FaceRecognizer.UNKNOWN:
            notice = f"SYSTEM NOTICE (não é mensagem do usuário): quem está falando agora é {name}."
        else:
            notice = (
                "SYSTEM NOTICE (não é mensagem do usuário): quem está falando agora "
                "é uma pessoa não reconhecida (visitante)."
            )
        logging.info("[%dms] Speaker changed → %s", self._elapsed_ms(), name)
        if self.realtime is not None:
            self.realtime.send_text(notice, respond=False)

    def _on_user_text(self, text: str, *, source: str) -> None:
        """Handle recognized or typed user text and trigger think gesture."""
        self._mark_user_activity()
        logging.info("[%dms] User %s: %s", self._elapsed_ms(), source, text)
        try:
            apply_event(self.state_machine, Event.STT_RECEIVED, self.motion_manager)
            self.robot_actions.gesture_think()
        except Exception as exc:
            logging.warning("[%dms] gesture.think failed: %s", self._elapsed_ms(), exc)
        if source == "said" and text:
            self._session_messages.append({"role": "user", "content": text})
            self._utterance_done.set()

    def _on_assistant_text(self, text: str) -> None:
        """Log assistant text responses emitted by realtime session."""
        logging.info("[%dms] Assistant text: %s", self._elapsed_ms(), text)
        _conv_log.info("Dobby ❯ %s", text)
        if text:
            self._session_messages.append({"role": "assistant", "content": text})

    def _on_assistant_audio_chunk(self, chunk) -> None:
        """Queue one streamed assistant audio chunk for playback."""
        try:
            samples = int(getattr(chunk, "size", len(chunk)))
            now = time.monotonic()
            self._last_audio_chunk_ts = now
            self._playback_end_ts = max(now, self._playback_end_ts) + samples / self.realtime_output_rate
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
        self._last_tool_call_ts = time.monotonic()
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
                # Someone introduced by another person may not have spoken yet: their voice is learned
                # later, when their face is recognized while they talk.
                if isinstance(output, dict) and output.get("ok") and enrolled_name and output.get("voice_from_turn"):
                    self._current_speaker = enrolled_name
                    if self.camera_worker is not None:
                        self.camera_worker.set_current_speaker(enrolled_name, hold_s=20.0)
                    if self._voice is not None and self._last_turn_voice_emb is not None:
                        self._voice.add_sample(enrolled_name, self._last_turn_voice_emb)
            return result
        finally:
            if is_openclaw_delegate:
                apply_event(self.state_machine, Event.DELEGATION_DONE, self.motion_manager)

    def _start_active_session(self) -> None:
        """Create and start the active realtime session for the current mode."""
        ha_section = self._build_ha_catalog_section()
        speaker_section = self._build_speaker_section()
        identity = self.identity_prompt_runtime + ha_section + speaker_section
        if self._presence is not None:
            identity += "\n\n" + load_prompt("presence.md")
        if self._touch is not None:
            identity += "\n\n" + load_prompt("touch.md")

        callbacks = ConversationCallbacks(
            on_speech_start=self._on_speech_start,
            on_user_transcript=lambda text: self._on_user_text(text, source="said"),
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
            raise RuntimeError("Failed to start realtime conversation session")

        logging.info("[%dms] Realtime session connected", self._elapsed_ms())

        if not self.recording_started:
            self.media_io.start_recording()
            self.recording_started = True
            logging.info("[%dms] Reachy microphone recording started", self._elapsed_ms())

        self.playback_started = False
        self.sleeping = False
        self.wakeword.reset()
        self._mark_user_activity()
        if self._presence is not None:
            self._presence.reset(time.time())  # new session: whoever is in view is context, not a greeting

    def _enter_sleep_mode(self) -> None:
        """Stop active session and place runtime into wakeword-only sleep mode."""
        if self.sleeping:
            return
        self.transitioning = "sleep"  # shown by the web UI: going to sleep takes several seconds
        try:
            self._enter_sleep_mode_now()
        finally:
            self.transitioning = ""

    def _enter_sleep_mode_now(self) -> None:

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

        if self.recording_started:
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
        self._sleep_requested_ts = time.monotonic()
        self.sleep_after_response.set()

    def _wake_from_sleep_mode(self, *, send_wake_notice: bool = True) -> None:
        """Wake runtime from sleep mode and restore active realtime session."""
        self.transitioning = "wake"
        try:
            self._wake_from_sleep_mode_now(send_wake_notice=send_wake_notice)
        finally:
            self.transitioning = ""

    def _wake_from_sleep_mode_now(self, *, send_wake_notice: bool) -> None:
        self.sleep_after_response.clear()
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
            speaker, _ = self.camera_worker.identify_current_speaker(wait_s=0.7)
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

    def record_mic_clip(self, seconds: float) -> np.ndarray:
        """Block for `seconds` and return everything the microphone delivered meanwhile (16 kHz mono)."""
        chunks: list = []
        self._mic_tap = chunks
        try:
            time.sleep(seconds)
        finally:
            self._mic_tap = None
        if not chunks:
            logging.warning("Wakeword clip: the microphone delivered no audio in %.1fs", seconds)
            return np.zeros(0, dtype=np.float32)
        rate = self.input_sample_rate
        audio = np.concatenate([resample_audio_chunk(_to_mono(chunk), rate, 16000) for chunk in chunks]).astype(np.float32)
        logging.info("Wakeword clip: %d chunks, %.1fs of audio in %.1fs, peak %.4f", len(chunks), audio.size / 16000.0, seconds, float(np.abs(audio).max()))
        return audio

    def _take_wake_request(self) -> bool:
        """While asleep, consume typed input: True when a /wake arrived (other text is dropped, /quit is kept)."""
        wake, keep = False, []
        while True:
            try:
                text = self.text_queue.get_nowait().strip()
            except Empty:
                break
            if text.lower() == "/wake":
                wake = True
            elif text.lower() in {"/quit", "/exit"}:
                keep.append(text)
            elif text:
                logging.info("[%dms] Ignoring text input while sleeping; wake Dobby first", self._elapsed_ms())
        for text in keep:
            self.text_queue.put(text)
        return wake

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
                logging.info("[%dms] Exit requested", self._elapsed_ms())
                return True
            if text.lower() == "/sleep":
                self._enter_sleep_mode()
                continue
            if text.lower() == "/wake":
                if self.sleeping:
                    self._wake_from_sleep_mode()
                continue

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


    def _log_mode_start(self) -> None:
        """Log startup context."""
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
