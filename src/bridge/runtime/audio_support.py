"""Shared audio queue processing and runtime health/shutdown helpers."""

from __future__ import annotations

import logging
import os
import time
from queue import Empty, Queue
from typing import Any, Callable, Dict, List, Optional

from bridge.config import BridgeConfig, build_home_assistant_websocket_url
from reachy.camera_worker import CameraWorker
from reachy.motion import MotionManager
from reachy.realtime_client import OpenAIRealtimeSession
from bridge.runtime.adapters.openclaw_gateway import OpenClawGatewayClient, OpenClawGatewayConfig
from bridge.state_machine import Event, StateMachine
from bridge.tools import (
    CameraSnapshotTool,
    DanceTool,
    EmotionTool,
    EnrollSpeakerTool,
    GoToSleepTool,
    HomeAssistantDiscoverTool,
    HomeAssistantExecuteActionTool,
    HomeAssistantWsClient,
    HomeAssistantWsConfig,
    OpenClawDelegateTool,
    ToolRegistry,
)
from bridge.tools.dance import DANCE_AVAILABLE

from bridge.runtime.common import apply_event, resample_audio_chunk
from bridge.runtime.ports import ConversationSessionPort, MediaIOPort
from bridge.runtime.wakeword import OfflineWakewordDetector


def require_sdk_instance(mode_name: str, reachy_sdk_instance: Any) -> None:
    """Validate sdk instance for modes that need Reachy media APIs."""
    if reachy_sdk_instance is None:
        raise RuntimeError(f"{mode_name} mode requires REACHY_BRIDGE_URL=sdk")


def resolve_llm_api_key(config: BridgeConfig) -> str:
    """Resolve configured LLM API key from the expected environment variable."""
    api_key = os.getenv(config.llm_api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {config.llm_api_key_env}")
    return api_key


def build_tool_registry(
    config: BridgeConfig,
    camera_worker: Optional[CameraWorker],
    motion_manager: Optional[MotionManager],
) -> ToolRegistry:
    """Build and populate runtime tool registry based on config."""
    tool_registry = ToolRegistry()
    tool_registry.register(
        GoToSleepTool(
            wakeword_enabled=config.offline_wakeword_enabled,
            wakeword_aliases=config.offline_wakeword_aliases,
        )
    )
    tool_registry.register(EmotionTool(motion_manager))
    if config.camera_tool_enabled and camera_worker is not None:
        tool_registry.register(CameraSnapshotTool(camera_worker))
    if (
        config.speaker_id_enabled
        and camera_worker is not None
        and getattr(camera_worker, "_face_recognizer", None) is not None
    ):
        tool_registry.register(EnrollSpeakerTool(camera_worker))
    if DANCE_AVAILABLE and motion_manager is not None:
        tool_registry.register(DanceTool(motion_manager))
    if config.openclaw_enabled and config.openclaw_ws_url:
        openclaw_client = OpenClawGatewayClient(
            OpenClawGatewayConfig(
                ws_url=config.openclaw_ws_url,
                bearer_token=config.openclaw_bearer_token,
                timeout_s=config.openclaw_timeout_s,
                default_language=config.stt_language,
                delegate_model=config.openclaw_delegate_model,
            )
        )
        tool_registry.register(
            OpenClawDelegateTool(openclaw_client, default_language=config.stt_language)
        )
        if config.notification_callback_base_url:
            from bridge.tools.create_reminder import CreateReminderTool
            from bridge.tools.cancel_reminder import CancelReminderTool
            tool_registry.register(
                CreateReminderTool(openclaw_client, callback_url=config.notification_callback_base_url)
            )
            tool_registry.register(CancelReminderTool(openclaw_client))
    if config.home_assistant_enabled:
        if not config.home_assistant_token:
            logging.warning("HOME_ASSISTANT_ENABLED is true but HOME_ASSISTANT_TOKEN is empty")
        elif not config.home_assistant_url:
            logging.warning("HOME_ASSISTANT_ENABLED is true but HOME_ASSISTANT_URL is empty")
        else:
            home_assistant_client = HomeAssistantWsClient(
                HomeAssistantWsConfig(
                    ws_url=build_home_assistant_websocket_url(config.home_assistant_url),
                    access_token=config.home_assistant_token,
                    timeout_s=config.home_assistant_timeout_s,
                )
            )
            tool_registry.register(HomeAssistantDiscoverTool(home_assistant_client))
            tool_registry.register(
                HomeAssistantExecuteActionTool(
                    home_assistant_client,
                    sensitive_domains=config.home_assistant_sensitive_domains,
                )
            )
    return tool_registry


def build_wakeword_detector(config: BridgeConfig) -> OfflineWakewordDetector:
    """Build offline wakeword detector using bridge config values."""
    return OfflineWakewordDetector(
        aliases=config.offline_wakeword_aliases,
        threshold=config.offline_wakeword_threshold,
        enabled=config.offline_wakeword_enabled,
        fallback_on_speech=config.offline_wakeword_fallback_on_speech,
        speech_rms_threshold=config.offline_wakeword_fallback_speech_rms_threshold,
        auto_calibration_enabled=config.offline_wakeword_auto_calibration_enabled,
        calibration_seconds=config.offline_wakeword_calibration_seconds,
        calibration_multiplier=config.offline_wakeword_calibration_multiplier,
        model_path=config.wakeword_model_path,
    )


def build_realtime_session(
    *,
    api_key: str,
    config: BridgeConfig,
    identity_prompt: str,
    tool_specs: Optional[List[Dict[str, Any]]],
    on_tool_call: Optional[Callable[[str, Dict[str, Any]], Any]],
    on_speech_start: Optional[Callable[[], None]] = None,
    on_user_transcript: Optional[Callable[[str], None]] = None,
    on_assistant_text: Optional[Callable[[str], None]] = None,
    on_assistant_audio_chunk: Optional[Callable[[Any], None]] = None,
    on_assistant_audio_done: Optional[Callable[[], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
) -> OpenAIRealtimeSession:
    """Create a configured OpenAI Realtime session instance."""
    return OpenAIRealtimeSession(
        api_key=api_key,
        api_base=config.llm_api_base,
        model=config.realtime_model,
        instructions=identity_prompt,
        language=config.stt_language,
        transcription_model=config.realtime_transcription_model,
        vad_silence_ms=config.realtime_vad_silence_ms,
        vad_prefix_padding_ms=config.realtime_vad_prefix_padding_ms,
        on_speech_start=on_speech_start,
        on_user_transcript=on_user_transcript,
        on_assistant_text=on_assistant_text,
        on_assistant_audio_chunk=on_assistant_audio_chunk,
        on_assistant_audio_done=on_assistant_audio_done,
        on_error=on_error,
        tool_specs=tool_specs or [],
        on_tool_call=on_tool_call,
    )


def process_audio_queue(
    audio_queue: "Queue[tuple[str, Any]]",
    media_io: MediaIOPort,
    output_sample_rate: int,
    realtime_output_rate: int,
    playback_started: bool,
    audio_chunks_total: int,
    responses_streamed: int,
    elapsed_ms: Callable[[], int],
    state_machine: StateMachine,
    motion_manager: Optional[MotionManager],
) -> tuple[bool, int, int]:
    """Drain and process queued assistant audio events."""
    while True:
        try:
            kind, payload = audio_queue.get_nowait()
        except Empty:
            break

        if kind == "chunk":
            if not playback_started:
                logging.info("[%dms] Playback started", elapsed_ms())
                media_io.start_playing()
                playback_started = True
            chunk = payload
            if output_sample_rate != realtime_output_rate:
                chunk = resample_audio_chunk(chunk, realtime_output_rate, output_sample_rate)
            media_io.push_audio_sample(chunk)
            audio_chunks_total += 1
        elif kind == "done":
            responses_streamed += 1
            logging.debug("[%dms] Streamed response completed", elapsed_ms())
            apply_event(state_machine, Event.RESPONSE_READY, motion_manager)
        elif kind == "force_stop":
            if playback_started:
                logging.info("[%dms] Playback stopped reason=force_stop", elapsed_ms())
                media_io.stop_playing()
                playback_started = False

    return playback_started, audio_chunks_total, responses_streamed


def maybe_log_health(
    *,
    now: float,
    last_health_log: float,
    elapsed_ms: Callable[[], int],
    realtime: ConversationSessionPort,
    playback_started: bool,
    audio_queue: "Queue[tuple[str, Any]]",
    audio_chunks_total: int,
    responses_streamed: int,
    camera_worker: Optional[CameraWorker],
    text_queue: Optional[Queue[str]] = None,
    text_turns: Optional[int] = None,
) -> float:
    """Emit periodic health diagnostics and return updated timestamp."""
    if (now - last_health_log) <= 5.0:
        return last_health_log

    vision_info = None
    if camera_worker is not None:
        try:
            vision_info = camera_worker.get_tracking_debug_snapshot()
        except Exception:
            vision_info = None

    if text_queue is None:
        logging.debug(
            "[%dms] Health ready=%s playback=%s audio_q=%s chunks=%s streamed=%s face=%s eye=%s",
            elapsed_ms(),
            realtime.wait_until_ready(timeout_s=0.0),
            playback_started,
            audio_queue.qsize(),
            audio_chunks_total,
            responses_streamed,
            (vision_info or {}).get("face_detected_recently"),
            (vision_info or {}).get("eye_center"),
        )
    else:
        logging.debug(
            "[%dms] Health ready=%s playback=%s audio_q=%s text_q=%s chunks=%s streamed=%s turns=%s face=%s eye=%s",
            elapsed_ms(),
            realtime.wait_until_ready(timeout_s=0.0),
            playback_started,
            audio_queue.qsize(),
            text_queue.qsize(),
            audio_chunks_total,
            responses_streamed,
            text_turns,
            (vision_info or {}).get("face_detected_recently"),
            (vision_info or {}).get("eye_center"),
        )

    return now


def shutdown_runtime(
    *,
    playback_started: bool,
    reachy_sdk_instance: Any,
    realtime: OpenAIRealtimeSession,
    motion_manager: Optional[MotionManager],
    camera_worker: Optional[CameraWorker],
    stop_recording: bool,
) -> None:
    """Stop playback/session/workers in a safe order."""
    if playback_started:
        try:
            reachy_sdk_instance.media.stop_playing()
        except Exception:
            pass

    realtime.stop()

    if stop_recording:
        try:
            reachy_sdk_instance.media.stop_recording()
        except Exception:
            pass

    if motion_manager is not None:
        motion_manager.stop()
    if camera_worker is not None:
        camera_worker.stop()
