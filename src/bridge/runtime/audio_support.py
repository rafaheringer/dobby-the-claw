"""Shared audio queue processing and runtime health/shutdown helpers."""

from __future__ import annotations

import logging
import os
import time
from queue import Empty, Queue
from typing import Any, Callable, Optional

from bridge.config import BridgeConfig
from bridge.reachy.camera_worker import CameraWorker
from bridge.reachy.motion import MotionManager
from bridge.reachy.realtime_client import OpenAIRealtimeSession
from bridge.state_machine import Event, StateMachine
from bridge.tools import CameraSnapshotTool, ToolRegistry

from bridge.runtime.common import apply_event, resample_audio_chunk


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


def build_tool_registry(config: BridgeConfig, camera_worker: Optional[CameraWorker]) -> ToolRegistry:
    """Build and populate runtime tool registry based on config."""
    tool_registry = ToolRegistry()
    if config.camera_tool_enabled and camera_worker is not None:
        tool_registry.register(CameraSnapshotTool(camera_worker))
    return tool_registry


def process_audio_queue(
    audio_queue: "Queue[tuple[str, Any]]",
    reachy_sdk_instance: Any,
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
                reachy_sdk_instance.media.start_playing()
                playback_started = True
            chunk = payload
            if output_sample_rate != realtime_output_rate:
                chunk = resample_audio_chunk(chunk, realtime_output_rate, output_sample_rate)
            reachy_sdk_instance.media.push_audio_sample(chunk)
            audio_chunks_total += 1
        elif kind == "done":
            responses_streamed += 1
            logging.debug("[%dms] Streamed response completed", elapsed_ms())
            apply_event(state_machine, Event.RESPONSE_READY, motion_manager)
        elif kind == "force_stop":
            if playback_started:
                logging.info("[%dms] Playback stopped reason=force_stop", elapsed_ms())
                reachy_sdk_instance.media.stop_playing()
                playback_started = False

    return playback_started, audio_chunks_total, responses_streamed


def maybe_log_health(
    *,
    now: float,
    last_health_log: float,
    elapsed_ms: Callable[[], int],
    realtime: OpenAIRealtimeSession,
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
