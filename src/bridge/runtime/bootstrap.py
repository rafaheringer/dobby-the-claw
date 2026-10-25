"""Bootstrap utilities for bridge runtime setup."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import os

from bridge.config import BridgeConfig
from reachy.camera_worker import CameraWorker, apply_fixed_mic_gain
from reachy.client import ReachyClient
from reachy.face_recognizer import FaceRecognizer
from reachy.motion import MotionManager
from reachy.person_detector import PersonDetector


def _build_face_detector() -> Any:
    """Create the SDK's YuNet face detector (the daemon's head tracking uses the same one)."""
    try:
        from reachy_mini.vision.face_detector import FaceDetector

        detector = FaceDetector()
        logging.info("Face detector: YuNet (reachy_mini.vision)")
        return detector
    except Exception as exc:
        logging.warning("Failed to initialize the YuNet face detector: %s", exc)
        return None


def _build_tilt_tracker() -> Any:
    """Create the MediaPipe Face Mesh tracker that measures head tilt on face crops, when available."""
    try:
        from reachy_mini_toolbox.vision import HeadTracker

        return HeadTracker()
    except Exception as exc:
        logging.warning("Head tilt imitation unavailable (MediaPipe Face Mesh): %s", exc)
        return None


@dataclass
class ReachyRuntime:
    """Reachy runtime objects initialized from environment config."""

    reachy: ReachyClient
    reachy_sdk_instance: Any
    camera_worker: Optional[CameraWorker]
    motion_manager: Optional[MotionManager]


def initialize_reachy_runtime(config: BridgeConfig) -> ReachyRuntime:
    """Initialize Reachy client and optional SDK camera/motion workers."""
    reachy = ReachyClient(config.reachy_bridge_url)
    reachy_sdk_instance = None
    camera_worker = None
    motion_manager = None
    is_sdk_runtime = config.reachy_bridge_url.strip().lower().startswith("sdk")

    face_recognizer: FaceRecognizer | None = None
    if config.speaker_id_enabled:
        profiles_dir = os.path.expanduser(config.speaker_id_profiles_dir)
        face_recognizer = FaceRecognizer(profiles_dir)
        if not face_recognizer.available:
            logging.warning(
                "SPEAKER_ID_ENABLED=1 but insightface is not available; "
                "install with: pip install insightface onnxruntime"
            )
            face_recognizer = None

    needs_video = (
        config.head_tracking_enabled
        or config.camera_tool_enabled
        or config.speaker_id_enabled
        or config.antenna_finger_tracking_enabled
    )

    if is_sdk_runtime:
        try:
            reachy_sdk_instance = reachy.get_sdk_instance()
            if needs_video:
                face_detector = _build_face_detector()
                tilt_tracker = _build_tilt_tracker()
                person_detector = PersonDetector(config.person_detector_model) if config.presence_enabled else None
                camera_worker = CameraWorker(
                    reachy_sdk_instance,
                    face_detector=face_detector,
                    tilt_tracker=tilt_tracker,
                    tracking_max_width=config.vision_tracking_max_width,
                    tracking_fps=config.vision_tracking_fps,
                    antenna_finger_tracking_enabled=config.antenna_finger_tracking_enabled,
                    antenna_finger_max_angle_deg=config.antenna_finger_max_angle_deg,
                    face_recognizer=face_recognizer,
                    person_detector=person_detector,
                    mic_fixed_gain=config.mic_fixed_gain,
                )
            else:
                logging.info(
                    "Vision features disabled (camera tool, head tracking, speaker id, "
                    "finger tracking all off); skipping CameraWorker to save CPU"
                )
                if config.mic_fixed_gain > 0:
                    apply_fixed_mic_gain(config.mic_fixed_gain)  # the camera worker keeps it otherwise
            motion_manager = MotionManager(
                reachy_sdk_instance,
                camera_worker=camera_worker,
            )
        except Exception as exc:
            logging.warning("Failed to initialize Reachy SDK runtime: %s", exc)
            reachy_sdk_instance = None
            camera_worker = None
            motion_manager = None

    return ReachyRuntime(
        reachy=reachy,
        reachy_sdk_instance=reachy_sdk_instance,
        camera_worker=camera_worker,
        motion_manager=motion_manager,
    )


def start_runtime_workers(runtime: ReachyRuntime) -> None:
    """Start optional SDK worker threads after lifecycle wake-up completes."""
    if runtime.camera_worker is not None:
        runtime.camera_worker.start()
    if runtime.motion_manager is not None:
        runtime.motion_manager.start()


def stop_runtime_workers(runtime: ReachyRuntime) -> None:
    """Stop optional SDK worker threads before lifecycle sleep/shutdown."""
    if runtime.motion_manager is not None:
        try:
            runtime.motion_manager.stop()
        except BaseException as exc:
            logging.warning("Failed to stop motion manager: %s", exc)

    if runtime.camera_worker is not None:
        try:
            runtime.camera_worker.stop()
        except BaseException as exc:
            logging.warning("Failed to stop camera worker: %s", exc)


def load_identity_prompt() -> str:
    """Load robot identity instructions from `prompts/identity.md`."""
    return load_prompt("identity.md")


def load_prompt(filename: str) -> str:
    """Load a non-empty prompt file from the `prompts/` directory."""
    src_root = Path(__file__).resolve().parents[2]
    prompt_path = src_root / "prompts" / filename
    try:
        content = prompt_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Failed to load prompt at {prompt_path}: {exc}") from exc

    if not content:
        raise RuntimeError(f"Prompt is empty: {prompt_path}")

    return content


def configure_third_party_loggers(app_log_level: int) -> None:
    """Tune noisy dependency loggers while preserving app diagnostics."""
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
