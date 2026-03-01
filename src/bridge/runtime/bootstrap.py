"""Bootstrap utilities for bridge runtime setup."""

from __future__ import annotations

import logging
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from bridge.config import BridgeConfig
from reachy.camera_worker import CameraWorker
from reachy.client import ReachyClient
from reachy.motion import MotionManager


def _build_head_tracker() -> Any:
    """Create optional MediaPipe-based head tracker when available."""
    if importlib.util.find_spec("mediapipe") is None:
        logging.warning("MediaPipe not installed; head tracking backend unavailable")
        return None

    try:
        from reachy_mini_toolbox.vision import HeadTracker

        tracker = HeadTracker()
        logging.info("Head tracking backend: MediaPipe HeadTracker")
        return tracker
    except Exception as exc:
        logging.warning("Failed to initialize MediaPipe HeadTracker: %s", exc)
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

    if config.reachy_bridge_url.strip().lower().startswith("sdk"):
        try:
            reachy_sdk_instance = reachy.get_sdk_instance()
            head_tracker = _build_head_tracker()
            camera_worker = CameraWorker(
                reachy_sdk_instance,
                head_tracker=head_tracker,
                debug_visual_window=config.vision_debug_window,
                debug_log_interval_s=config.vision_debug_log_interval_s,
            )
            motion_manager = MotionManager(
                reachy_sdk_instance,
                camera_worker=camera_worker,
            )
        except Exception:
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
        except Exception as exc:
            logging.warning("Failed to stop motion manager: %s", exc)

    if runtime.camera_worker is not None:
        try:
            runtime.camera_worker.stop()
        except Exception as exc:
            logging.warning("Failed to stop camera worker: %s", exc)


def load_identity_prompt() -> str:
    """Load robot identity instructions from `prompts/identity.txt`."""
    src_root = Path(__file__).resolve().parents[2]
    identity_path = src_root / "prompts" / "identity.txt"
    try:
        content = identity_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"Failed to load identity prompt at {identity_path}: {exc}") from exc

    if not content:
        raise RuntimeError(f"Identity prompt is empty: {identity_path}")

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
