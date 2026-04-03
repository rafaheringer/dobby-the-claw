"""Realtime voice mode runtime loop."""

from __future__ import annotations

from typing import Any, Optional

from bridge.config import BridgeConfig
from reachy.camera_worker import CameraWorker
from reachy.client import ReachyClient
from reachy.motion import MotionManager
from bridge.state_machine import StateMachine

from bridge.runtime.ha_client_factory import build_ha_client
from bridge.runtime.orchestrator import RuntimeOrchestrator


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
    orchestrator = RuntimeOrchestrator(
        mode_name="Realtime",
        state_machine=state_machine,
        reachy=reachy,
        motion_manager=motion_manager,
        camera_worker=camera_worker,
        config=config,
        reachy_sdk_instance=reachy_sdk_instance,
        identity_prompt=identity_prompt,
        idle_sleep_timeout_s=idle_sleep_timeout_s,
        interactive_text=False,
        active_mode_uses_mic_recording=True,
        ha_client=build_ha_client(config),
    )
    orchestrator.run()
