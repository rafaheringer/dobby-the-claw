"""TUI dashboard mode — wraps realtime voice + text input in a Textual app."""

from __future__ import annotations

import logging
import queue
import sys
import threading
from typing import Any, Optional

from bridge.config import BridgeConfig
from reachy.camera_worker import CameraWorker
from reachy.client import ReachyClient
from reachy.motion import MotionManager
from bridge.state_machine import StateMachine

from bridge.runtime.ha_client_factory import build_ha_client
from bridge.runtime.orchestrator import RuntimeOrchestrator
from bridge.tui.app import DobbyTUI
from bridge.tui.log_handler import TUILogHandler


def run_tui_loop(
    state_machine: StateMachine,
    reachy: ReachyClient,
    motion_manager: Optional[MotionManager],
    camera_worker: Optional[CameraWorker],
    config: BridgeConfig,
    reachy_sdk_instance: Any,
    identity_prompt: str,
    idle_sleep_timeout_s: float,
) -> None:
    """Run the bridge with a live TUI dashboard (status + logs + text input)."""
    log_queue: queue.Queue[logging.LogRecord] = queue.Queue()

    orchestrator = RuntimeOrchestrator(
        mode_name="TUI",
        state_machine=state_machine,
        reachy=reachy,
        motion_manager=motion_manager,
        camera_worker=camera_worker,
        config=config,
        reachy_sdk_instance=reachy_sdk_instance,
        identity_prompt=identity_prompt,
        idle_sleep_timeout_s=idle_sleep_timeout_s,
        interactive_text=True,
        active_mode_uses_mic_recording=True,
        skip_stdin=True,
        ha_client=build_ha_client(config),
    )

    app = DobbyTUI(
        config=config,
        text_queue=orchestrator.text_queue,
        log_queue=log_queue,
        mode_name="realtime",
    )

    # Route all log output through the TUI queue; suppress terminal stream handlers
    root_logger = logging.getLogger()
    tui_handler = TUILogHandler(log_queue)
    tui_handler.setFormatter(logging.Formatter())
    original_handlers = root_logger.handlers[:]
    for h in original_handlers:
        if isinstance(h, logging.StreamHandler) and h.stream in (sys.stdout, sys.stderr):
            root_logger.removeHandler(h)
    root_logger.addHandler(tui_handler)

    # Run orchestrator in background daemon thread
    orch_thread = threading.Thread(target=orchestrator.run, daemon=True, name="orchestrator")
    orch_thread.start()

    try:
        app.run()
    finally:
        # Restore original log handlers
        root_logger.removeHandler(tui_handler)
        for h in original_handlers:
            root_logger.addHandler(h)
        # Ensure orchestrator exits (on_unmount already posted /quit; belt-and-suspenders)
        orchestrator.text_queue.put("/quit")
        orch_thread.join(timeout=5)
