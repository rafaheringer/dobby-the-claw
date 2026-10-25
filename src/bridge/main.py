"""Entry point for the Dobby bridge process."""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time

from bridge.config import BridgeConfig
from bridge.runtime import (
    configure_third_party_loggers,
    initialize_reachy_runtime,
    load_identity_prompt,
    build_orchestrator,
    start_runtime_workers,
    stop_runtime_workers,
)
from bridge.state_machine import StateMachine
from web_ui.logbus import install_log_queue
from web_ui.server import start_web_ui


EXIT_DAEMON_LOST = 75  # scripts/run-windows.ps1 relaunches the bridge on this code
_DAEMON_LOST_AFTER_S = 10.0


def _watch_daemon_connection(sdk_instance) -> None:
    """Exit the process when the Reachy daemon connection stays down.

    The SDK client never reconnects, and the camera, motion and audio workers all share it, so a
    restart is the only clean recovery.
    """
    client = getattr(sdk_instance, "client", None)
    if client is None or not hasattr(client, "is_connected"):
        return
    down_since = None
    while True:
        time.sleep(1.0)
        try:
            alive = bool(client.is_connected())
        except Exception:
            alive = False
        if alive:
            down_since = None
            continue
        down_since = down_since or time.monotonic()
        if time.monotonic() - down_since >= _DAEMON_LOST_AFTER_S:
            logging.error("Lost the Reachy daemon connection for %.0fs; exiting to be relaunched", _DAEMON_LOST_AFTER_S)
            logging.shutdown()
            os._exit(EXIT_DAEMON_LOST)


def main() -> None:
    """Start the bridge: realtime voice plus the web UI (status, chat, camera, logs)."""
    parser = argparse.ArgumentParser(description="Dobby bridge")
    parser.add_argument(
        "--no-headtracking",
        action="store_true",
        help="Start with head tracking disabled (default follows HEAD_TRACKING_ENABLED).",
    )
    parser.add_argument(
        "--idle-sleep-timeout-s",
        type=float,
        default=None,
        help="Idle seconds before entering sleep mode (<=0 disables idle sleep).",
    )
    args = parser.parse_args()

    log_level_name = os.getenv("BRIDGE_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
    )
    configure_third_party_loggers(log_level)

    config = BridgeConfig.from_env()
    log_queue = install_log_queue() if config.web_ui_enabled else None
    start_with_headtracking = config.head_tracking_enabled and (not args.no_headtracking)
    idle_sleep_timeout_s = (
        float(args.idle_sleep_timeout_s)
        if args.idle_sleep_timeout_s is not None
        else float(config.idle_sleep_timeout_s)
    )
    state_machine = StateMachine()
    identity_prompt = load_identity_prompt()
    runtime = initialize_reachy_runtime(config)

    logging.info("Bridge starting")
    logging.info("Log level: %s", log_level_name)
    logging.info("Reachy Bridge API: %s", config.reachy_bridge_url)
    logging.info(
        "Startup options: headtracking=%s idle_sleep_timeout_s=%.1f offline_wakeword=%s",
        start_with_headtracking,
        idle_sleep_timeout_s,
        config.offline_wakeword_enabled,
    )

    try:
        # The SDK's wake_up() does not energize the motors: with torque off (after sleep) its smooth
        # move does nothing, and the later enable_motors() makes the head snap to the target.
        runtime.reachy.enable_motors()
        runtime.reachy.wake_up()
        logging.info("Reachy wake_up sent")
    except Exception as exc:
        logging.warning("Failed to wake Reachy at startup: %s", exc)

    if config.reachy_output_volume >= 0:
        try:
            runtime.reachy.set_output_volume(config.reachy_output_volume)
        except Exception as exc:
            logging.warning("Failed to set native Reachy output volume: %s", exc)

    if runtime.camera_worker is not None:
        runtime.camera_worker.set_head_tracking_enabled(start_with_headtracking)

    start_runtime_workers(runtime)
    if runtime.reachy_sdk_instance is not None:
        threading.Thread(
            target=_watch_daemon_connection, args=(runtime.reachy_sdk_instance,), name="daemon-watchdog", daemon=True
        ).start()

    orchestrator = build_orchestrator(
        state_machine=state_machine,
        reachy=runtime.reachy,
        motion_manager=runtime.motion_manager,
        camera_worker=runtime.camera_worker,
        config=config,
        reachy_sdk_instance=runtime.reachy_sdk_instance,
        identity_prompt=identity_prompt,
        idle_sleep_timeout_s=idle_sleep_timeout_s,
        accept_text=config.web_ui_enabled,
    )

    try:
        if log_queue is not None:
            start_web_ui(orchestrator, config, log_queue)
        orchestrator.run()
    except KeyboardInterrupt:
        logging.info("Bridge interrupted by user (Ctrl+C), shutting down")
    finally:
        stop_runtime_workers(runtime)
        try:
            runtime.reachy.goto_sleep()
            logging.info("Reachy goto_sleep sent")
        except Exception as exc:
            logging.warning("Failed to send Reachy goto_sleep at shutdown: %s", exc)


if __name__ == "__main__":
    main()
