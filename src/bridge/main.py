"""Entry point for the Dobby bridge process."""

from __future__ import annotations

import argparse
import logging
import os
import time

from bridge.config import BridgeConfig
from bridge.runtime import (
    configure_third_party_loggers,
    initialize_reachy_runtime,
    load_identity_prompt,
    run_chat_loop,
    run_realtime_loop,
    run_tui_loop,
    start_runtime_workers,
    stop_runtime_workers,
)
from bridge.state_machine import StateMachine


def main() -> None:
    """Start the bridge process in idle, realtime, or chat mode."""
    parser = argparse.ArgumentParser(description="Dobby bridge")
    parser.add_argument(
        "--mode",
        choices=["idle", "realtime", "chat", "tui"],
        default=os.getenv("BRIDGE_MODE", "realtime"),
    )
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
    logging.info("Run mode: %s", args.mode)
    logging.info(
        "Vision debug window=%s log_interval=%.1fs",
        config.vision_debug_window,
        config.vision_debug_log_interval_s,
    )
    logging.info(
        "Startup options: headtracking=%s idle_sleep_timeout_s=%.1f offline_wakeword=%s",
        start_with_headtracking,
        idle_sleep_timeout_s,
        config.offline_wakeword_enabled,
    )

    try:
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

    try:
        if args.mode == "realtime":
            run_realtime_loop(
                state_machine=state_machine,
                reachy=runtime.reachy,
                motion_manager=runtime.motion_manager,
                camera_worker=runtime.camera_worker,
                config=config,
                reachy_sdk_instance=runtime.reachy_sdk_instance,
                identity_prompt=identity_prompt,
                idle_sleep_timeout_s=idle_sleep_timeout_s,
            )
            return

        if args.mode == "chat":
            run_chat_loop(
                state_machine=state_machine,
                reachy=runtime.reachy,
                motion_manager=runtime.motion_manager,
                camera_worker=runtime.camera_worker,
                config=config,
                reachy_sdk_instance=runtime.reachy_sdk_instance,
                identity_prompt=identity_prompt,
                idle_sleep_timeout_s=idle_sleep_timeout_s,
            )
            return

        if args.mode == "tui":
            run_tui_loop(
                state_machine=state_machine,
                reachy=runtime.reachy,
                motion_manager=runtime.motion_manager,
                camera_worker=runtime.camera_worker,
                config=config,
                reachy_sdk_instance=runtime.reachy_sdk_instance,
                identity_prompt=identity_prompt,
                idle_sleep_timeout_s=idle_sleep_timeout_s,
            )
            return

        while True:
            time.sleep(1)
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
