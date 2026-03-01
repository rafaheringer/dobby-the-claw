from typing import Any
import logging
import time
import tempfile
import json
from urllib import error, request

from reachy_mini.utils import create_head_pose

from bridge.reachy.actions import (
    AntennaCycleGestureAction,
    AntennaWaveGestureAction,
    CameraCaptureSnapshotAction,
    GazeLookAtAction,
    HeadMoveAction,
    ListeningGestureAction,
    ReachyAction,
    ThinkGestureAction,
)
from bridge.reachy.results import ReachyActionResult


class ReachyClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self._use_sdk = base_url.strip().lower().startswith("sdk")
        self._reachy_mini = None
        self._sdk_instance = None
        if self._use_sdk:
            try:
                from reachy_mini import ReachyMini

                self._reachy_mini = ReachyMini
            except ImportError:
                self._reachy_mini = None

    def execute_typed_action(self, action: ReachyAction) -> ReachyActionResult:
        if self._use_sdk:
            if self._reachy_mini is None:
                raise RuntimeError("Reachy Mini SDK not available")
            return self._execute_typed_action_sdk(action)

        raise NotImplementedError("Reachy client not implemented yet")

    def _execute_typed_action_sdk(self, action: ReachyAction) -> ReachyActionResult:
        if isinstance(action, (AntennaWaveGestureAction, ThinkGestureAction, ListeningGestureAction)):
            return self._run_antenna_gesture_sdk(
                amplitude_rad=float(action.amplitude_rad),
                duration_s=float(action.duration_s),
            )
        if isinstance(action, HeadMoveAction):
            return self._run_move_head_sdk(yaw=action.yaw, pitch=action.pitch, roll=action.roll)
        if isinstance(action, GazeLookAtAction):
            return self._run_look_at_sdk(u=action.u, v=action.v, duration_s=action.duration_s)
        if isinstance(action, AntennaCycleGestureAction):
            return self._run_antenna_cycles_sdk(
                amplitude_rad=action.amplitude_rad,
                cycles=action.cycles,
                duration_s=action.duration_s,
            )
        if isinstance(action, CameraCaptureSnapshotAction):
            return self._run_camera_snapshot_sdk()
        return ReachyActionResult(ok=False, message=f"Unsupported typed action: {type(action).__name__}")

    def _run_antenna_gesture_sdk(self, amplitude_rad: float, duration_s: float) -> ReachyActionResult:
        mini = self.get_sdk_instance()
        start = mini.get_present_antenna_joint_positions()
        mini.set_target_antenna_joint_positions([amplitude_rad, -amplitude_rad])
        time.sleep(duration_s + 0.2)
        mini.set_target_antenna_joint_positions(start)
        time.sleep(duration_s)
        return ReachyActionResult(ok=True, message="Antenna gesture complete")

    def _run_move_head_sdk(self, yaw: float, pitch: float, roll: float) -> ReachyActionResult:
        mini = self.get_sdk_instance()
        pose = create_head_pose(roll=roll, pitch=pitch, yaw=yaw, degrees=True)
        mini.set_target_head_pose(pose)
        return ReachyActionResult(ok=True, message="Head target set")

    def _run_look_at_sdk(self, u: int, v: int, duration_s: float) -> ReachyActionResult:
        mini = self.get_sdk_instance()
        mini.look_at_image(u, v, duration=duration_s, perform_movement=True)
        return ReachyActionResult(ok=True, message="Look-at completed")

    def _run_antenna_cycles_sdk(self, amplitude_rad: float, cycles: int, duration_s: float) -> ReachyActionResult:
        mini = self.get_sdk_instance()
        start = mini.get_present_antenna_joint_positions()
        for index in range(max(cycles, 1)):
            value = amplitude_rad if index % 2 == 0 else -amplitude_rad
            mini.set_target_antenna_joint_positions([value, -value])
            time.sleep(duration_s)
        mini.set_target_antenna_joint_positions(start)
        return ReachyActionResult(ok=True, message="Antenna gesture complete")

    def _run_camera_snapshot_sdk(self) -> ReachyActionResult:
        mini = self.get_sdk_instance()
        frame = mini.media.get_frame()
        if frame is None:
            return ReachyActionResult(ok=False, message="No frame available")
        try:
            import cv2
        except ImportError:
            return ReachyActionResult(ok=False, message="opencv-python not installed")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            path = temp_file.name
        cv2.imwrite(path, frame)
        return ReachyActionResult(ok=True, message="Snapshot saved", path=path)

    def get_sdk_instance(self):
        if not self._use_sdk or self._reachy_mini is None:
            raise RuntimeError("Reachy Mini SDK not available")
        if self._sdk_instance is None:
            self._sdk_instance = self._reachy_mini()
        return self._sdk_instance

    def wake_up(self) -> None:
        if not self._use_sdk:
            logging.warning("Reachy wake_up ignored: non-SDK client path is not implemented")
            return

        mini = self.get_sdk_instance()
        mini_any: Any = mini
        if hasattr(mini_any, "wake_up"):
            mini_any.wake_up()
            return
        if hasattr(mini_any, "wake"):
            mini_any.wake()
            return
        raise RuntimeError("Reachy SDK instance does not expose wake_up/wake")

    def goto_sleep(self) -> None:
        if not self._use_sdk:
            logging.warning("Reachy goto_sleep ignored: non-SDK client path is not implemented")
            return

        mini = self.get_sdk_instance()
        mini_any: Any = mini
        if hasattr(mini_any, "goto_sleep"):
            mini_any.goto_sleep()
            return
        if hasattr(mini_any, "sleep"):
            mini_any.sleep()
            return
        raise RuntimeError("Reachy SDK instance does not expose goto_sleep/sleep")

    def set_output_volume(self, volume: int) -> None:
        if not self._use_sdk:
            logging.warning("Reachy set_output_volume ignored: non-SDK client path is not implemented")
            return

        mini = self.get_sdk_instance()
        mini_any: Any = mini
        level = max(0, min(100, int(volume)))

        if hasattr(mini_any, "set_output_volume"):
            mini_any.set_output_volume(level)
            return

        status: dict[str, Any] = {}
        try:
            status_raw = mini_any.client.get_status()
            if isinstance(status_raw, dict):
                status = status_raw
        except Exception:
            status = {}


        payload = json.dumps({"volume": level}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        last_error: Exception | None = None

        url = f"http://127.0.0.1:8000/api/volume/set"
        req = request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=2.0) as response:
                if 200 <= int(response.status) < 300:
                    logging.info("Reachy output volume set to %s via %s", level, url)
                    return
        except (error.HTTPError, error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc

        raise RuntimeError(
            f"Failed to set Reachy output volume via native daemon API; last_error={last_error}"
        )
