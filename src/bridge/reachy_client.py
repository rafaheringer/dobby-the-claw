from typing import Any, Dict, Optional
import time

from reachy_mini.utils import create_head_pose


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

    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if self._use_sdk:
            if self._reachy_mini is None:
                raise RuntimeError("Reachy Mini SDK not available")
            return self._execute_action_sdk(action)

        # TODO: Implement reachy-bridge API call.
        raise NotImplementedError("Reachy client not implemented yet")

    def _execute_action_sdk(self, action: Dict[str, Any]) -> Dict[str, Any]:
        action_type = (action.get("type") or action.get("action") or "").strip().lower()
        if action_type in {"gesture.antenna_wave", "gesture.think", "gesture.listening"}:
            amplitude = float(action.get("amplitude_rad", 0.35))
            duration_s = float(action.get("duration_s", 0.6))
            mini = self.get_sdk_instance()
            start = mini.get_present_antenna_joint_positions()
            mini.set_target_antenna_joint_positions([amplitude, -amplitude])
            time.sleep(duration_s + 0.2)
            mini.set_target_antenna_joint_positions(start)
            time.sleep(duration_s)
            return {"ok": True, "message": "Antenna gesture complete"}

        if action_type in {"move_head", "gesture.head"}:
            yaw = float(action.get("yaw", 0.0))
            pitch = float(action.get("pitch", 0.0))
            roll = float(action.get("roll", 0.0))
            mini = self.get_sdk_instance()
            pose = create_head_pose(roll=roll, pitch=pitch, yaw=yaw, degrees=True)
            mini.set_target_head_pose(pose)
            return {"ok": True, "message": "Head target set"}

        if action_type in {"look_at", "gaze.look_at"}:
            u = int(action.get("u", 0))
            v = int(action.get("v", 0))
            duration_s = float(action.get("duration_s", 0.0))
            mini = self.get_sdk_instance()
            mini.look_at_image(u, v, duration=duration_s, perform_movement=True)
            return {"ok": True, "message": "Look-at completed"}

        if action_type == "antenna_gesture":
            amplitude = float(action.get("amplitude_rad", 0.2))
            cycles = int(action.get("cycles", 2))
            duration_s = float(action.get("duration_s", 0.4))
            mini = self.get_sdk_instance()
            start = mini.get_present_antenna_joint_positions()
            for i in range(max(cycles, 1)):
                value = amplitude if i % 2 == 0 else -amplitude
                mini.set_target_antenna_joint_positions([value, -value])
                time.sleep(duration_s)
            mini.set_target_antenna_joint_positions(start)
            return {"ok": True, "message": "Antenna gesture complete"}

        return {"ok": False, "message": f"Unsupported action type: {action_type}"}

    def get_sdk_instance(self):
        if not self._use_sdk or self._reachy_mini is None:
            raise RuntimeError("Reachy Mini SDK not available")
        if self._sdk_instance is None:
            self._sdk_instance = self._reachy_mini()
        return self._sdk_instance
