"""Reachy action client with SDK-backed execution helpers."""

from typing import Any
import logging
import time
import tempfile
import platform
import subprocess

from reachy_mini.utils import create_head_pose

from reachy.actions import (
    AntennaCycleGestureAction,
    AntennaWaveGestureAction,
    CameraCaptureSnapshotAction,
    GazeLookAtAction,
    HeadMoveAction,
    ListeningGestureAction,
    PlayRecordedMoveAction,
    ReachyAction,
    ThinkGestureAction,
)
from reachy.results import ReachyActionResult


class ReachyClient:
    """Execute typed Reachy actions via SDK integration path."""

    def __init__(self, base_url: str) -> None:
        """Initialize client and detect whether SDK mode is enabled."""
        self.base_url = base_url
        self._use_sdk = base_url.strip().lower().startswith("sdk")
        self._reachy_mini = None
        self._sdk_instance = None
        self._recorded_moves_cache: dict[str, Any] = {}
        if self._use_sdk:
            try:
                from reachy_mini import ReachyMini

                self._reachy_mini = ReachyMini
            except ImportError:
                self._reachy_mini = None

    def execute_typed_action(self, action: ReachyAction) -> ReachyActionResult:
        """Execute a typed Reachy action and return normalized result."""
        if self._use_sdk:
            if self._reachy_mini is None:
                raise RuntimeError("Reachy Mini SDK not available")
            return self._execute_typed_action_sdk(action)

        raise NotImplementedError("Reachy client not implemented yet")

    def _execute_typed_action_sdk(self, action: ReachyAction) -> ReachyActionResult:
        """Dispatch typed action to specific SDK execution helper."""
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
        if isinstance(action, PlayRecordedMoveAction):
            return self._run_recorded_move_sdk(
                dataset_name=action.dataset_name,
                move_name=action.move_name,
                play_frequency=action.play_frequency,
                initial_goto_duration=action.initial_goto_duration,
                sound=action.sound,
            )
        return ReachyActionResult(ok=False, message=f"Unsupported typed action: {type(action).__name__}")

    def _run_antenna_gesture_sdk(self, amplitude_rad: float, duration_s: float) -> ReachyActionResult:
        """Run a symmetric antenna gesture and restore initial position."""
        mini = self.get_sdk_instance()
        start = mini.get_present_antenna_joint_positions()
        mini.set_target_antenna_joint_positions([amplitude_rad, -amplitude_rad])
        time.sleep(duration_s + 0.2)
        mini.set_target_antenna_joint_positions(start)
        time.sleep(duration_s)
        return ReachyActionResult(ok=True, message="Antenna gesture complete")

    def _run_move_head_sdk(self, yaw: float, pitch: float, roll: float) -> ReachyActionResult:
        """Set Reachy head pose target from yaw/pitch/roll angles."""
        mini = self.get_sdk_instance()
        pose = create_head_pose(roll=roll, pitch=pitch, yaw=yaw, degrees=True)
        mini.set_target_head_pose(pose)
        return ReachyActionResult(ok=True, message="Head target set")

    def _run_look_at_sdk(self, u: int, v: int, duration_s: float) -> ReachyActionResult:
        """Execute gaze look-at command using image coordinates."""
        mini = self.get_sdk_instance()
        mini.look_at_image(u, v, duration=duration_s, perform_movement=True)
        return ReachyActionResult(ok=True, message="Look-at completed")

    def _run_antenna_cycles_sdk(self, amplitude_rad: float, cycles: int, duration_s: float) -> ReachyActionResult:
        """Run alternating antenna cycles and restore initial position."""
        mini = self.get_sdk_instance()
        start = mini.get_present_antenna_joint_positions()
        for index in range(max(cycles, 1)):
            value = amplitude_rad if index % 2 == 0 else -amplitude_rad
            mini.set_target_antenna_joint_positions([value, -value])
            time.sleep(duration_s)
        mini.set_target_antenna_joint_positions(start)
        return ReachyActionResult(ok=True, message="Antenna gesture complete")

    def _run_camera_snapshot_sdk(self) -> ReachyActionResult:
        """Capture camera frame and save it to a temporary JPEG file."""
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

    def _run_recorded_move_sdk(
        self,
        *,
        dataset_name: str,
        move_name: str,
        play_frequency: float,
        initial_goto_duration: float,
        sound: bool,
    ) -> ReachyActionResult:
        """Play one recorded move from a HuggingFace dataset via Reachy SDK."""
        mini = self.get_sdk_instance()
        dataset = str(dataset_name).strip()
        move = str(move_name).strip()
        if not dataset:
            return ReachyActionResult(ok=False, message="dataset_name is required")
        if not move:
            return ReachyActionResult(ok=False, message="move_name is required")

        try:
            from reachy_mini.motion.recorded_move import RecordedMoves

            recorded_moves = self._recorded_moves_cache.get(dataset)
            if recorded_moves is None:
                recorded_moves = RecordedMoves(dataset)
                self._recorded_moves_cache[dataset] = recorded_moves

            recorded_move = recorded_moves.get(move)
            mini.async_play_move(
                recorded_move,
                play_frequency=float(play_frequency),
                initial_goto_duration=float(initial_goto_duration),
                sound=bool(sound),
            )
            return ReachyActionResult(
                ok=True,
                message=f"Recorded move queued dataset={dataset} move={move}",
            )
        except Exception as exc:
            return ReachyActionResult(
                ok=False,
                message=f"Failed to play recorded move '{move}' from '{dataset}': {exc}",
            )

    def get_sdk_instance(self):
        """Lazily create and return singleton Reachy SDK instance."""
        if not self._use_sdk or self._reachy_mini is None:
            raise RuntimeError("Reachy Mini SDK not available")
        if self._sdk_instance is None:
            self._sdk_instance = self._reachy_mini()
        return self._sdk_instance

    def wake_up(self) -> None:
        """Wake Reachy if SDK exposes wake capability."""
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
        """Put Reachy in sleep posture if supported by SDK."""
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

    def disable_motors(self) -> None:
        """Cut torque on all motors (compliant/free mode) to save power."""
        if not self._use_sdk:
            return
        mini: Any = self.get_sdk_instance()
        if hasattr(mini, "disable_motors"):
            mini.disable_motors()

    def enable_motors(self) -> None:
        """Re-energize all motors after sleep."""
        if not self._use_sdk:
            return
        mini: Any = self.get_sdk_instance()
        if hasattr(mini, "enable_motors"):
            mini.enable_motors()

    def set_output_volume(self, volume: int) -> None:
        """Set Reachy speaker output volume on Linux via `amixer`."""
        if not self._use_sdk:
            logging.warning("Reachy set_output_volume ignored: non-SDK client path is not implemented")
            return

        self.get_sdk_instance()
        level = max(0, min(100, int(volume)))

        if platform.system() != "Linux":
            raise RuntimeError("Native Reachy output volume control is currently implemented only on Linux")

        card_name = self._get_linux_audio_card_name()
        try:
            subprocess.run(
                ["amixer", "-c", card_name, "sset", "PCM", f"{level}%"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=True,
            )
            subprocess.run(
                ["amixer", "-c", card_name, "sset", "PCM,1", "100%"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=True,
            )
            logging.info("Reachy output volume set to %s via amixer card=%s", level, card_name)
            return
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Failed to set Reachy output volume via amixer: {exc}") from exc

    def _get_linux_audio_card_name(self) -> str:
        """Best-effort detection of ALSA card name for Reachy output."""
        try:
            result = subprocess.run(
                ["aplay", "-l"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            output = result.stdout.lower()
            if "reachy mini audio" in output:
                return "Audio"
            if "respeaker" in output:
                return "Array"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return "Audio"
