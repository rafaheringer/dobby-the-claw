"""Movement manager aligned with Reachy conversation app architecture.

This module implements:
- A primary move queue (sequential moves),
- Secondary additive offsets (speech and face tracking),
- A single high-frequency control loop calling `set_target`.
"""

from __future__ import annotations

import logging
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import compose_world_offset, linear_pose_interpolation

from bridge.speech_tapper import HOP_MS, SwayRollRT
from bridge.state_machine import State

logger = logging.getLogger(__name__)

CONTROL_LOOP_FREQUENCY_HZ = 100.0
FullBodyPose = Tuple[NDArray[np.float64], Tuple[float, float], float]


class Move:
    """Minimal move protocol used by the primary move queue."""

    @property
    def duration(self) -> float:
        """Return move duration in seconds."""
        raise NotImplementedError

    def evaluate(self, t: float) -> Tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate move at elapsed time `t`."""
        raise NotImplementedError


class BreathingMove(Move):
    """Breathing move ported from conversation app parameters."""

    def __init__(
        self,
        interpolation_start_pose: NDArray[np.float64],
        interpolation_start_antennas: Tuple[float, float],
        interpolation_duration: float = 1.0,
    ) -> None:
        """Initialize breathing interpolation and periodic motion parameters."""
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_duration = interpolation_duration

        self.neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.neutral_antennas = np.array([0.0, 0.0])

        self.breathing_z_amplitude = 0.005
        self.breathing_frequency = 0.1
        self.antenna_sway_amplitude = np.deg2rad(15)
        self.antenna_frequency = 0.5

    @property
    def duration(self) -> float:
        """Run continuously until interrupted by another move."""
        return float("inf")

    def evaluate(self, t: float) -> Tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]:
        """Evaluate breathing pose and antenna sway at time `t`."""
        if t < self.interpolation_duration:
            interpolation_t = t / self.interpolation_duration
            head_pose = linear_pose_interpolation(
                self.interpolation_start_pose,
                self.neutral_head_pose,
                interpolation_t,
            )
            antennas_interp = (1 - interpolation_t) * self.interpolation_start_antennas + interpolation_t * self.neutral_antennas
            antennas = antennas_interp.astype(np.float64)
        else:
            breathing_time = t - self.interpolation_duration
            z_offset = self.breathing_z_amplitude * np.sin(2 * np.pi * self.breathing_frequency * breathing_time)
            head_pose = create_head_pose(x=0, y=0, z=z_offset, roll=0, pitch=0, yaw=0, degrees=True, mm=False)
            antenna_sway = self.antenna_sway_amplitude * np.sin(2 * np.pi * self.antenna_frequency * breathing_time)
            antennas = np.array([antenna_sway, -antenna_sway], dtype=np.float64)

        return (head_pose, antennas, 0.0)


def combine_full_body(primary_pose: FullBodyPose, secondary_pose: FullBodyPose) -> FullBodyPose:
    """Combine primary absolute pose with secondary world offsets."""
    primary_head, primary_antennas, primary_body_yaw = primary_pose
    secondary_head, secondary_antennas, secondary_body_yaw = secondary_pose

    combined_head = compose_world_offset(primary_head, secondary_head, reorthonormalize=True)
    combined_antennas = (
        primary_antennas[0] + secondary_antennas[0],
        primary_antennas[1] + secondary_antennas[1],
    )
    combined_body_yaw = primary_body_yaw + secondary_body_yaw
    return (combined_head, combined_antennas, combined_body_yaw)


def clone_full_body_pose(pose: FullBodyPose) -> FullBodyPose:
    """Create a deep-copy-like tuple copy for pose snapshots."""
    head, antennas, body_yaw = pose
    return (head.copy(), (float(antennas[0]), float(antennas[1])), float(body_yaw))


@dataclass
class MovementState:
    """State container for movement loop runtime data."""

    current_move: Move | None = None
    move_start_time: float | None = None
    last_activity_time: float = 0.0
    speech_offsets: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    face_tracking_offsets: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    last_primary_pose: FullBodyPose | None = None

    def update_activity(self) -> None:
        """Mark current monotonic time as latest activity."""
        self.last_activity_time = time.monotonic()


class MotionManager:
    """Reference-style movement manager with additive offsets and 100Hz loop."""

    def __init__(self, current_robot, camera_worker: Any = None) -> None:
        """Initialize queues, state and synchronization primitives."""
        self.current_robot = current_robot
        self.camera_worker = camera_worker
        self._now = time.monotonic

        self.state = MovementState(last_activity_time=self._now())
        neutral_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.state.last_primary_pose = (neutral_pose, (0.0, 0.0), 0.0)

        self.move_queue: deque[Move] = deque()
        self.idle_inactivity_delay = 0.3
        self.target_frequency = CONTROL_LOOP_FREQUENCY_HZ
        self.target_period = 1.0 / self.target_frequency

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._is_listening = False
        self._breathing_active = False
        self._last_commanded_pose: FullBodyPose = clone_full_body_pose(self.state.last_primary_pose or (neutral_pose, (0.0, 0.0), 0.0))
        self._listening_antennas: Tuple[float, float] = self._last_commanded_pose[1]
        self._antenna_unfreeze_blend = 1.0
        self._antenna_blend_duration = 0.4
        self._last_listening_blend_time = self._now()

        self._command_queue: "Queue[Tuple[str, Any]]" = Queue()

        self._speech_offsets_lock = threading.Lock()
        self._pending_speech_offsets: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._speech_offsets_dirty = False

        self._shared_state_lock = threading.Lock()
        self._shared_last_activity_time = self.state.last_activity_time
        self._shared_is_listening = self._is_listening

        self._speech_animation_stop = threading.Event()
        self._speech_animation_thread: threading.Thread | None = None
        self._sway_rt = SwayRollRT()

    def start(self) -> None:
        """Start the movement worker loop thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop movement loop, speech animation and reset to neutral pose."""
        self.stop_speaking()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        try:
            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            self.current_robot.goto_target(head=neutral_head_pose, antennas=[0.0, 0.0], duration=1.5, body_yaw=0.0)
        except Exception as exc:
            logger.debug("Failed to reset neutral pose: %s", exc)

    def set_state(self, state: State) -> None:
        """Map bridge state to movement listening mode and activity updates."""
        self.mark_activity()
        self.set_listening(state in {State.LISTENING, State.THINKING})

    def queue_move(self, move: Move) -> None:
        """Queue a primary move for execution."""
        self._command_queue.put(("queue_move", move))

    def clear_move_queue(self) -> None:
        """Clear active and pending primary moves."""
        self._command_queue.put(("clear_queue", None))

    def set_speech_offsets(self, offsets: Tuple[float, float, float, float, float, float]) -> None:
        """Update secondary speech offsets in a thread-safe way."""
        with self._speech_offsets_lock:
            self._pending_speech_offsets = offsets
            self._speech_offsets_dirty = True

    def mark_activity(self) -> None:
        """Signal user/system activity to idle logic."""
        self._command_queue.put(("mark_activity", None))

    def set_listening(self, listening: bool) -> None:
        """Enable or disable listening mode with antenna freeze behavior."""
        self._command_queue.put(("set_listening", listening))

    def enable_head_tracking(self, enabled: bool) -> None:
        """Enable or disable camera-based head tracking."""
        if self.camera_worker is not None:
            self.camera_worker.set_head_tracking_enabled(enabled)

    def start_speaking(self) -> None:
        """Start generic speaking activity flag for idle suppression."""
        self.mark_activity()

    def stop_speaking(self) -> None:
        """Stop any running speech animation and reset speech offsets."""
        self._speech_animation_stop.set()
        if self._speech_animation_thread is not None:
            self._speech_animation_thread.join(timeout=1.0)
            self._speech_animation_thread = None
        self._speech_animation_stop.clear()
        self._sway_rt.reset()
        self.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def animate_speech_from_wav(self, wav_path: str) -> None:
        """Animate speech offsets from a WAV file using SwayRollRT."""
        self.stop_speaking()

        def _run() -> None:
            try:
                with wave.open(wav_path, "rb") as wf:
                    sr = wf.getframerate()
                    channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    if sampwidth != 2:
                        return
                    chunk_frames = int(sr * 0.2)
                    while not self._speech_animation_stop.is_set():
                        data = wf.readframes(chunk_frames)
                        if not data:
                            break
                        pcm = np.frombuffer(data, dtype=np.int16)
                        if channels > 1:
                            pcm = pcm.reshape(-1, channels).T
                        results = self._sway_rt.feed(pcm, sr)
                        for r in results:
                            if self._speech_animation_stop.is_set():
                                break
                            offsets = (
                                r["x_mm"] / 1000.0,
                                r["y_mm"] / 1000.0,
                                r["z_mm"] / 1000.0,
                                r["roll_rad"],
                                r["pitch_rad"],
                                r["yaw_rad"],
                            )
                            self.set_speech_offsets(offsets)
                            time.sleep(HOP_MS / 1000.0)
            finally:
                self.set_speech_offsets((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        self._speech_animation_thread = threading.Thread(target=_run, daemon=True)
        self._speech_animation_thread.start()

    def _poll_signals(self, current_time: float) -> None:
        """Apply queued commands and pending speech offsets."""
        self._apply_pending_offsets()
        while True:
            try:
                command, payload = self._command_queue.get_nowait()
            except Empty:
                break
            self._handle_command(command, payload, current_time)

    def _apply_pending_offsets(self) -> None:
        """Consume staged speech offsets atomically."""
        speech_offsets = None
        with self._speech_offsets_lock:
            if self._speech_offsets_dirty:
                speech_offsets = self._pending_speech_offsets
                self._speech_offsets_dirty = False
        if speech_offsets is not None:
            self.state.speech_offsets = speech_offsets
            self.state.update_activity()

    def _handle_command(self, command: str, payload: Any, current_time: float) -> None:
        """Handle one worker command."""
        if command == "queue_move" and isinstance(payload, Move):
            self.move_queue.append(payload)
            self.state.update_activity()
        elif command == "clear_queue":
            self.move_queue.clear()
            self.state.current_move = None
            self.state.move_start_time = None
            self._breathing_active = False
        elif command == "mark_activity":
            self.state.update_activity()
        elif command == "set_listening":
            desired = bool(payload)
            if self._is_listening != desired:
                self._is_listening = desired
                self._last_listening_blend_time = current_time
                if desired:
                    self._listening_antennas = (
                        float(self._last_commanded_pose[1][0]),
                        float(self._last_commanded_pose[1][1]),
                    )
                    self._antenna_unfreeze_blend = 0.0
                self.state.update_activity()

    def _publish_shared_state(self) -> None:
        """Publish idle-related state snapshot for thread-safe reads."""
        with self._shared_state_lock:
            self._shared_last_activity_time = self.state.last_activity_time
            self._shared_is_listening = self._is_listening

    def _manage_move_queue(self, current_time: float) -> None:
        """Advance primary move queue and handle completion."""
        if self.state.current_move is None or (
            self.state.move_start_time is not None
            and current_time - self.state.move_start_time >= self.state.current_move.duration
        ):
            self.state.current_move = None
            self.state.move_start_time = None
            if self.move_queue:
                self.state.current_move = self.move_queue.popleft()
                self.state.move_start_time = current_time
                self._breathing_active = isinstance(self.state.current_move, BreathingMove)

    def _manage_breathing(self, current_time: float) -> None:
        """Start breathing move automatically when idle and not listening."""
        if self.state.current_move is None and not self.move_queue and not self._is_listening and not self._breathing_active:
            idle_for = current_time - self.state.last_activity_time
            if idle_for >= self.idle_inactivity_delay:
                try:
                    _, current_antennas = self.current_robot.get_current_joint_positions()
                    current_head_pose = self.current_robot.get_current_head_pose()
                    self._breathing_active = True
                    self.state.update_activity()
                    self.move_queue.append(
                        BreathingMove(
                            interpolation_start_pose=current_head_pose,
                            interpolation_start_antennas=current_antennas,
                            interpolation_duration=1.0,
                        )
                    )
                except Exception as exc:
                    self._breathing_active = False
                    logger.debug("Failed to start breathing: %s", exc)

    def _get_primary_pose(self, current_time: float) -> FullBodyPose:
        """Return current primary pose from move evaluation or last pose."""
        if self.state.current_move is not None and self.state.move_start_time is not None:
            move_time = current_time - self.state.move_start_time
            head, antennas, body_yaw = self.state.current_move.evaluate(move_time)
            if head is None:
                head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            if antennas is None:
                antennas = np.array([0.0, 0.0])
            if body_yaw is None:
                body_yaw = 0.0
            primary = (head.copy(), (float(antennas[0]), float(antennas[1])), float(body_yaw))
            self.state.last_primary_pose = clone_full_body_pose(primary)
            return primary

        if self.state.last_primary_pose is not None:
            return clone_full_body_pose(self.state.last_primary_pose)

        neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        primary = (neutral_head_pose, (0.0, 0.0), 0.0)
        self.state.last_primary_pose = clone_full_body_pose(primary)
        return primary

    def _get_secondary_pose(self) -> FullBodyPose:
        """Compose secondary pose from speech and camera offsets."""
        secondary_offsets = [
            self.state.speech_offsets[0] + self.state.face_tracking_offsets[0],
            self.state.speech_offsets[1] + self.state.face_tracking_offsets[1],
            self.state.speech_offsets[2] + self.state.face_tracking_offsets[2],
            self.state.speech_offsets[3] + self.state.face_tracking_offsets[3],
            self.state.speech_offsets[4] + self.state.face_tracking_offsets[4],
            self.state.speech_offsets[5] + self.state.face_tracking_offsets[5],
        ]
        secondary_head_pose = create_head_pose(
            x=secondary_offsets[0],
            y=secondary_offsets[1],
            z=secondary_offsets[2],
            roll=secondary_offsets[3],
            pitch=secondary_offsets[4],
            yaw=secondary_offsets[5],
            degrees=False,
            mm=False,
        )
        return (secondary_head_pose, (0.0, 0.0), 0.0)

    def _calculate_blended_antennas(self, target_antennas: Tuple[float, float]) -> Tuple[float, float]:
        """Freeze antennas while listening, then blend back smoothly."""
        now = self._now()
        if self._is_listening:
            self._antenna_unfreeze_blend = 0.0
            return self._listening_antennas

        dt = max(0.0, now - self._last_listening_blend_time)
        self._last_listening_blend_time = now
        if self._antenna_blend_duration <= 0:
            self._antenna_unfreeze_blend = 1.0
        else:
            self._antenna_unfreeze_blend = min(1.0, self._antenna_unfreeze_blend + dt / self._antenna_blend_duration)

        blend = self._antenna_unfreeze_blend
        antennas_cmd = (
            self._listening_antennas[0] * (1.0 - blend) + target_antennas[0] * blend,
            self._listening_antennas[1] * (1.0 - blend) + target_antennas[1] * blend,
        )
        if blend >= 1.0:
            self._listening_antennas = (float(target_antennas[0]), float(target_antennas[1]))
        return antennas_cmd

    def _update_face_tracking(self) -> None:
        """Pull face tracking offsets from camera worker."""
        if self.camera_worker is None:
            self.state.face_tracking_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return
        self.state.face_tracking_offsets = self.camera_worker.get_face_tracking_offsets()

    def _issue_control_command(self, head: NDArray[np.float64], antennas: Tuple[float, float], body_yaw: float) -> None:
        """Send fused pose to robot using a single set_target call."""
        try:
            self.current_robot.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
            self._last_commanded_pose = clone_full_body_pose((head, antennas, body_yaw))
        except Exception as exc:
            logger.debug("Failed to set robot target: %s", exc)

    def get_status(self) -> Dict[str, Any]:
        """Return movement status snapshot for diagnostics."""
        return {
            "queue_size": len(self.move_queue),
            "is_listening": self._is_listening,
            "breathing_active": self._breathing_active,
        }

    def working_loop(self) -> None:
        """Run main movement loop at target frequency."""
        prev_loop_start = self._now()
        while not self._stop_event.is_set():
            loop_start = self._now()
            _ = loop_start - prev_loop_start
            prev_loop_start = loop_start

            self._poll_signals(loop_start)
            self._manage_move_queue(loop_start)
            self._manage_breathing(loop_start)
            self._update_face_tracking()

            primary = self._get_primary_pose(loop_start)
            secondary = self._get_secondary_pose()
            head, antennas, body_yaw = combine_full_body(primary, secondary)
            antennas_cmd = self._calculate_blended_antennas(antennas)
            self._issue_control_command(head, antennas_cmd, body_yaw)

            self._publish_shared_state()
            computation_time = self._now() - loop_start
            sleep_time = max(0.0, self.target_period - computation_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
