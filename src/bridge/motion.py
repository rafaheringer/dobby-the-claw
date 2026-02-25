import logging
import math
import threading
import time
from typing import Optional

from reachy_mini.utils import create_head_pose

from bridge.camera_worker import CameraWorker
from bridge.state_machine import State


class MotionManager:
    def __init__(
        self,
        reachy_mini,
        camera_worker: Optional[CameraWorker] = None,
        loop_hz: float = 10.0,
    ) -> None:
        self._reachy_mini = reachy_mini
        self._camera_worker = camera_worker
        self._loop_hz = loop_hz
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._state = State.IDLE
        self._speaking = False
        self._head_tracking_enabled = True
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is None:
            return
        self._thread.join(timeout=2.0)
        self._thread = None

    def set_state(self, state: State) -> None:
        with self._lock:
            self._state = state

    def start_speaking(self) -> None:
        with self._lock:
            self._speaking = True

    def stop_speaking(self) -> None:
        with self._lock:
            self._speaking = False

    def enable_head_tracking(self, enabled: bool) -> None:
        with self._lock:
            self._head_tracking_enabled = enabled

    def _run(self) -> None:
        interval = 1.0 / max(self._loop_hz, 1.0)
        start_time = time.time()
        while not self._stop.is_set():
            now = time.time()
            elapsed = now - start_time

            with self._lock:
                state = self._state
                speaking = self._speaking
                tracking = self._head_tracking_enabled

            try:
                if tracking and state in {State.IDLE, State.LISTENING}:
                    if self._apply_head_tracking():
                        self._apply_speaking_gesture(elapsed, speaking)
                        time.sleep(interval)
                        continue

                self._apply_state_pose(state, elapsed, speaking)
            except Exception as exc:
                logging.debug("Motion update failed: %s", exc)

            time.sleep(interval)

    def _apply_head_tracking(self) -> bool:
        if self._camera_worker is None:
            return False
        face = self._camera_worker.get_face_center()
        if face is None:
            return False
        u, v = face
        pose = self._reachy_mini.look_at_image(u, v, duration=0.0, perform_movement=False)
        self._reachy_mini.set_target_head_pose(pose)
        return True

    def _apply_state_pose(self, state: State, elapsed: float, speaking: bool) -> None:
        yaw = 0.0
        pitch = 0.0
        roll = 0.0

        if state == State.IDLE:
            pitch = -4.0 + 2.0 * math.sin(elapsed * 0.7)
            roll = 1.5 * math.sin(elapsed * 0.5)
            yaw = 2.0 * math.sin(elapsed * 0.3)
        elif state == State.LISTENING:
            pitch = -10.0
            roll = 0.5 * math.sin(elapsed * 0.6)
        elif state == State.THINKING:
            yaw = 4.0 * math.sin(elapsed * 1.2)
            pitch = -6.0 + 1.0 * math.sin(elapsed * 1.0)
        elif state == State.CONFIRMING:
            pitch = -8.0
            roll = -3.0
        elif state == State.ERROR:
            pitch = -12.0
            roll = 4.0

        if speaking:
            pitch += 2.0 * math.sin(elapsed * 2.5)
            roll += 1.0 * math.sin(elapsed * 2.0)

        head_pose = create_head_pose(roll=roll, pitch=pitch, yaw=yaw, degrees=True)
        self._reachy_mini.set_target_head_pose(head_pose)
        self._apply_speaking_gesture(elapsed, speaking)

    def _apply_speaking_gesture(self, elapsed: float, speaking: bool) -> None:
        if not speaking:
            return
        amplitude = 0.2
        value = amplitude * math.sin(elapsed * 6.0)
        self._reachy_mini.set_target_antenna_joint_positions([value, -value])
