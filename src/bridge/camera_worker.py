"""Camera worker with face-tracking offsets.

This module follows the same control idea used by
`reachy_mini_conversation_app`: continuously fetch camera frames,
estimate a face target, derive pose offsets with `look_at_image`,
and smoothly interpolate back to neutral when tracking is lost.
"""

import logging
import threading
import time
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini.utils.interpolation import linear_pose_interpolation

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None


logger = logging.getLogger(__name__)


class CameraWorker:
    """Thread-safe camera worker with frame buffering and face tracking."""

    def __init__(self, reachy_mini, head_tracker: Any = None) -> None:
        """Initialize camera worker dependencies and tracking state."""
        self.reachy_mini = reachy_mini
        self.head_tracker = head_tracker

        self.latest_frame: NDArray[np.uint8] | None = None
        self.frame_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.is_head_tracking_enabled = True
        self.face_tracking_offsets: List[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.face_tracking_lock = threading.Lock()

        self.last_face_detected_time: float | None = None
        self.interpolation_start_time: float | None = None
        self.interpolation_start_pose: NDArray[np.float64] | None = None
        self.face_lost_delay = 2.0
        self.interpolation_duration = 1.0
        self.previous_head_tracking_state = self.is_head_tracking_enabled

        self._fallback_face_detector = None
        if cv2 is not None:
            cascade_root = getattr(cv2, "data", None)
            if cascade_root is not None:
                cascade = cascade_root.haarcascades + "haarcascade_frontalface_default.xml"
            else:
                cascade = "haarcascade_frontalface_default.xml"
            self._fallback_face_detector = cv2.CascadeClassifier(cascade)

    def get_latest_frame(self) -> NDArray[np.uint8] | None:
        """Return the latest BGR frame copy in a thread-safe way."""
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_face_tracking_offsets(self) -> Tuple[float, float, float, float, float, float]:
        """Return current tracking offsets in meters/radians."""
        with self.face_tracking_lock:
            offsets = self.face_tracking_offsets
            return (offsets[0], offsets[1], offsets[2], offsets[3], offsets[4], offsets[5])

    def set_head_tracking_enabled(self, enabled: bool) -> None:
        """Enable or disable head tracking updates."""
        self.is_head_tracking_enabled = enabled
        logger.info("Head tracking %s", "enabled" if enabled else "disabled")

    def start(self) -> None:
        """Start the camera worker thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        logger.debug("Camera worker started")

    def stop(self) -> None:
        """Stop the camera worker thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        logger.debug("Camera worker stopped")

    def working_loop(self) -> None:
        """Run camera polling and tracking-offset updates."""
        logger.debug("Starting camera working loop")
        neutral_pose = np.eye(4)
        self.previous_head_tracking_state = self.is_head_tracking_enabled

        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                frame = self.reachy_mini.media.get_frame()

                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame

                    if self.previous_head_tracking_state and not self.is_head_tracking_enabled:
                        self.last_face_detected_time = current_time
                        self.interpolation_start_time = None
                        self.interpolation_start_pose = None

                    self.previous_head_tracking_state = self.is_head_tracking_enabled

                    if self.is_head_tracking_enabled:
                        eye_center = self._get_eye_center(frame)
                        if eye_center is not None:
                            self.last_face_detected_time = current_time
                            self.interpolation_start_time = None

                            h, w, _ = frame.shape
                            eye_center_norm = (eye_center + 1) / 2
                            eye_center_pixels = [eye_center_norm[0] * w, eye_center_norm[1] * h]

                            target_pose = self.reachy_mini.look_at_image(
                                eye_center_pixels[0],
                                eye_center_pixels[1],
                                duration=0.0,
                                perform_movement=False,
                            )

                            translation = target_pose[:3, 3] * 0.6
                            rotation = R.from_matrix(target_pose[:3, :3]).as_euler("xyz", degrees=False) * 0.6

                            with self.face_tracking_lock:
                                self.face_tracking_offsets = [
                                    float(translation[0]),
                                    float(translation[1]),
                                    float(translation[2]),
                                    float(rotation[0]),
                                    float(rotation[1]),
                                    float(rotation[2]),
                                ]

                    if self.last_face_detected_time is not None:
                        time_since_face_lost = current_time - self.last_face_detected_time
                        if time_since_face_lost >= self.face_lost_delay:
                            if self.interpolation_start_time is None:
                                self.interpolation_start_time = current_time
                                with self.face_tracking_lock:
                                    current_translation = self.face_tracking_offsets[:3]
                                    current_rotation_euler = self.face_tracking_offsets[3:]
                                    pose_matrix = np.eye(4, dtype=np.float64)
                                    pose_matrix[:3, 3] = current_translation
                                    pose_matrix[:3, :3] = R.from_euler("xyz", current_rotation_euler).as_matrix()
                                    self.interpolation_start_pose = pose_matrix

                            elapsed = current_time - self.interpolation_start_time
                            t = min(1.0, elapsed / self.interpolation_duration)
                            if self.interpolation_start_pose is None:
                                interpolated_pose = neutral_pose
                            else:
                                interpolated_pose = linear_pose_interpolation(self.interpolation_start_pose, neutral_pose, t)
                            translation = interpolated_pose[:3, 3]
                            rotation = R.from_matrix(interpolated_pose[:3, :3]).as_euler("xyz", degrees=False)
                            with self.face_tracking_lock:
                                self.face_tracking_offsets = [
                                    float(translation[0]),
                                    float(translation[1]),
                                    float(translation[2]),
                                    float(rotation[0]),
                                    float(rotation[1]),
                                    float(rotation[2]),
                                ]

                            if t >= 1.0:
                                self.last_face_detected_time = None
                                self.interpolation_start_time = None
                                self.interpolation_start_pose = None

                time.sleep(0.04)
            except Exception as exc:
                logger.error("Camera worker error: %s", exc)
                time.sleep(0.1)

        logger.debug("Camera worker thread exited")

    def _get_eye_center(self, frame: NDArray[np.uint8]) -> Optional[np.ndarray]:
        """Return eye center in normalized coordinates [-1, 1]."""
        if self.head_tracker is not None:
            eye_center, _ = self.head_tracker.get_head_position(frame)
            return eye_center

        if cv2 is None or self._fallback_face_detector is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._fallback_face_detector.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        center_x = x + (w / 2.0)
        center_y = y + (h / 2.0)
        h_img, w_img = frame.shape[:2]
        return np.array([(center_x / w_img) * 2.0 - 1.0, (center_y / h_img) * 2.0 - 1.0], dtype=np.float32)
