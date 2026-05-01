"""Camera worker with face-tracking offsets.

This module follows the same control idea used by
`reachy_mini_conversation_app`: continuously fetch camera frames,
estimate a face target, derive pose offsets with `look_at_image`,
and smoothly interpolate back to neutral when tracking is lost.
"""

import logging
import os
import platform
import threading
import time
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini.utils.interpolation import linear_pose_interpolation
from reachy.face_recognizer import FaceRecognizer
from reachy.finger_antenna_controller import FingerAntennaController
from reachy.head_roll_controller import HeadRollController

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from reachy_mini.media.audio_doa import AudioDoA as _AudioDoA
except ImportError:  # pragma: no cover - hardware optional
    _AudioDoA = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class CameraWorker:
    """Thread-safe camera worker with frame buffering and face tracking."""

    def __init__(
        self,
        reachy_mini,
        head_tracker: Any = None,
        debug_visual_window: bool = False,
        debug_log_interval_s: float = 1.0,
        antenna_finger_tracking_enabled: bool = True,
        antenna_finger_max_angle_deg: float = 28.0,
        face_recognizer: Optional[FaceRecognizer] = None,
        audio_doa_enabled: bool = True,
    ) -> None:
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

        self._last_eye_center: Optional[np.ndarray] = None
        self._last_head_tilt_rad = 0.0
        self._last_imitated_roll_rad = 0.0
        self._last_head_tilt_bias_rad = 0.0
        self._last_filtered_tilt_rad = 0.0
        self._last_target_pixels: Optional[Tuple[float, float]] = None
        self._last_frame_size: Tuple[int, int] = (0, 0)

        self._debug_visual_window = bool(debug_visual_window and cv2 is not None)
        self._debug_window_name = "Reachy Vision Debug"
        self._debug_window_failed = False
        self._debug_log_interval_s = max(0.2, float(debug_log_interval_s))
        self._last_debug_log_ts = 0.0
        self._last_missing_frame_log_ts = 0.0
        self._missing_frame_since_ts: float | None = None
        self._allow_local_camera_fallback = bool(
            platform.system() == "Windows" and self._debug_visual_window and cv2 is not None
        )
        self._local_camera_capture = None
        self._local_camera_index: int | None = None
        self._using_local_camera_fallback = False
        self._last_local_camera_probe_ts = 0.0
        self._local_camera_retry_interval_s = 2.0
        self._configure_debug_window_environment()

        self._head_tracker_missing_logged = False
        self._roll_controller = HeadRollController()

        self._look_at_translation_gain = 0.5
        self._yaw_from_center_gain = 0.90
        self._yaw_from_center_deadband = 0.06
        self._yaw_from_center_max_rad = 1.20
        self._yaw_from_center_smoothing_alpha = 0.16
        self._smoothed_yaw_from_center = 0.0
        self._pitch_from_center_gain = 0.22
        self._pitch_from_center_deadband = 0.08
        self._pitch_from_center_max_rad = 0.22
        self._eye_center_deadband = 0.035
        self._eye_center_smoothing_alpha = 0.22
        self._smoothed_eye_center: Optional[np.ndarray] = None
        self._camera_loop_period_s = 0.02

        self._finger_antenna_controller = FingerAntennaController(
            enabled=antenna_finger_tracking_enabled,
            max_angle_deg=antenna_finger_max_angle_deg,
        )
        self._hand_control_lock = threading.Lock()
        self._hand_control_active = False
        self._hand_control_offsets: Tuple[float, float] = (0.0, 0.0)
        self._last_index_finger_count = 0

        # Speaker identification
        self._face_recognizer = face_recognizer
        self._current_speaker: str | None = None
        self._speaker_lock = threading.Lock()
        self._last_identify_ts: float = 0.0
        self._identify_interval_s: float = 10.0

        # Audio Direction-of-Arrival for steering head toward speaker
        self._audio_doa: Optional[object] = None
        self._doa_target_yaw_rad: float | None = None
        self._doa_active_until: float = 0.0
        if audio_doa_enabled and _AudioDoA is not None:
            try:
                self._audio_doa = _AudioDoA()
                logger.info("AudioDoA initialized")
            except Exception as exc:
                logger.debug("AudioDoA not available: %s", exc)

    def _configure_debug_window_environment(self) -> None:
        """Prepare Qt env vars to reduce noisy warnings in OpenCV debug windows."""
        if not self._debug_visual_window:
            return
        if "QT_QPA_FONTDIR" in os.environ:
            return

        font_candidates = (
            "/usr/share/fonts/truetype/dejavu",
            "/usr/share/fonts/dejavu",
            "/usr/share/fonts/truetype/liberation",
        )
        for path in font_candidates:
            if os.path.isdir(path):
                os.environ["QT_QPA_FONTDIR"] = path
                logger.debug("Vision debug: QT_QPA_FONTDIR set to %s", path)
                break

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

    def get_antenna_finger_control(self) -> Tuple[bool, Tuple[float, float], int]:
        """Return current antenna hand-control state: active flag, offsets and finger count."""
        with self._hand_control_lock:
            return (self._hand_control_active, self._hand_control_offsets, self._last_index_finger_count)

    def set_head_tracking_enabled(self, enabled: bool) -> None:
        """Enable or disable head tracking updates."""
        self.is_head_tracking_enabled = enabled
        logger.info("Head tracking %s", "enabled" if enabled else "disabled")

    def get_current_speaker(self) -> str | None:
        """Return the last identified speaker name, or None if unknown."""
        with self._speaker_lock:
            return self._current_speaker

    def identify_current_speaker(self, wait_s: float = 0.8) -> str | None:
        """Activate DOA steering, wait for head to turn, then identify speaker.

        Returns the speaker name (or FaceRecognizer.UNKNOWN for unrecognized
        faces), or None if face recognition is not available.
        """
        if self._face_recognizer is None or not self._face_recognizer.available:
            return None

        if self._audio_doa is not None:
            self._activate_doa_steering(duration_s=max(wait_s + 1.0, 2.5))

        time.sleep(max(0.1, float(wait_s)))

        frame = self.get_latest_frame()
        if frame is None:
            return None

        name, confidence = self._face_recognizer.identify(frame)
        logger.info("Speaker ID: %s (similarity=%.2f)", name, confidence)

        with self._speaker_lock:
            self._current_speaker = name
        return name

    def _activate_doa_steering(self, duration_s: float = 2.5) -> None:
        """Read DOA and activate temporary yaw steering toward the sound source."""
        try:
            angle_rad, speech_detected = self._audio_doa.get_DoA()  # type: ignore[union-attr]
            # Mapping: π/2 = front (yaw=0), 0 = left (+yaw), π = right (-yaw)
            yaw_target = float(np.clip(-(angle_rad - np.pi / 2) * 0.8, -1.2, 1.2))
            self._doa_target_yaw_rad = yaw_target
            self._doa_active_until = time.time() + duration_s
            logger.info(
                "DOA steering: angle=%.2f rad speech=%s → yaw_target=%.2f rad",
                angle_rad, speech_detected, yaw_target,
            )
        except Exception as exc:
            logger.debug("DOA steering failed: %s", exc)

    def _maybe_identify_face_background(self, frame: NDArray, current_time: float) -> None:
        """Spawn background thread to identify face and update current speaker."""
        if self._face_recognizer is None or not self._face_recognizer.available:
            return
        if (current_time - self._last_identify_ts) < self._identify_interval_s:
            return
        self._last_identify_ts = current_time
        frame_copy = frame.copy()
        threading.Thread(
            target=self._background_identify, args=(frame_copy,), daemon=True
        ).start()

    def _background_identify(self, frame: NDArray) -> None:
        """Run face identification and update _current_speaker (background thread)."""
        try:
            name, confidence = self._face_recognizer.identify(frame)  # type: ignore[union-attr]
            with self._speaker_lock:
                if self._current_speaker != name:
                    logger.debug(
                        "Speaker updated: %s → %s (similarity=%.2f)",
                        self._current_speaker, name, confidence,
                    )
                self._current_speaker = name if name != FaceRecognizer.UNKNOWN else None
        except Exception as exc:
            logger.debug("Background face identification failed: %s", exc)

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
        self._close_local_camera_fallback()
        self._finger_antenna_controller.close()
        if self._debug_visual_window and cv2 is not None:
            try:
                cv2.destroyWindow(self._debug_window_name)
            except Exception:
                pass
        logger.debug("Camera worker stopped")

    def get_tracking_debug_snapshot(self) -> dict:
        """Return lightweight face-tracking diagnostics for logging/debug UI."""
        current_time = time.time()
        with self.face_tracking_lock:
            offsets = tuple(self.face_tracking_offsets)
        face_detected_recently = False
        time_since_face_s: Optional[float] = None
        if self.last_face_detected_time is not None:
            time_since_face_s = max(0.0, current_time - self.last_face_detected_time)
            face_detected_recently = time_since_face_s <= self.face_lost_delay
        finger_control_active, antenna_finger_offsets, index_finger_count = self.get_antenna_finger_control()
        return {
            "tracking_enabled": self.is_head_tracking_enabled,
            "face_detected_recently": face_detected_recently,
            "time_since_face_s": time_since_face_s,
            "offsets": offsets,
            "eye_center": None if self._last_eye_center is None else (float(self._last_eye_center[0]), float(self._last_eye_center[1])),
            "head_tilt_rad": float(self._last_head_tilt_rad),
            "head_tilt_bias_rad": float(self._last_head_tilt_bias_rad),
            "imitated_roll_rad": float(self._last_imitated_roll_rad),
            "neutral_lock": bool(self._roll_controller.is_neutral_locked),
            "target_pixels": self._last_target_pixels,
            "frame_size": self._last_frame_size,
            "finger_control_active": bool(finger_control_active),
            "index_finger_count": int(index_finger_count),
            "antenna_finger_offsets": tuple(antenna_finger_offsets),
        }

    def working_loop(self) -> None:
        """Run camera polling and tracking-offset updates."""
        logger.debug("Starting camera working loop")
        neutral_pose = np.eye(4)
        self.previous_head_tracking_state = self.is_head_tracking_enabled

        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                frame = self._read_camera_frame(current_time)

                if frame is None:
                    if self._missing_frame_since_ts is None:
                        self._missing_frame_since_ts = current_time
                    self._maybe_emit_missing_frame_log(current_time)
                    self._render_debug_placeholder(current_time)
                else:
                    self._missing_frame_since_ts = None
                    with self.frame_lock:
                        self.latest_frame = frame

                    self._update_antenna_finger_control(frame, current_time)

                    if self.previous_head_tracking_state and not self.is_head_tracking_enabled:
                        self.last_face_detected_time = current_time
                        self.interpolation_start_time = None
                        self.interpolation_start_pose = None
                        self._last_head_tilt_rad = 0.0
                        self._last_imitated_roll_rad = 0.0
                        self._last_filtered_tilt_rad = 0.0
                        self._last_head_tilt_bias_rad = 0.0
                        self._smoothed_eye_center = None
                        self._smoothed_yaw_from_center = 0.0
                        self._roll_controller.reset()

                    self.previous_head_tracking_state = self.is_head_tracking_enabled

                    if self.is_head_tracking_enabled:
                        tracking_target = self._get_tracking_target(frame)
                        eye_center = None if tracking_target is None else tracking_target[0]
                        head_tilt_rad = 0.0 if tracking_target is None else float(tracking_target[1])
                        eye_center = self._stabilize_eye_center(eye_center)
                        self._last_eye_center = eye_center
                        self._last_head_tilt_rad = head_tilt_rad
                        if eye_center is not None:
                            self.last_face_detected_time = current_time
                            self.interpolation_start_time = None

                            h, w, _ = frame.shape
                            self._last_frame_size = (w, h)
                            eye_center_norm = (eye_center + 1) / 2
                            eye_center_pixels = [eye_center_norm[0] * w, eye_center_norm[1] * h]
                            self._last_target_pixels = (float(eye_center_pixels[0]), float(eye_center_pixels[1]))

                            target_pose = self.reachy_mini.look_at_image(
                                eye_center_pixels[0],
                                eye_center_pixels[1],
                                duration=0.0,
                                perform_movement=False,
                            )

                            translation = target_pose[:3, 3] * self._look_at_translation_gain
                            center_error_x = float(eye_center[0])
                            center_error_y = float(eye_center[1])

                            yaw_from_center = 0.0
                            if abs(center_error_x) >= self._yaw_from_center_deadband:
                                yaw_from_center = -center_error_x * self._yaw_from_center_gain
                            yaw_from_center = float(
                                np.clip(
                                    yaw_from_center,
                                    -self._yaw_from_center_max_rad,
                                    self._yaw_from_center_max_rad,
                                )
                            )
                            yaw_alpha = float(np.clip(self._yaw_from_center_smoothing_alpha, 0.01, 1.0))
                            self._smoothed_yaw_from_center = (
                                (1.0 - yaw_alpha) * self._smoothed_yaw_from_center
                                + yaw_alpha * yaw_from_center
                            )

                            pitch_from_center = 0.0
                            if abs(center_error_y) >= self._pitch_from_center_deadband:
                                pitch_from_center = center_error_y * self._pitch_from_center_gain
                            pitch_from_center = float(
                                np.clip(
                                    pitch_from_center,
                                    -self._pitch_from_center_max_rad,
                                    self._pitch_from_center_max_rad,
                                )
                            )

                            imitated_roll = self._roll_controller.update(head_tilt_rad, now=current_time)
                            self._last_imitated_roll_rad = imitated_roll
                            self._last_filtered_tilt_rad = self._roll_controller.last_filtered_tilt_rad
                            self._last_head_tilt_bias_rad = self._roll_controller.bias_rad

                            with self.face_tracking_lock:
                                self.face_tracking_offsets = [
                                    float(translation[0]),
                                    float(translation[1]),
                                    float(translation[2]),
                                    float(imitated_roll),
                                    float(pitch_from_center),
                                    float(self._smoothed_yaw_from_center),
                                ]
                            self._maybe_identify_face_background(frame, current_time)
                        else:
                            self._smoothed_eye_center = None
                            if (
                                self._doa_target_yaw_rad is not None
                                and current_time < self._doa_active_until
                            ):
                                # DOA active: steer head toward sound source
                                alpha = 0.12
                                self._smoothed_yaw_from_center = (
                                    (1.0 - alpha) * self._smoothed_yaw_from_center
                                    + alpha * self._doa_target_yaw_rad
                                )
                                with self.face_tracking_lock:
                                    self.face_tracking_offsets[5] = float(
                                        self._smoothed_yaw_from_center
                                    )
                                # Suppress neutral interpolation while DOA is active
                                self.last_face_detected_time = current_time
                            else:
                                if self._doa_target_yaw_rad is not None:
                                    self._doa_target_yaw_rad = None
                                self._smoothed_yaw_from_center *= 0.8

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

                    self._maybe_emit_debug_log(current_time)
                    self._render_debug_visual(frame)

                time.sleep(self._camera_loop_period_s)
            except Exception as exc:
                logger.error("Camera worker error: %s", exc)
                time.sleep(0.1)

        logger.debug("Camera worker thread exited")

    def _read_camera_frame(self, current_time: float) -> NDArray[np.uint8] | None:
        """Read a frame from Reachy SDK or Windows local webcam fallback."""
        frame = self.reachy_mini.media.get_frame()
        if frame is not None:
            if self._using_local_camera_fallback:
                logger.info("Reachy SDK camera frames recovered; disabling local camera fallback")
                self._close_local_camera_fallback()
            return frame

        return self._read_local_camera_fallback(current_time)

    def _read_local_camera_fallback(self, current_time: float) -> NDArray[np.uint8] | None:
        """Read from a local Windows webcam when SDK frames are unavailable."""
        if not self._allow_local_camera_fallback or cv2 is None:
            return None

        capture = self._local_camera_capture
        if capture is None:
            if (current_time - self._last_local_camera_probe_ts) < self._local_camera_retry_interval_s:
                return None
            self._last_local_camera_probe_ts = current_time
            capture = self._open_local_camera_fallback()
            if capture is None:
                return None

        ok, frame = capture.read()
        if not ok or frame is None:
            logger.warning("Local camera fallback failed to read a frame; retrying")
            self._close_local_camera_fallback()
            return None

        return np.asarray(frame, dtype=np.uint8)

    def _open_local_camera_fallback(self):
        """Open the first usable Windows webcam for debug fallback."""
        if cv2 is None:
            return None

        backend_candidates = []
        if platform.system() == "Windows":
            backend_candidates.append(cv2.CAP_DSHOW)
        backend_candidates.append(cv2.CAP_ANY)

        for index in (0, 1, 2):
            for backend in backend_candidates:
                capture = cv2.VideoCapture(index, backend)
                if not capture.isOpened():
                    capture.release()
                    continue

                ok, frame = capture.read()
                if ok and frame is not None:
                    self._local_camera_capture = capture
                    self._local_camera_index = index
                    self._using_local_camera_fallback = True
                    logger.warning(
                        "Reachy SDK camera unavailable; using local Windows camera index=%s for debug",
                        index,
                    )
                    return capture

                capture.release()

        logger.warning("Reachy SDK camera unavailable and no local Windows camera fallback was opened")
        return None

    def _close_local_camera_fallback(self) -> None:
        """Release local webcam fallback resources."""
        capture = self._local_camera_capture
        self._local_camera_capture = None
        self._local_camera_index = None
        self._using_local_camera_fallback = False
        if capture is None:
            return
        try:
            capture.release()
        except Exception:
            pass

    def _get_tracking_target(self, frame: NDArray[np.uint8]) -> Optional[Tuple[np.ndarray, float]]:
        """Return face target center and head tilt roll estimate in radians."""
        if self.head_tracker is None:
            if not self._head_tracker_missing_logged:
                logger.warning("Head tracking backend not available; no face target will be produced")
                self._head_tracker_missing_logged = True
            return None

        try:
            result = self.head_tracker.get_head_position(frame)
        except Exception as exc:
            logger.debug("Head tracker failed: %s", exc)
            return None

        if result is None:
            return None

        eye_center = None
        tracker_payload: Any = None
        if isinstance(result, tuple):
            if len(result) >= 1:
                eye_center = result[0]
            if len(result) >= 2:
                tracker_payload = result[1]
        else:
            eye_center = result

        if eye_center is None:
            return None

        tilt = self._extract_tracker_tilt_rad(tracker_payload)
        if tilt is None:
            tilt = 0.0
        return (np.asarray(eye_center, dtype=np.float32), tilt)

    def _extract_tracker_tilt_rad(self, payload: Any) -> Optional[float]:
        """Extract tilt roll from tracker payload using tolerant key parsing."""
        if payload is None:
            return None

        value: Any = None
        if isinstance(payload, dict):
            for key in (
                "roll_rad",
                "head_roll_rad",
                "tilt_rad",
                "roll",
                "head_roll",
                "tilt",
            ):
                if key in payload:
                    value = payload[key]
                    break
        elif isinstance(payload, (float, int, np.floating)):
            value = payload
        elif isinstance(payload, (tuple, list)) and payload:
            first = payload[0]
            if isinstance(first, (float, int, np.floating)):
                value = first

        if value is None:
            return None

        try:
            tilt = float(value)
        except (TypeError, ValueError):
            return None

        if abs(tilt) > np.pi:
            tilt = float(np.deg2rad(tilt))
        return float(np.clip(tilt, -np.deg2rad(30.0), np.deg2rad(30.0)))

    def _update_antenna_finger_control(self, frame: NDArray[np.uint8], current_time: float) -> None:
        """Update antenna offsets from finger controller output."""
        state = self._finger_antenna_controller.update(frame, current_time)
        with self._hand_control_lock:
            self._hand_control_active = bool(state.active)
            self._hand_control_offsets = (float(state.offsets[0]), float(state.offsets[1]))
            self._last_index_finger_count = int(state.finger_count)

    def _stabilize_eye_center(self, eye_center: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Low-pass + deadband stabilization for target center to avoid micro-corrections."""
        if eye_center is None:
            return None

        center = np.asarray(eye_center, dtype=np.float32).reshape(2)
        if self._smoothed_eye_center is None:
            self._smoothed_eye_center = center
            return center

        delta = center - self._smoothed_eye_center
        stabilized = center.copy()
        for index in (0, 1):
            if abs(float(delta[index])) < self._eye_center_deadband:
                stabilized[index] = self._smoothed_eye_center[index]

        alpha = float(np.clip(self._eye_center_smoothing_alpha, 0.01, 1.0))
        self._smoothed_eye_center = ((1.0 - alpha) * self._smoothed_eye_center) + (alpha * stabilized)
        return self._smoothed_eye_center.astype(np.float32)

    def _draw_tilt_indicator(self, vis: NDArray[np.uint8], detected_tilt_rad: float, imitated_roll_rad: float) -> None:
        """Draw compact tilt plot (detected vs applied) in debug view."""
        cv = cv2
        if cv is None:
            return
        h, w = vis.shape[:2]
        cx = w - 110
        cy = 78
        radius = 44
        cv.circle(vis, (cx, cy), radius, (180, 180, 180), 1)

        baseline_left = (cx - radius, cy)
        baseline_right = (cx + radius, cy)
        cv.line(vis, baseline_left, baseline_right, (120, 120, 120), 1)

        def _endpoint(angle_rad: float, color: Tuple[int, int, int], label: str, label_y: int) -> None:
            dx = int(np.cos(-angle_rad) * radius)
            dy = int(np.sin(-angle_rad) * radius)
            end = (cx + dx, cy + dy)
            cv.arrowedLine(vis, (cx, cy), end, color, 2, tipLength=0.2)
            cv.putText(vis, label, (cx - radius, label_y), cv.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv.LINE_AA)

        _endpoint(float(detected_tilt_rad), (0, 255, 255), "det", cy + radius + 16)
        _endpoint(float(imitated_roll_rad), (0, 255, 0), "app", cy + radius + 32)
        cv.putText(vis, "tilt", (cx - 18, cy + radius + 16), cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv.LINE_AA)

    def _maybe_emit_debug_log(self, current_time: float) -> None:
        """Emit periodic compact tracking telemetry for debugging."""
        if (current_time - self._last_debug_log_ts) < self._debug_log_interval_s:
            return
        self._last_debug_log_ts = current_time
        snapshot = self.get_tracking_debug_snapshot()
        offsets = snapshot["offsets"]
        logger.debug(
            "Vision debug enabled=%s face=%s lock=%s since=%.2fs eye=%s tilt=%.3f bias=%.3f imitated_roll=%.3f target=%s offs_xyz=(%.3f,%.3f,%.3f) offs_rpy=(%.3f,%.3f,%.3f) finger_active=%s fingers=%d ant=(%.3f,%.3f)",
            snapshot["tracking_enabled"],
            snapshot["face_detected_recently"],
            snapshot["neutral_lock"],
            snapshot["time_since_face_s"] if snapshot["time_since_face_s"] is not None else -1.0,
            snapshot["eye_center"],
            snapshot["head_tilt_rad"],
            snapshot["head_tilt_bias_rad"],
            snapshot["imitated_roll_rad"],
            snapshot["target_pixels"],
            offsets[0],
            offsets[1],
            offsets[2],
            offsets[3],
            offsets[4],
            offsets[5],
            snapshot["finger_control_active"],
            snapshot["index_finger_count"],
            snapshot["antenna_finger_offsets"][0],
            snapshot["antenna_finger_offsets"][1],
        )

    def _maybe_emit_missing_frame_log(self, current_time: float) -> None:
        """Log camera starvation periodically when debug mode is enabled."""
        if not self._debug_visual_window:
            return
        if (current_time - self._last_missing_frame_log_ts) < self._debug_log_interval_s:
            return
        self._last_missing_frame_log_ts = current_time
        if self._using_local_camera_fallback:
            logger.info(
                "Vision debug window is using local Windows camera fallback (index=%s)",
                self._local_camera_index,
            )
            return
        logger.info("Vision debug window is waiting for camera frames from Reachy SDK")

    def _render_debug_placeholder(self, current_time: float) -> None:
        """Render a placeholder window while waiting for the first camera frame."""
        if not self._debug_visual_window or cv2 is None or self._debug_window_failed:
            return
        try:
            vis = np.zeros((360, 640, 3), dtype=np.uint8)
            elapsed = 0.0 if self._missing_frame_since_ts is None else max(0.0, current_time - self._missing_frame_since_ts)
            cv2.putText(
                vis,
                "Reachy Vision Debug",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                "Waiting for camera frames from Reachy SDK...",
                (24, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 215, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"head_tracking={self.is_head_tracking_enabled} elapsed={elapsed:.1f}s",
                (24, 152),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                "If this stays blank, inspect the remote daemon camera stream.",
                (24, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(self._debug_window_name, vis)
            cv2.waitKey(1)
        except Exception as exc:
            self._debug_window_failed = True
            logger.warning("Vision debug window disabled after failure: %s", exc)

    def _render_debug_visual(self, frame: NDArray[np.uint8]) -> None:
        """Render optional visual overlay to inspect face detection and target mapping."""
        if not self._debug_visual_window or cv2 is None or self._debug_window_failed:
            return
        try:
            vis = frame.copy()
            h, w = vis.shape[:2]
            center = (w // 2, h // 2)
            cv2.drawMarker(vis, center, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)

            if self._last_target_pixels is not None:
                tx, ty = self._last_target_pixels
                target = (int(tx), int(ty))
                cv2.circle(vis, target, 8, (0, 255, 0), 2)
                cv2.line(vis, center, target, (0, 255, 0), 1)

            with self.face_tracking_lock:
                offsets = tuple(self.face_tracking_offsets)
            face_detected = self.last_face_detected_time is not None and (time.time() - self.last_face_detected_time) <= self.face_lost_delay
            status_color = (0, 220, 0) if face_detected else (0, 0, 220)
            frame_source = "local" if self._using_local_camera_fallback else "reachy"
            status_text = (
                f"source={frame_source} tracking={self.is_head_tracking_enabled} "
                f"face={face_detected} lock={self._roll_controller.is_neutral_locked}"
            )
            cv2.putText(vis, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)
            cv2.putText(
                vis,
                f"xyz=({offsets[0]:+.3f},{offsets[1]:+.3f},{offsets[2]:+.3f})",
                (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"rpy=({offsets[3]:+.3f},{offsets[4]:+.3f},{offsets[5]:+.3f})",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"tilt_raw={self._last_head_tilt_rad:+.3f} bias={self._last_head_tilt_bias_rad:+.3f} filt={self._last_filtered_tilt_rad:+.3f} roll={self._last_imitated_roll_rad:+.3f}",
                (10, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            hand_active, hand_offsets, finger_count = self.get_antenna_finger_control()
            hand_color = (0, 255, 0) if hand_active else (150, 150, 150)
            cv2.putText(
                vis,
                f"fingers={finger_count} antenna=({hand_offsets[0]:+.3f},{hand_offsets[1]:+.3f})",
                (10, 114),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                hand_color,
                1,
                cv2.LINE_AA,
            )

            self._draw_tilt_indicator(
                vis,
                detected_tilt_rad=self._last_head_tilt_rad,
                imitated_roll_rad=self._last_imitated_roll_rad,
            )

            cv2.imshow(self._debug_window_name, vis)
            cv2.waitKey(1)
        except Exception as exc:
            self._debug_window_failed = True
            logger.warning("Vision debug window disabled after failure: %s", exc)
