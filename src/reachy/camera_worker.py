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
from dataclasses import replace
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini.utils.interpolation import linear_pose_interpolation
from reachy.face_recognizer import FaceRecognizer
from reachy.finger_antenna_controller import FingerAntennaController
from reachy.head_roll_controller import HeadRollController
from reachy.presence import PET_STRONG_SCORE, Box, GazeSelector, PersonTracker, Track, bearing_to_image_x

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
        tracking_max_width: int = 640,
        tracking_fps: float = 15.0,
        antenna_finger_tracking_enabled: bool = True,
        antenna_finger_max_angle_deg: float = 28.0,
        face_recognizer: Optional[FaceRecognizer] = None,
        audio_doa_enabled: bool = True,
        person_detector: Any = None,
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
        # Opt-in: with the robot attached, probing the PC webcam whenever a robot frame is late
        # makes the two cameras alternate.
        self._allow_local_camera_fallback = bool(
            platform.system() == "Windows"
            and self._debug_visual_window
            and cv2 is not None
            and os.getenv("VISION_LOCAL_CAMERA_FALLBACK", "0").strip().lower() in ("1", "true", "yes", "on")
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
        self._tracking_max_width = max(0, int(tracking_max_width))
        self._tracking_period_s = 0.0 if tracking_fps <= 0 else (1.0 / float(tracking_fps))
        self._last_tracking_update_ts = 0.0

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
        self._speaker_hold_until: float = 0.0  # per-turn decisions win over background face ID

        # Audio Direction-of-Arrival for steering head toward speaker
        # Lazy init: AudioDoA shares GStreamer resources with the camera pipeline.
        # Initializing it in __init__ (before the camera loop starts) can corrupt
        # the video GStreamer pipeline. We defer until first use.
        self._audio_doa: Optional[object] = None
        self._audio_doa_enabled = audio_doa_enabled
        self._audio_doa_init_attempted = False
        self._doa_target_yaw_rad: float | None = None
        self._doa_active_until: float = 0.0

        # Person detection: follow people too far away for the face tracker and choose whom to look at.
        self._person_detector = person_detector if getattr(person_detector, "available", False) else None
        self._person_lock = threading.Lock()
        self._person_tracker = PersonTracker()
        self._pet_trackers = self._new_pet_trackers()
        self._last_person_log_ts = 0.0
        self._speaking_track: tuple[int, float] | None = None  # (track id, when) of the person the voice came from
        self._speaker_face_certain = True  # False when the last speaker ID had to guess among several faces
        self._gaze = GazeSelector()
        self._gaze_target: Optional[Box] = None
        self._person_update_ts = 0.0
        self._person_aspect = 16 / 9  # frame width/height, to place heads inside normalized boxes
        self._person_period_s = 0.25
        self._person_thread: threading.Thread | None = None
        self._speaker_set_ts = 0.0

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

    def set_current_speaker(self, name: str | None, hold_s: float = 0.0) -> None:
        """Set the current speaker, keeping background face ID from overriding it for `hold_s`."""
        with self._speaker_lock:
            self._current_speaker = name
            self._speaker_hold_until = time.time() + hold_s
            self._speaker_set_ts = time.time()

    def identify_current_speaker(self, wait_s: float = 0.8, update: bool = True) -> tuple[str | None, float]:
        """Activate DOA steering, wait for head to turn, then identify speaker.

        Returns (name, similarity): name is FaceRecognizer.UNKNOWN for unrecognized
        faces (similarity 0.0 when no face is in view), or None if face recognition
        is not available. With `update` the result becomes the current speaker.
        """
        if self._face_recognizer is None or not self._face_recognizer.available:
            return None, 0.0

        asked_at = time.time()
        if self._audio_doa is not None:
            self._activate_doa_steering(duration_s=max(wait_s + 1.0, 2.5))

        time.sleep(max(0.1, float(wait_s)))

        frame = self.get_latest_frame()
        if frame is None:
            return None, 0.0

        # With several people the largest face may not be the one talking: use the person the voice comes from.
        people, speaking_box = 0, None
        with self._person_lock:
            if time.time() - self._person_update_ts < 2.0:
                verified = self._person_tracker.verified()
                people = len(verified)
                if self._speaking_track is not None and self._speaking_track[1] >= asked_at:
                    speaking_box = next((t.box for t in verified if t.id == self._speaking_track[0]), None)
        crop = self._head_crop(frame, speaking_box) if people > 1 and speaking_box is not None else None
        result = self._face_recognizer.identify_face(crop) if crop is not None else None
        if result is not None:
            name, confidence = result
        else:
            name, confidence = self._face_recognizer.identify(frame)
        self._speaker_face_certain = people <= 1 or result is not None
        logger.info(
            "Speaker ID: %s (similarity=%.2f%s)",
            name, confidence, "" if self._speaker_face_certain else f", {people} people, unsure who talks",
        )

        if update:
            with self._speaker_lock:
                self._current_speaker = name
        return name, confidence

    def _activate_doa_steering(self, duration_s: float = 2.5) -> None:
        """Read DOA and activate temporary yaw steering toward the sound source."""
        if not self._audio_doa_init_attempted:
            self._audio_doa_init_attempted = True
            if self._audio_doa_enabled and _AudioDoA is not None:
                try:
                    self._audio_doa = _AudioDoA()
                    logger.info("AudioDoA initialized (lazy)")
                except Exception as exc:
                    logger.debug("AudioDoA not available: %s", exc)
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
                if time.time() < self._speaker_hold_until:
                    return
                if self._current_speaker != name:
                    logger.debug(
                        "Speaker updated: %s → %s (similarity=%.2f)",
                        self._current_speaker, name, confidence,
                    )
                self._current_speaker = name if name != FaceRecognizer.UNKNOWN else None
        except Exception as exc:
            logger.debug("Background face identification failed: %s", exc)

    def get_presence(self) -> list[tuple[int, Optional[str], float]]:
        """Return the people in view as (track id, recognized name or None, confirmed timestamp)."""
        if self._person_detector is None:
            return []
        with self._person_lock:
            if time.time() - self._person_update_ts > 2.0:
                return []
            return [(t.id, t.name, t.confirmed_at or t.first_seen) for t in self._person_tracker.verified()]

    @staticmethod
    def _new_pet_trackers() -> dict[str, PersonTracker]:
        """One tracker per pet kind; ids are offset so the gaze never confuses a pet with a person."""
        return {"cat": PersonTracker(PET_STRONG_SCORE, 10_000), "dog": PersonTracker(PET_STRONG_SCORE, 20_000)}

    def get_pets(self) -> list[str]:
        """Return the label of each pet in view ("cat", "dog")."""
        if self._person_detector is None:
            return []
        with self._person_lock:
            if time.time() - self._person_update_ts > 2.0:
                return []
            return [label for label, tracker in self._pet_trackers.items() for _ in tracker.verified()]

    def _person_loop(self) -> None:
        """Detect people a few times per second, keep tracks, pick the gaze target and name tracks by face."""
        while not self._stop_event.is_set():
            started = time.time()
            with self.frame_lock:
                frame = self.latest_frame
            if frame is not None:
                try:
                    self._update_persons(frame, started)
                except Exception as exc:
                    logger.debug("Person update failed: %s", exc)
            self._stop_event.wait(max(0.02, self._person_period_s - (time.time() - started)))

    def _update_persons(self, frame: NDArray[np.uint8], now: float) -> None:
        detections = self._person_detector.detect(frame)  # type: ignore[union-attr]
        self._person_aspect = frame.shape[1] / float(frame.shape[0])
        people = [d for d in detections if d.label == "person"]
        pets = {label: [d for d in detections if d.label == label] for label in self._pet_trackers}
        speech_x = self._speech_image_x(now)
        with self._speaker_lock:
            speaker = self._current_speaker if now - self._speaker_set_ts < 20.0 else None
        with self._person_lock:
            confirmed = self._person_tracker.update([d.box for d in people], now, [d.score for d in people])
            verified = [t for t in confirmed if self._person_tracker.is_verified(t)]
            pet_tracks = []
            for label, tracker in self._pet_trackers.items():
                tracker.update([d.box for d in pets[label]], now, [d.score for d in pets[label]])
                pet_tracks += tracker.verified()
            # Nobody around: keep an eye on the cat.
            if speech_x is not None and verified:
                nearest = min(verified, key=lambda t: abs(t.box.center_x - speech_x))
                if abs(nearest.box.center_x - speech_x) < 0.35:
                    self._speaking_track = (nearest.id, now)
            target = self._gaze.choose(verified or pet_tracks, now, speaker_name=speaker, speech_x=speech_x)
            self._gaze_target = None if target is None else target.box
            self._person_update_ts = now
            to_identify = self._next_track_to_identify(confirmed, now, target)
        if to_identify is not None:
            self._identify_track(frame, *to_identify)
        if now - self._last_person_log_ts >= 2.0 and logger.isEnabledFor(logging.DEBUG):
            self._last_person_log_ts = now
            with self._person_lock:
                tracks = [
                    f"#{t.id}{'' if self._person_tracker.is_verified(t) else '?'} {t.name or '-'} "
                    f"score={t.score:.2f} face={t.face_seen} h={t.box.height:.2f}"
                    for t in self._person_tracker.tracks
                ]
            logger.debug(
                "Persons raw=%s tracks=[%s] pets=%s",
                [round(d.score, 2) for d in people],
                "; ".join(tracks),
                [f"{d.label}:{d.score:.2f}" for group in pets.values() for d in group],
            )

    def _speech_image_x(self, now: float) -> Optional[float]:
        """Image x the current voice direction maps to, while DOA steering is active."""
        doa_yaw = self._doa_target_yaw_rad
        if doa_yaw is None or now >= self._doa_active_until:
            return None
        with self.face_tracking_lock:
            head_yaw = self.face_tracking_offsets[5]
        return bearing_to_image_x(doa_yaw, head_yaw)

    def _next_track_to_identify(self, tracks: list[Track], now: float, target: Optional[Track]) -> Optional[tuple[int, Box]]:
        """Pick one visible unnamed track whose face-ID retry is due (fewest attempts, then the gaze target)."""
        if self._face_recognizer is None or not self._face_recognizer.available:
            return None
        due = [
            t for t in tracks
            if t.name is None and t.last_seen == now
            and now - t.last_id_attempt >= (2.5 if t.id_attempts < 6 else 10.0)
        ]
        if not due:
            return None
        track = min(due, key=lambda t: (t.id_attempts, t is not target, -t.box.height))
        track.id_attempts += 1
        track.last_id_attempt = now
        return track.id, track.box

    def _identify_track(self, frame: NDArray[np.uint8], track_id: int, box: Box) -> None:
        """Run face recognition on the head region of one person box."""
        crop = self._head_crop(frame, box)
        if crop is None:
            return
        result = self._face_recognizer.identify_face(crop)  # type: ignore[union-attr]
        if result is None:
            return
        name, score = result
        with self._person_lock:
            tracks = self._person_tracker.tracks
            track = next((t for t in tracks if t.id == track_id), None)
            if track is None:
                return
            track.face_seen = True
            if name == FaceRecognizer.UNKNOWN:
                return
            for other in tracks:
                if other is not track and other.name == name:
                    if other.name_score >= score:
                        return  # the same face can't be in two places; keep the better match
                    other.name = None
            track.name, track.name_score = name, score
        logger.info("Person %d identified as %s (similarity=%.2f)", track_id, name, score)

    @staticmethod
    def _head_crop(frame: NDArray[np.uint8], box: Box) -> Optional[NDArray[np.uint8]]:
        """Head region of one person box, or None when too small to hold a face."""
        h, w = frame.shape[:2]
        region = box.head_box(w / float(h))
        x0, x1 = int(max(0.0, region.x0) * w), int(min(1.0, region.x1) * w)
        y0, y1 = int(max(0.0, region.y0) * h), int(min(1.0, region.y1) * h)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return None
        return frame[y0:y1, x0:x1]

    @property
    def speaker_face_certain(self) -> bool:
        """False when the last speaker ID saw several people and could not tell which one was talking."""
        return self._speaker_face_certain

    def get_enrollment_view(self) -> tuple[list[Track], Optional[int], Optional[NDArray[np.uint8]]]:
        """People in view (copies, left to right), the track id of whoever is talking, and the latest frame."""
        frame = self.get_latest_frame()
        now = time.time()
        with self._speaker_lock:
            speaker = self._current_speaker if now - self._speaker_set_ts < 30.0 else None
        with self._person_lock:
            if self._person_detector is None or now - self._person_update_ts > 2.0:
                return [], None, frame
            people = sorted((replace(t) for t in self._person_tracker.verified()), key=lambda t: t.box.center_x)
            speaker_id = None
            if speaker and speaker != FaceRecognizer.UNKNOWN:
                named = [t.id for t in people if t.name == speaker]
                speaker_id = named[0] if len(named) == 1 else None
            if speaker_id is None and self._speaking_track is not None and now - self._speaking_track[1] < 30.0:
                speaker_id = self._speaking_track[0]
        return people, speaker_id, frame

    def capture_track_faces(self, track_id: int, count: int = 4, interval_s: float = 0.35) -> list[NDArray[np.uint8]]:
        """Head crops of one person over a few frames, following the track as they move."""
        crops = []
        for index in range(count):
            if index:
                time.sleep(interval_s)
            frame = self.get_latest_frame()
            with self._person_lock:
                box = next((t.box for t in self._person_tracker.tracks if t.id == track_id), None)
            if frame is None or box is None:
                continue
            crop = self._head_crop(frame, box)
            if crop is not None:
                crops.append(crop.copy())
        return crops

    def name_track(self, track_id: int, name: str) -> None:
        """Label a track right after enrolling it, so presence and gaze know who it is."""
        with self._person_lock:
            for track in self._person_tracker.tracks:
                if track.id == track_id:
                    track.name, track.name_score, track.face_seen = name, 1.0, True
                elif track.name == name:
                    track.name = None

    def _choose_tracking_target(
        self, face_target: Optional[Tuple[np.ndarray, float]], now: float
    ) -> Optional[Tuple[np.ndarray, float]]:
        """Combine the face tracker with person boxes: faces are precise, person boxes reach farther."""
        speech_x = self._speech_image_x(now)
        if speech_x is not None and not -0.15 <= speech_x <= 1.15:
            return None  # someone is talking well outside the view: let DOA turn the head
        with self._person_lock:
            box = self._gaze_target if now - self._person_update_ts < 1.0 else None
        if box is None:
            return face_target
        if face_target is not None:
            fx, fy = (np.asarray(face_target[0], dtype=np.float32) + 1.0) / 2.0
            if box.x0 - 0.05 <= fx <= box.x1 + 0.05 and box.y0 - 0.05 <= fy <= box.y1 + 0.05:
                return face_target
        hx, hy = box.head_point(self._person_aspect)
        return (np.array([2.0 * hx - 1.0, 2.0 * hy - 1.0], dtype=np.float32), 0.0)

    def start(self) -> None:
        """Start the camera worker thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        if self._person_detector is not None:
            self._person_thread = threading.Thread(target=self._person_loop, daemon=True)
            self._person_thread.start()
        logger.debug("Camera worker started")

    def stop(self) -> None:
        """Stop the camera worker thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if self._person_thread is not None:
            self._person_thread.join(timeout=3.0)
            self._person_thread = None
        with self._person_lock:  # people seen before sleeping are stale after waking up
            self._person_tracker = PersonTracker()
            self._pet_trackers = self._new_pet_trackers()
            self._gaze_target = None
            self._person_update_ts = 0.0
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

                    tracking_frame = self._prepare_tracking_frame(frame)
                    should_run_tracking = self._should_run_tracking(current_time)

                    if should_run_tracking:
                        self._update_antenna_finger_control(tracking_frame, current_time)

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

                    if self.is_head_tracking_enabled and should_run_tracking:
                        tracking_target = self._choose_tracking_target(
                            self._get_tracking_target(tracking_frame), current_time
                        )
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

    def _prepare_tracking_frame(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Downscale frames for CPU-bound tracking while preserving full-res snapshots."""
        if cv2 is None or self._tracking_max_width <= 0:
            return frame

        height, width = frame.shape[:2]
        if width <= self._tracking_max_width:
            return frame

        scale = self._tracking_max_width / float(width)
        target_size = (self._tracking_max_width, max(1, int(round(height * scale))))
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    def _should_run_tracking(self, current_time: float) -> bool:
        """Return whether CPU-bound tracking should run on this loop iteration."""
        if self._tracking_period_s <= 0.0:
            return True
        if (current_time - self._last_tracking_update_ts) < self._tracking_period_s:
            return False
        self._last_tracking_update_ts = current_time
        return True

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
        """Log camera starvation periodically."""
        if (current_time - self._last_missing_frame_log_ts) < self._debug_log_interval_s:
            return
        elapsed = max(0.0, current_time - self._missing_frame_since_ts) if self._missing_frame_since_ts else 0.0
        if elapsed < 3.0:
            return
        self._last_missing_frame_log_ts = current_time
        if self._using_local_camera_fallback:
            logger.debug(
                "Camera: using local fallback (index=%s)",
                self._local_camera_index,
            )
            return
        logger.warning("Camera: no frames from Reachy SDK for %.0fs — check daemon video pipeline", elapsed)

    def _render_debug_placeholder(self, current_time: float) -> None:
        """Render a placeholder window while waiting for the first camera frame."""
        if not self._debug_visual_window or cv2 is None or self._debug_window_failed:
            return
        try:
            elapsed = 0.0 if self._missing_frame_since_ts is None else max(0.0, current_time - self._missing_frame_since_ts)
            if self.latest_frame is not None and elapsed < 2.0:
                cv2.waitKey(1)  # a late frame is normal: keep the last image instead of flashing the placeholder
                return
            vis = np.zeros((360, 640, 3), dtype=np.uint8)
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

            with self._person_lock:
                people = [
                    (t.box, t.name, t.box == self._gaze_target, self._person_tracker.is_verified(t), t.score, t.face_seen)
                    for t in self._person_tracker.confirmed()
                ]
                pets = [
                    (t.box, label, t.box == self._gaze_target, t.score)
                    for label, tracker in self._pet_trackers.items()
                    for t in tracker.verified()
                ]
            for box, label, is_target, score in pets:
                color = (0, 200, 255) if is_target else (0, 220, 120)
                p0, p1 = (int(box.x0 * w), int(box.y0 * h)), (int(box.x1 * w), int(box.y1 * h))
                cv2.rectangle(vis, p0, p1, color, 2)
                cv2.putText(vis, f"{label} {score:.2f}", (p0[0] + 4, p0[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            for box, name, is_target, verified, score, face_seen in people:
                # orange = gaze target, blue = person, gray = unverified (no face yet, weak score)
                color = (0, 200, 255) if is_target else (200, 120, 0) if verified else (140, 140, 140)
                p0, p1 = (int(box.x0 * w), int(box.y0 * h)), (int(box.x1 * w), int(box.y1 * h))
                cv2.rectangle(vis, p0, p1, color, 2)
                label = f"{name or '?'} {score:.2f}{' face' if face_seen else ''}"
                cv2.putText(vis, label, (p0[0] + 4, p0[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

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
