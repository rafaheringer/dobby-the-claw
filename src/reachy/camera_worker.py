"""Camera worker with face-tracking offsets.

Faces come from the SDK's YuNet detector, the one the daemon's own head tracking uses, and the
head aims at the nose like it does. YuNet's eye points are too coarse for head tilt, so the tilt
Dobby imitates comes from MediaPipe Face Mesh run on a square crop around the followed face. Unlike the daemon, which follows the largest face, the target
is whoever is talking or else the nearest person, with person boxes reaching people too far for
the face detector. The head eases toward the target and back to neutral when it is lost.
"""

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

from reachy_mini.utils.interpolation import linear_pose_interpolation
from reachy_mini.vision.look_at import DEFAULT_HEAD_TO_CAMERA_TRANSFORM
from reachy.face_recognizer import FaceRecognizer
from reachy.finger_antenna_controller import FingerAntennaController
from reachy.head_roll_controller import HeadRollController
from reachy.presence import FOCAL_OVER_WIDTH, PET_STRONG_SCORE, Box, GazeSelector, PersonTracker, Track, bearing_to_image_x

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    from reachy_mini.media.audio_doa import AudioDoA as _AudioDoA
except ImportError:  # pragma: no cover - hardware optional
    _AudioDoA = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

_MISSING_FRAME_LOG_INTERVAL_S = 1.0
_ID_WINDOW = 6  # recent face looks kept per tracked person
_ID_VOTES = 2  # agreeing looks needed to name a person
_ID_RETRY_S = 1.5  # between looks at someone not yet named
_ID_RECHECK_S = 12.0  # between looks at someone already named
_FACE_MIN_AREA = 0.003  # fraction of the frame a face must cover to be picked up when following nobody
_FACE_KEEP_RAD = 0.25  # a face this close to the followed one is the same person
_FACE_KEEP_S = 1.5  # how long the followed face is remembered through missed detections


@dataclass(frozen=True)
class FaceSighting:
    """One detected face: nose in [-1, 1] image coordinates, box (x, y, w, h) in pixels, frame fraction covered."""

    center: Tuple[float, float]
    box: Tuple[float, float, float, float]
    area: float


def _open_respeaker() -> Any:
    from reachy_mini.media.audio_control_utils import init_respeaker_usb

    return init_respeaker_usb()


_MIC_GAIN_CHECK_S = 5.0


def ensure_fixed_mic_gain(respeaker: Any, lock: threading.Lock, gain: float) -> bool:
    """Keep the mic array AGC off at `gain`; True when something had to be changed.

    The AGC drops the gain within ~100 ms of a hand touching the array (and never recovers), and
    raises it up to 64x in a quiet room, so a TV or footsteps reach the petting level.
    """
    with lock:
        agc_on = any(respeaker.read("PP_AGCONOFF")[1:])  # status byte, then the int32 bytes
        current = float(respeaker.read("PP_AGCGAIN")[0])
        if not agc_on and abs(current - gain) < 0.01:
            return False
        if agc_on:
            respeaker.write("PP_AGCONOFF", [0])
        respeaker.write("PP_AGCGAIN", [float(gain)])
    logger.info("Mic gain fixed at %.1f (AGC was %s, gain %.3f)", gain, "on" if agc_on else "off", current)
    return True


def apply_fixed_mic_gain(gain: float) -> threading.Thread:
    """Fix the mic array gain on a background thread through a short-lived device handle."""

    def _apply() -> None:
        handle = _open_respeaker()
        if handle is None:
            logger.warning("Fixed mic gain not applied: mic array control unavailable")
            return
        try:
            ensure_fixed_mic_gain(handle, threading.Lock(), gain)
        except Exception as exc:
            logger.warning("Fixed mic gain not applied: %s", exc)
        finally:
            handle.close()

    thread = threading.Thread(target=_apply, name="mic-fixed-gain", daemon=True)
    thread.start()
    return thread


class CameraWorker:
    """Thread-safe camera worker with frame buffering and face tracking."""

    def __init__(
        self,
        reachy_mini,
        face_detector: Any = None,
        tilt_tracker: Any = None,
        tracking_max_width: int = 640,
        tracking_fps: float = 15.0,
        antenna_finger_tracking_enabled: bool = True,
        antenna_finger_max_angle_deg: float = 28.0,
        face_recognizer: Optional[FaceRecognizer] = None,
        audio_doa_enabled: bool = True,
        person_detector: Any = None,
        mic_fixed_gain: float = 0.0,
    ) -> None:
        """Initialize camera worker dependencies and tracking state."""
        self.reachy_mini = reachy_mini
        self.face_detector = face_detector
        self.tilt_tracker = tilt_tracker

        self.latest_frame: NDArray[np.uint8] | None = None
        self.frame_lock = threading.Lock()
        self._latest_frame_head: NDArray[np.float64] = np.eye(3)  # head rotation when it was captured

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

        self._last_missing_frame_log_ts = 0.0
        self._missing_frame_since_ts: float | None = None

        self._face_detector_missing_logged = False
        self._followed_face: Optional[Tuple[float, Tuple[float, float]]] = None  # (when, world direction)
        self._roll_controller = HeadRollController()

        self._look_at_translation_gain = 0.5
        # Each sighting becomes a direction in the world (the head pose when its frame was captured
        # plus the angle from the image center), so late sightings, like the ~4 fps person boxes,
        # still point to the right place. The head moves a fraction of the way there per frame.
        self._center_gain = 0.35
        self._center_deadband_rad = 0.03
        self._head_bias: Tuple[float, float] = (0.0, 0.0)  # real head minus commanded (pitch, yaw)
        self._head_bias_alpha = 0.05
        self._tracking_yaw_max_rad = 1.20
        self._tracking_pitch_max_rad = 0.45
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
        self._doa_latest: Optional[tuple[float, float, bool]] = None  # (time, angle, speech)
        self._audio_doa_enabled = audio_doa_enabled
        self._audio_doa_init_attempted = False
        self._doa_target_yaw_rad: float | None = None
        self._doa_active_until: float = 0.0
        self._doa_lock = threading.Lock()  # one USB conversation with the mic array at a time
        self._doa_speech: deque[tuple[float, float]] = deque(maxlen=80)  # (time, angle) while speech is heard
        self._doa_thread: threading.Thread | None = None
        self._mic_fixed_gain = mic_fixed_gain  # mic array gain kept with its AGC off; 0 leaves the AGC

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
        self._gaze_head: NDArray[np.float64] = np.eye(3)  # head rotation when the gaze box's frame was captured
        self._person_update_ts = 0.0
        self._person_frame: Optional[NDArray[np.uint8]] = None  # the frame the tracks' boxes belong to
        self._person_aspect = 16 / 9  # frame width/height, to place heads inside normalized boxes
        self._person_period_s = 0.25
        self._person_thread: threading.Thread | None = None
        self._speaker_set_ts = 0.0

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
        result = None
        if people > 1 and speaking_box is not None:
            seen = self._face_recognizer.identify_person(frame, self._box_px(frame, speaking_box))
            result = (seen.name, seen.score) if seen is not None else None
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

    def _doa_loop(self) -> None:
        """Sample the mic array's direction of arrival and remember the angles heard while someone speaks.

        One reading when the transcript arrives usually comes after the speech is over, so the angle must
        be collected while the person talks. Started late: opening the device before the video pipeline
        runs can disturb it.
        """
        if self._stop_event.wait(3.0):
            return
        try:
            doa = _AudioDoA()  # type: ignore[misc]
        except Exception as exc:
            logger.warning("AudioDoA not available, speaker direction disabled: %s", exc)
            return
        if not getattr(doa, "available", True):
            logger.warning("No ReSpeaker microphone array found, speaker direction disabled")
            return
        self._audio_doa = doa
        logger.info("AudioDoA sampling started")
        next_gain_check = 0.0
        while not self._stop_event.wait(0.12):
            respeaker = getattr(doa, "respeaker", None)
            if self._mic_fixed_gain > 0 and respeaker is not None and time.monotonic() >= next_gain_check:
                next_gain_check = time.monotonic() + _MIC_GAIN_CHECK_S
                try:
                    ensure_fixed_mic_gain(respeaker, self._doa_lock, self._mic_fixed_gain)
                except Exception as exc:
                    logger.debug("Mic gain check failed: %s", exc)
            try:
                with self._doa_lock:
                    reading = doa.get_DoA()
            except Exception as exc:
                logger.debug("DOA read failed: %s", exc)
                self._stop_event.wait(1.0)
                continue
            if reading is None:
                continue
            self._doa_latest = (time.monotonic(), float(reading[0]), bool(reading[1]))
            if reading[1]:
                self._doa_speech.append((time.time(), float(reading[0])))

    def latest_doa(self, max_age_s: float = 1.0) -> Optional[tuple[float, bool]]:
        """The newest direction-of-arrival reading (angle in radians, speech heard), if recent."""
        latest = self._doa_latest
        if latest is None or time.monotonic() - latest[0] > max_age_s:
            return None
        return latest[1], latest[2]

    def _recent_speech_angle(self, window_s: float = 4.0) -> Optional[float]:
        """Median arrival angle of the speech heard in the last `window_s` seconds, or None."""
        cutoff = time.time() - window_s
        angles = [angle for ts, angle in list(self._doa_speech) if ts >= cutoff]
        return float(np.median(angles)) if len(angles) >= 3 else None

    def _activate_doa_steering(self, duration_s: float = 2.5) -> None:
        """Turn toward the sound source: temporary yaw steering from the direction heard while talking."""
        try:
            angle_rad = self._recent_speech_angle()
            speech_detected = angle_rad is not None
            if angle_rad is None:  # no speech sampled lately: take one reading as a weaker hint
                with self._doa_lock:
                    reading = self._audio_doa.get_DoA()  # type: ignore[union-attr]
                if reading is None:
                    return
                angle_rad = float(reading[0])
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
                frame_head = self._latest_frame_head
            if frame is not None:
                try:
                    self._update_persons(frame, started, frame_head)
                except Exception as exc:
                    logger.debug("Person update failed: %s", exc)
            self._stop_event.wait(max(0.02, self._person_period_s - (time.time() - started)))

    def _update_persons(self, frame: NDArray[np.uint8], now: float, frame_head: NDArray[np.float64]) -> None:
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
            self._gaze_head = frame_head
            self._person_update_ts = now
            self._person_frame = frame
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
        """Pick one visible track whose face look is due: unnamed first, named ones re-checked now and then."""
        if self._face_recognizer is None or not self._face_recognizer.available:
            return None

        def interval(track: Track) -> float:
            if track.name is not None:
                return _ID_RECHECK_S  # still the same person? swaps happen when people cross or leave
            return _ID_RETRY_S if track.id_attempts < 8 else 6.0

        due = [t for t in tracks if t.last_seen == now and now - t.last_id_attempt >= interval(t)]
        if not due:
            return None
        track = min(due, key=lambda t: (t.name is not None, t.id_attempts, t is not target, -t.box.height))
        track.id_attempts += 1
        track.last_id_attempt = now
        return track.id, track.box

    @staticmethod
    def _box_px(frame: NDArray[np.uint8], box: Box) -> tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        return int(box.x0 * w), int(box.y0 * h), int(box.x1 * w), int(box.y1 * h)

    def _identify_track(self, frame: NDArray[np.uint8], track_id: int, box: Box) -> None:
        """Look at one person's face and fold the result into that track's identity."""
        seen = self._face_recognizer.identify_person(frame, self._box_px(frame, box))  # type: ignore[union-attr]
        if seen is None:
            return  # face not visible or too poor to judge: no evidence either way
        with self._person_lock:
            tracks = self._person_tracker.tracks
            track = next((t for t in tracks if t.id == track_id), None)
            if track is None:
                return
            track.face_seen = True
            track.observations = (track.observations + [(seen.name, seen.score)])[-_ID_WINDOW:]
            self._settle_identity(track, tracks)

    def _settle_identity(self, track: Track, tracks: list[Track]) -> None:
        """Name a track only when several looks agree; drop the name when looks stop agreeing."""
        recent = track.observations
        if track.name is not None:
            last = recent[-2:]
            if len(last) == 2 and all(name != track.name for name, _ in last):
                logger.info("Person %d no longer matches %s; identifying again", track.id, track.name)
                track.name, track.name_score = None, 0.0
                track.observations = list(last)
                recent = track.observations
            else:
                agreeing = [score for name, score in recent if name == track.name]
                if agreeing:
                    track.name_score = sum(agreeing) / len(agreeing)
                return

        votes: dict[str, list[float]] = {}
        for name, score in recent:
            if name != FaceRecognizer.UNKNOWN:
                votes.setdefault(name, []).append(score)
        if not votes:
            return
        name, scores = max(votes.items(), key=lambda item: (len(item[1]), sum(item[1])))
        rivals = [len(other) for other_name, other in votes.items() if other_name != name]
        if len(scores) < _ID_VOTES or (rivals and max(rivals) >= len(scores)):
            return
        score = sum(scores) / len(scores)
        for other in tracks:
            if other is not track and other.name == name:
                if other.name_score >= score:
                    return  # the same face can't be in two places; keep the better match
                other.name, other.name_score = None, 0.0
        track.name, track.name_score = name, score
        logger.info("Person %d identified as %s (similarity=%.2f, %d looks)", track.id, name, score, len(scores))

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
        """Tight face crops of one person over a few frames, following the track as they move."""
        crops = []
        for index in range(count):
            if index:
                time.sleep(interval_s)
            with self._person_lock:  # the box and the frame it was found in must come from the same moment
                frame = self._person_frame
                box = next((t.box for t in self._person_tracker.tracks if t.id == track_id), None)
            if frame is None or box is None:
                continue
            crop = self._face_recognizer.face_crop_for_person(frame, self._box_px(frame, box))  # type: ignore[union-attr]
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
        self,
        faces: list[FaceSighting],
        frame_head: NDArray[np.float64],
        now: float,
        width: int,
        height: int,
    ) -> Optional[Tuple[np.ndarray, Optional[FaceSighting], Tuple[float, float]]]:
        """Pick what to look at: the chosen person's face, their estimated head when no face is seen, or a face.

        Returns the target's image center, its face (None for a head estimated from a person box) and its
        (pitch, yaw) direction in the world.
        """
        speech_x = self._speech_image_x(now)
        if speech_x is not None and not -0.15 <= speech_x <= 1.15:
            return None  # someone is talking well outside the view: let DOA turn the head
        with self._person_lock:
            box = self._gaze_target if now - self._person_update_ts < 1.0 else None
            box_head = self._gaze_head
        sightings = [
            (face, self._world_direction(np.asarray(face.center), frame_head, width, height)) for face in faces
        ]
        if self._followed_face is not None and now - self._followed_face[0] > _FACE_KEEP_S:
            self._followed_face = None

        chosen = None
        if box is not None:
            # The box comes from an older frame, so compare directions rather than image positions.
            box_pitch, box_yaw = self._world_direction(
                np.array([2.0 * box.center_x - 1.0, 2.0 * box.center_y - 1.0]), box_head, width, height
            )
            half_yaw = box.width / 2.0 / FOCAL_OVER_WIDTH + 0.05
            half_pitch = box.height * height / width / 2.0 / FOCAL_OVER_WIDTH + 0.05
            inside = [
                (face, direction)
                for face, direction in sightings
                if abs(direction[1] - box_yaw) <= half_yaw and abs(direction[0] - box_pitch) <= half_pitch
            ]
            if inside:
                chosen = max(inside, key=lambda item: item[0].area)
            else:
                hx, hy = box.head_point(self._person_aspect)
                center = np.array([2.0 * hx - 1.0, 2.0 * hy - 1.0], dtype=np.float32)
                self._followed_face = None
                return (center, None, self._world_direction(center, box_head, width, height))
        elif sightings:
            if self._followed_face is not None:
                followed = self._followed_face[1]
                nearest = min(sightings, key=lambda item: math.dist(item[1], followed))
                if math.dist(nearest[1], followed) <= _FACE_KEEP_RAD:
                    chosen = nearest
            if chosen is None:
                largest = max(sightings, key=lambda item: item[0].area)
                if self._followed_face is None and largest[0].area >= _FACE_MIN_AREA:
                    chosen = largest
        if chosen is None:
            return None
        face, direction = chosen
        self._followed_face = (now, direction)
        return (np.asarray(face.center, dtype=np.float32), face, direction)

    def _world_direction(
        self, center: np.ndarray, head: NDArray[np.float64], width: int, height: int
    ) -> Tuple[float, float]:
        """(pitch, yaw) the head must take to face an image point ([-1, 1] coordinates) seen with rotation `head`.

        The camera turns with the head, so a tilted head also tilts the image: the point's ray is taken
        through the full head rotation (roll included) into the world before reading pitch and yaw.
        """
        u = (float(center[0]) + 1.0) / 2.0 * width
        v = (float(center[1]) + 1.0) / 2.0 * height
        x_n, y_n = self._normalized_ray(u, v, width, height)
        ray = head @ (DEFAULT_HEAD_TO_CAMERA_TRANSFORM[:3, :3] @ np.array([x_n, y_n, 1.0]))
        return float(np.arctan2(-ray[2], np.hypot(ray[0], ray[1]))), float(np.arctan2(ray[1], ray[0]))

    @staticmethod
    def _facing(head: NDArray[np.float64]) -> Tuple[float, float]:
        """(pitch, yaw) of the direction the head faces, whatever its roll."""
        forward = head[:, 0]
        return float(np.arctan2(-forward[2], np.hypot(forward[0], forward[1]))), float(np.arctan2(forward[1], forward[0]))

    def _update_head_bias(self, commanded_pitch: float, commanded_yaw: float) -> None:
        """Learn, slowly and only while the head is nearly still, how far the real head sits from the command.

        Looking far up, the head mechanics settle up to ~0.1 rad off the commanded pose.
        """
        measured_pitch, measured_yaw = self._facing(self._measured_head_rotation())
        alpha = self._head_bias_alpha
        self._head_bias = (
            (1.0 - alpha) * self._head_bias[0] + alpha * float(np.clip(measured_pitch - commanded_pitch, -0.2, 0.2)),
            (1.0 - alpha) * self._head_bias[1] + alpha * float(np.clip(measured_yaw - commanded_yaw, -0.2, 0.2)),
        )

    def _measured_head_rotation(self) -> NDArray[np.float64]:
        """Head rotation as the robot reports it, or the tracking offsets when it can't tell."""
        try:
            return np.asarray(self.reachy_mini.get_current_head_pose(), dtype=np.float64)[:3, :3].copy()
        except Exception:
            with self.face_tracking_lock:
                return R.from_euler("xyz", self.face_tracking_offsets[3:]).as_matrix()

    def start(self) -> None:
        """Start the camera worker thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.working_loop, daemon=True)
        self._thread.start()
        if self._person_detector is not None:
            self._person_thread = threading.Thread(target=self._person_loop, daemon=True)
            self._person_thread.start()
        if self._audio_doa_enabled and _AudioDoA is not None and self._doa_thread is None:
            self._doa_thread = threading.Thread(target=self._doa_loop, daemon=True, name="doa")
            self._doa_thread.start()
        logger.debug("Camera worker started")

    def stop(self) -> None:
        """Stop the camera worker thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        if self._person_thread is not None:
            self._person_thread.join(timeout=3.0)
            self._person_thread = None
        if self._doa_thread is not None:
            self._doa_thread.join(timeout=2.0)
            self._doa_thread = None
        with self._person_lock:  # people seen before sleeping are stale after waking up
            self._person_tracker = PersonTracker()
            self._pet_trackers = self._new_pet_trackers()
            self._gaze_target = None
            self._person_update_ts = 0.0
        self._finger_antenna_controller.close()
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
                else:
                    self._missing_frame_since_ts = None
                    frame_head = self._measured_head_rotation()
                    with self.frame_lock:
                        self.latest_frame = frame
                        self._latest_frame_head = frame_head

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
                        self._roll_controller.reset()

                    self.previous_head_tracking_state = self.is_head_tracking_enabled

                    if self.is_head_tracking_enabled and should_run_tracking:
                        h, w, _ = frame.shape
                        tracking_target = self._choose_tracking_target(
                            self._detect_faces(tracking_frame), frame_head, current_time, w, h
                        )
                        eye_center = None if tracking_target is None else np.clip(tracking_target[0], -0.99, 0.99)
                        target_face = None if tracking_target is None else tracking_target[1]
                        seen_tilt = None if target_face is None else self._face_tilt(tracking_frame, target_face)
                        # The image tilts with Dobby's own head roll; add it back to get the person's real tilt.
                        head_tilt_rad = 0.0 if seen_tilt is None else seen_tilt + float(R.from_matrix(frame_head).as_euler("xyz")[0])
                        self._last_eye_center = eye_center
                        self._last_head_tilt_rad = head_tilt_rad
                        if eye_center is not None:
                            self.last_face_detected_time = current_time
                            self.interpolation_start_time = None

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
                            target_pitch, target_yaw = tracking_target[2]
                            with self.face_tracking_lock:
                                tracking_pitch = self.face_tracking_offsets[4]
                                tracking_yaw = self.face_tracking_offsets[5]
                            target_pitch -= self._head_bias[0]
                            target_yaw -= self._head_bias[1]
                            if abs(target_yaw - tracking_yaw) >= self._center_deadband_rad:
                                tracking_yaw += self._center_gain * (target_yaw - tracking_yaw)
                            if abs(target_pitch - tracking_pitch) >= self._center_deadband_rad:
                                tracking_pitch += self._center_gain * (target_pitch - tracking_pitch)
                            if max(abs(target_pitch - tracking_pitch), abs(target_yaw - tracking_yaw)) < 0.05:
                                self._update_head_bias(tracking_pitch, tracking_yaw)
                            tracking_yaw = float(
                                np.clip(tracking_yaw, -self._tracking_yaw_max_rad, self._tracking_yaw_max_rad)
                            )
                            tracking_pitch = float(
                                np.clip(tracking_pitch, -self._tracking_pitch_max_rad, self._tracking_pitch_max_rad)
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
                                    tracking_pitch,
                                    tracking_yaw,
                                ]
                            self._maybe_identify_face_background(frame, current_time)
                        else:
                            if (
                                self._doa_target_yaw_rad is not None
                                and current_time < self._doa_active_until
                            ):
                                # DOA active: steer head toward sound source
                                alpha = 0.12
                                with self.face_tracking_lock:
                                    self.face_tracking_offsets[5] = float(
                                        (1.0 - alpha) * self.face_tracking_offsets[5]
                                        + alpha * self._doa_target_yaw_rad
                                    )
                                # Suppress neutral interpolation while DOA is active
                                self.last_face_detected_time = current_time
                            elif self._doa_target_yaw_rad is not None:
                                self._doa_target_yaw_rad = None

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
                                self._head_bias = (0.0, 0.0)  # it depends on the pose; neutral has none
                                self.last_face_detected_time = None
                                self.interpolation_start_time = None
                                self.interpolation_start_pose = None


                time.sleep(self._camera_loop_period_s)
            except Exception as exc:
                logger.error("Camera worker error: %s", exc)
                time.sleep(0.1)

        logger.debug("Camera worker thread exited")

    def _read_camera_frame(self, current_time: float) -> NDArray[np.uint8] | None:
        """Read a frame from the Reachy SDK."""
        return self.reachy_mini.media.get_frame()

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

    def _detect_faces(self, frame: NDArray[np.uint8]) -> list[FaceSighting]:
        """Detect every face in the frame, aimed at the nose like the daemon's head tracking."""
        if self.face_detector is None:
            if not self._face_detector_missing_logged:
                logger.warning("Face detector not available; only person boxes can be followed")
                self._face_detector_missing_logged = True
            return []

        try:
            faces = self.face_detector.detect(frame)
        except Exception as exc:
            logger.debug("Face detection failed: %s", exc)
            return []

        height, width = frame.shape[:2]
        sightings = []
        for face in faces:
            nose_x, nose_y = face.nose
            sightings.append(
                FaceSighting(
                    center=(nose_x / max(width - 1, 1) * 2 - 1, nose_y / max(height - 1, 1) * 2 - 1),
                    box=tuple(float(v) for v in face.bbox),
                    area=face.bbox[2] * face.bbox[3] / float(width * height),
                )
            )
        return sightings

    def _face_tilt(self, frame: NDArray[np.uint8], face: FaceSighting) -> Optional[float]:
        """Head tilt (rad) of one face in the image, from Face Mesh on a square crop (None when it can't tell)."""
        if self.tilt_tracker is None:
            return None
        height, width = frame.shape[:2]
        x, y, w, h = face.box
        side = 1.8 * max(w, h)
        cx, cy = x + w / 2.0, y + h / 2.0
        x0, y0 = int(max(0.0, cx - side / 2)), int(max(0.0, cy - side / 2))
        x1, y1 = int(min(width, cx + side / 2)), int(min(height, cy + side / 2))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        try:
            _, roll = self.tilt_tracker.get_head_position(frame[y0:y1, x0:x1])
        except Exception as exc:
            logger.debug("Face tilt failed: %s", exc)
            return None
        return None if roll is None else float(roll)

    def _update_antenna_finger_control(self, frame: NDArray[np.uint8], current_time: float) -> None:
        """Update antenna offsets from finger controller output."""
        state = self._finger_antenna_controller.update(frame, current_time)
        with self._hand_control_lock:
            self._hand_control_active = bool(state.active)
            self._hand_control_offsets = (float(state.offsets[0]), float(state.offsets[1]))
            self._last_index_finger_count = int(state.finger_count)

    def _normalized_ray(self, u: float, v: float, width: int, height: int) -> Tuple[float, float]:
        """Undistorted normalized camera coordinates (x right, y down, at depth 1) of pixel (u, v)."""
        camera = getattr(getattr(self.reachy_mini, "media", None), "camera", None)
        K = getattr(camera, "K", None)
        if cv2 is not None and K is not None:
            resolution = getattr(camera, "resolution", None) or (width, height)
            point = np.array([[[u * resolution[0] / width, v * resolution[1] / height]]], dtype=np.float32)
            x_n, y_n = cv2.undistortPoints(point, K, getattr(camera, "D", None))[0, 0]
        else:
            focal = FOCAL_OVER_WIDTH * width
            x_n, y_n = (u - width / 2.0) / focal, (v - height / 2.0) / focal
        return float(x_n), float(y_n)

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

    def _maybe_emit_missing_frame_log(self, current_time: float) -> None:
        """Log camera starvation periodically."""
        if (current_time - self._last_missing_frame_log_ts) < _MISSING_FRAME_LOG_INTERVAL_S:
            return
        elapsed = max(0.0, current_time - self._missing_frame_since_ts) if self._missing_frame_since_ts else 0.0
        if elapsed < 3.0:
            return
        self._last_missing_frame_log_ts = current_time
        logger.warning("Camera: no frames from Reachy SDK for %.0fs — check daemon video pipeline", elapsed)

    def render_debug_frame(
        self, show_people: bool = True, show_tracking: bool = True, show_voice: bool = True
    ) -> NDArray[np.uint8] | None:
        """Latest frame with detection overlays for the debug UI, or None while there is no image."""
        vis = self.get_latest_frame()
        if vis is None or cv2 is None:
            return None
        font, line = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
        h, w = vis.shape[:2]
        center = (w // 2, h // 2)
        cv2.drawMarker(vis, center, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
        now = time.time()

        if show_people:
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
                cv2.putText(vis, f"{label} {score:.2f}", (p0[0] + 4, p0[1] + 22), font, 0.6, color, 2, line)
            for box, name, is_target, verified, score, face_seen in people:
                # orange = gaze target, blue = person, gray = unverified (no face yet, weak score)
                color = (0, 200, 255) if is_target else (200, 120, 0) if verified else (140, 140, 140)
                p0, p1 = (int(box.x0 * w), int(box.y0 * h)), (int(box.x1 * w), int(box.y1 * h))
                cv2.rectangle(vis, p0, p1, color, 2)
                label = f"{name or '?'} {score:.2f}{' face' if face_seen else ''}"
                cv2.putText(vis, label, (p0[0] + 4, p0[1] + 22), font, 0.6, color, 2, line)

        if show_voice:
            speech_x = self._speech_image_x(now)
            if speech_x is not None:
                x = int(min(max(speech_x, 0.0), 1.0) * w)
                cv2.line(vis, (x, 0), (x, h), (255, 0, 255), 2)
                cv2.putText(vis, "voz", (min(x + 6, w - 50), h - 12), font, 0.6, (255, 0, 255), 2, line)

        if show_tracking:
            if self._last_target_pixels is not None:
                target = (int(self._last_target_pixels[0]), int(self._last_target_pixels[1]))
                cv2.circle(vis, target, 8, (0, 255, 0), 2)
                cv2.line(vis, center, target, (0, 255, 0), 1)
            with self.face_tracking_lock:
                offsets = tuple(self.face_tracking_offsets)
            face_detected = self.last_face_detected_time is not None and (now - self.last_face_detected_time) <= self.face_lost_delay
            status_color = (0, 220, 0) if face_detected else (0, 0, 220)
            hand_active, hand_offsets, finger_count = self.get_antenna_finger_control()
            rows = [
                (f"tracking={self.is_head_tracking_enabled} face={face_detected} lock={self._roll_controller.is_neutral_locked}", status_color, 0.6, 2),
                (f"xyz=({offsets[0]:+.3f},{offsets[1]:+.3f},{offsets[2]:+.3f})", (255, 255, 0), 0.55, 1),
                (f"rpy=({offsets[3]:+.3f},{offsets[4]:+.3f},{offsets[5]:+.3f})", (255, 255, 0), 0.55, 1),
                (
                    f"tilt_raw={self._last_head_tilt_rad:+.3f} bias={self._last_head_tilt_bias_rad:+.3f} "
                    f"filt={self._last_filtered_tilt_rad:+.3f} roll={self._last_imitated_roll_rad:+.3f}",
                    (255, 255, 0), 0.55, 1,
                ),
                (
                    f"fingers={finger_count} antenna=({hand_offsets[0]:+.3f},{hand_offsets[1]:+.3f})",
                    (0, 255, 0) if hand_active else (150, 150, 150), 0.55, 1,
                ),
            ]
            for index, (text, color, scale, thickness) in enumerate(rows):
                cv2.putText(vis, text, (10, 24 + 22 * index), font, scale, color, thickness, line)
            self._draw_tilt_indicator(
                vis, detected_tilt_rad=self._last_head_tilt_rad, imitated_roll_rad=self._last_imitated_roll_rad
            )
        return vis
