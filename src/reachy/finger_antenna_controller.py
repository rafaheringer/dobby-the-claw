"""Index-finger driven antenna control.

This module isolates MediaPipe Hands processing and converts raised index
finger gestures into left/right antenna target angles.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import mediapipe as mp
    from mediapipe.python.solutions import hands as mp_hands
except ImportError:  # pragma: no cover - optional dependency
    mp = None
    mp_hands = None

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FingerAntennaControl:
    """Finger control state snapshot for antenna commands."""

    active: bool
    offsets: Tuple[float, float]
    finger_count: int


class FingerAntennaController:
    """Converts raised index finger gestures to antenna angles."""

    def __init__(
        self,
        enabled: bool = True,
        max_angle_deg: float = 180.0,
        hold_time_s: float = 0.04,
        smoothing_alpha: float = 1.0,
        deadband_deg: float = 0.0,
    ) -> None:
        """Initialize finger tracking thresholds, smoothing, and backend state."""
        self._enabled = bool(enabled and mp is not None and cv2 is not None)
        self._max_angle_rad = float(np.deg2rad(max_angle_deg))
        self._hold_time_s = float(max(0.0, hold_time_s))
        self._smoothing_alpha = float(np.clip(smoothing_alpha, 0.01, 1.0))
        self._deadband_rad = float(np.deg2rad(max(0.0, deadband_deg)))
        self._max_delta_rad = float(np.deg2rad(14.0))

        self._hands: Any | None = None
        self._last_detected_time: float | None = None
        self._last_offsets: Tuple[float, float] = (0.0, 0.0)
        self._last_finger_count = 0
        self._active = False

        self._init_hands()

    @property
    def enabled(self) -> bool:
        """Return whether the controller is effectively enabled."""
        return self._enabled

    def close(self) -> None:
        """Release underlying hand-tracking resources."""
        if self._hands is None:
            return
        try:
            self._hands.close()
        except Exception:
            pass

    def get_state(self) -> FingerAntennaControl:
        """Return current state snapshot."""
        return FingerAntennaControl(self._active, self._last_offsets, self._last_finger_count)

    def update(self, frame_bgr: NDArray[np.uint8], now_s: float) -> FingerAntennaControl:
        """Update controller from a BGR frame and return latest antenna command."""
        cv = cv2
        if not self._enabled or self._hands is None or cv is None:
            self._active = False
            self._last_offsets = (0.0, 0.0)
            self._last_finger_count = 0
            return self.get_state()

        try:
            flipped = cv.flip(frame_bgr, 1)
            rgb = cv.cvtColor(flipped, cv.COLOR_BGR2RGB)
            results = self._hands.process(rgb)
        except Exception as exc:
            logger.debug("Finger tracker processing failed: %s", exc)
            return self.get_state()

        controls = self._extract_controls(results)
        if controls is not None:
            left_target, right_target, finger_count = controls
            prev_left, prev_right = self._last_offsets
            alpha = self._smoothing_alpha
            next_left = (1.0 - alpha) * prev_left + alpha * left_target
            next_right = (1.0 - alpha) * prev_right + alpha * right_target

            multiturn = self._allow_multiturn(
                [next_left, next_right],
                [prev_left, prev_right],
                self._max_delta_rad,
            )
            next_left = float(multiturn[0])
            next_right = float(multiturn[1])

            if abs(next_left) < self._deadband_rad:
                next_left = 0.0
            if abs(next_right) < self._deadband_rad:
                next_right = 0.0

            self._active = True
            self._last_offsets = (float(next_left), float(next_right))
            self._last_finger_count = int(finger_count)
            self._last_detected_time = now_s
            return self.get_state()

        hold_active = self._last_detected_time is not None and (now_s - self._last_detected_time) <= self._hold_time_s
        if hold_active:
            return self.get_state()

        self._active = False
        self._last_offsets = (0.0, 0.0)
        self._last_finger_count = 0
        return self.get_state()

    def _init_hands(self) -> None:
        """Initialize MediaPipe Hands backend."""
        if not self._enabled:
            if mp is None:
                logger.info("Antenna finger tracking disabled: mediapipe not available")
            elif cv2 is None:
                logger.info("Antenna finger tracking disabled: opencv not available")
            return

        try:
            self._hands = mp_hands.Hands(
                static_image_mode=False,
                model_complexity=0,
                max_num_hands=2,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.45,
            )
            logger.info("Antenna finger tracking enabled (MediaPipe Hands)")
        except Exception as exc:
            self._enabled = False
            self._hands = None
            logger.warning("Failed to initialize MediaPipe Hands: %s", exc)

    def _extract_controls(self, results: Any) -> Optional[Tuple[float, float, int]]:
        """Extract left/right antenna targets from one or two raised index fingers."""
        if results is None:
            return None

        hand_landmarks_list = getattr(results, "multi_hand_landmarks", None)
        if not hand_landmarks_list:
            return None

        handedness_list = getattr(results, "multi_handedness", None) or []

        controls: List[Dict[str, float | str]] = []
        for idx, hand_landmarks in enumerate(hand_landmarks_list):
            label = "unknown"
            if idx < len(handedness_list):
                try:
                    label = str(handedness_list[idx].classification[0].label).lower()
                except Exception:
                    label = "unknown"

            landmarks = hand_landmarks.landmark
            middle_finger_pip = landmarks[9]
            palm_center = self._norm((middle_finger_pip.x, middle_finger_pip.y))

            index_tip = self._norm((landmarks[8].x, landmarks[8].y))
            index_mcp = self._norm((landmarks[5].x, landmarks[5].y))
            index_pip = self._norm((landmarks[6].x, landmarks[6].y))

            is_raised = bool(index_tip[1] < index_pip[1] < index_mcp[1])
            if not is_raised:
                continue

            angle_deg = -self._finger_orientation_deg(index_mcp, index_tip)
            angle_rad = float(np.deg2rad(angle_deg))
            angle_rad = float(np.clip(angle_rad, -self._max_angle_rad, self._max_angle_rad))
            controls.append({"label": label, "x": float(palm_center[0]), "angle": angle_rad, "palm_x": float(palm_center[0])})

        finger_count = len(controls)
        if finger_count == 0:
            return None

        if finger_count == 1:
            angle = float(controls[0]["angle"])
            return (angle, angle, 1)

        rightmost_hand = min(controls, key=lambda h: float(h["palm_x"]))
        leftmost_hand = max(controls, key=lambda h: float(h["palm_x"]))

        left_angle = float(leftmost_hand["angle"])
        right_angle = float(rightmost_hand["angle"])
        return (left_angle, right_angle, min(2, finger_count))

    def _norm(self, xy: Tuple[float, float]) -> np.ndarray:
        """Normalize coordinates from [0,1] to [-1,1] and flip x axis."""
        return np.array([-(xy[0] - 0.5) * 2.0, (xy[1] - 0.5) * 2.0], dtype=np.float64)

    def _finger_orientation_deg(self, mcp: np.ndarray, tip: np.ndarray) -> float:
        """Return orientation in degrees where 0 means vertical up."""
        v = np.array([tip[0] - mcp[0], tip[1] - mcp[1]], dtype=np.float64)
        v[1] = -v[1]
        return float(math.degrees(math.atan2(v[0], v[1])))

    def _angle_diff(self, a: float, b: float) -> float:
        """Return smallest angular difference between two angles."""
        d = a - b
        d = ((d + math.pi) % (2.0 * math.pi)) - math.pi
        return float(d)

    def _allow_multiturn(self, new_joints: List[float], prev_joints: List[float], max_delta: float) -> List[float]:
        """Limit per-step angular change while preserving shortest path behavior."""
        output = copy.deepcopy(new_joints)
        for idx in range(len(output)):
            diff = self._angle_diff(output[idx], prev_joints[idx])
            if abs(diff) > max_delta:
                diff = max_delta if diff > 0 else -max_delta
            output[idx] = float(prev_joints[idx] + diff)
            if output[idx] > 3.0 * math.pi:
                output[idx] -= 2.0 * math.pi
            elif output[idx] < -3.0 * math.pi:
                output[idx] += 2.0 * math.pi
        return output
