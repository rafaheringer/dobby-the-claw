"""Finger antenna control ignores a hand right in front of the camera (reaching to pet)."""

from types import SimpleNamespace

import numpy as np

from reachy.finger_antenna_controller import FingerAntennaController


def _hand(scale):
    # 21 landmarks of a raised index finger, centred in the frame, spanning `scale` of it.
    pts = [(0.5, 0.5)] * 21
    pts[5], pts[6], pts[8], pts[9] = (0.5, 0.6), (0.5, 0.55), (0.5, 0.5), (0.5, 0.6)
    pts[0] = (0.5, 0.6 + scale)  # wrist
    landmarks = [SimpleNamespace(x=x, y=y) for x, y in pts]
    return SimpleNamespace(landmark=landmarks)


def _controller():
    c = object.__new__(FingerAntennaController)
    c._max_angle_rad = float(np.pi)
    return c


def test_normal_hand_controls_antennas():
    results = SimpleNamespace(multi_hand_landmarks=[_hand(0.2)], multi_handedness=[])
    assert _controller()._extract_controls(results) is not None


def test_hand_filling_the_frame_is_ignored():
    results = SimpleNamespace(multi_hand_landmarks=[_hand(0.35)], multi_handedness=[])
    assert _controller()._extract_controls(results) is None
