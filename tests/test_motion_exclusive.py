"""MotionManager helpers that let touch moves own the head and antennas."""

import numpy as np
from reachy_mini.utils import create_head_pose

from reachy.motion import fade_secondary_pose, next_secondary_weight
from reachy.touch_moves import PetMove, TapMove


def _offset_pose():
    return (create_head_pose(z=10, yaw=15, mm=True, degrees=True), (0.0, 0.0), 0.3)


def test_fade_secondary_pose_zero_weight_is_identity():
    head, antennas, body_yaw = fade_secondary_pose(_offset_pose(), 0.0)
    assert np.allclose(head, np.eye(4)) and antennas == (0.0, 0.0) and body_yaw == 0.0


def test_fade_secondary_pose_full_weight_is_unchanged():
    head, _, body_yaw = fade_secondary_pose(_offset_pose(), 1.0)
    assert np.allclose(head, _offset_pose()[0]) and body_yaw == 0.3


def test_secondary_weight_fades_out_and_back():
    w = 1.0
    for _ in range(5):
        w = next_secondary_weight(w, exclusive=True, dt=0.1)
    assert w == 0.0
    w = next_secondary_weight(w, exclusive=False, dt=0.2)
    assert 0.4 <= w <= 0.6


def test_touch_moves_are_exclusive():
    assert PetMove.exclusive and TapMove.exclusive
