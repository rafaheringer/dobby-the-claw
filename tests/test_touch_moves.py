"""Pose moves played when Dobby is petted or tapped."""

import numpy as np
import pytest

from reachy.touch_moves import PetMove, TapMove


def test_pet_move_reaches_pose_and_releases():
    m = PetMove()
    _, ant, _ = m.evaluate(0.6)
    assert ant[0] == pytest.approx(-0.9, abs=0.16) and ant[1] == pytest.approx(0.9, abs=0.16)
    m.release(at=5.0)
    assert m.duration == pytest.approx(5.8)
    _, ant_end, _ = m.evaluate(5.8)
    assert abs(ant_end[0]) < 0.05


def test_pet_move_holds_until_released():
    m = PetMove(hold_s=30.0)
    assert m.duration == pytest.approx(30.8)


def test_tap_move_is_short_and_returns_to_neutral():
    m = TapMove()
    assert m.duration == pytest.approx(1.2)
    head, ant, _ = m.evaluate(1.2)
    assert np.allclose(head, np.eye(4), atol=1e-3) and np.allclose(ant, 0, atol=1e-3)
