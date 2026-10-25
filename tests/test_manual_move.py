"""Manual control from the web UI's digital twin: target parsing, clamping, pose decomposition, smoothing."""

import math

import numpy as np
import pytest
from reachy_mini.utils import create_head_pose

from reachy.manual import LIMITS, ManualMove, ManualTarget, target_from_pose


def test_target_from_message_clamps_to_safe_limits():
    t = ManualTarget.from_message(
        {
            "x": 1,
            "y": -1,
            "z": 0.5,
            "roll": 9,
            "pitch": -9,
            "yaw": 9,
            "antennas": [9, -9],
            "yawBody": 9,
        }
    )
    assert t.x == LIMITS["x"][1] and t.y == LIMITS["y"][0] and t.z == LIMITS["z"][1]
    assert (
        t.roll == LIMITS["roll"][1] and t.pitch == LIMITS["pitch"][0] and t.yaw == LIMITS["yaw"][1]
    )
    assert (
        t.antennas == (LIMITS["antenna"][1], LIMITS["antenna"][0])
        and t.body_yaw == LIMITS["yawBody"][1]
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"x": "abc"},
        {"x": float("nan")},
        {"yaw": float("inf")},
        {"antennas": [1]},
        {"antennas": "no"},
        {"z": None},
    ],
)
def test_target_from_message_rejects_garbage(bad):
    with pytest.raises(ValueError):
        ManualTarget.from_message(bad)


def test_target_from_message_keeps_missing_fields_from_base():
    base = ManualTarget(yaw=0.4, antennas=(0.2, -0.2))
    t = ManualTarget.from_message({"pitch": 0.1}, base=base)
    assert t.yaw == 0.4 and t.pitch == 0.1 and t.antennas == (0.2, -0.2)


def test_pose_decomposition_round_trips_the_sdk_head_pose():
    head = create_head_pose(
        x=0.01, y=-0.005, z=0.012, roll=0.2, pitch=-0.3, yaw=0.5, degrees=False, mm=False
    )
    t = target_from_pose((head, (0.7, -0.4), 0.9))
    assert (
        t.x == pytest.approx(0.01) and t.y == pytest.approx(-0.005) and t.z == pytest.approx(0.012)
    )
    assert (
        t.roll == pytest.approx(0.2)
        and t.pitch == pytest.approx(-0.3)
        and t.yaw == pytest.approx(0.5)
    )
    assert t.antennas == pytest.approx((0.7, -0.4)) and t.body_yaw == pytest.approx(0.9)


def test_manual_move_owns_the_head_and_never_ends():
    assert ManualMove.exclusive
    assert math.isinf(ManualMove(ManualTarget()).duration)


def test_manual_move_eases_from_the_start_pose_to_the_target():
    move = ManualMove(ManualTarget(yaw=0.0))
    move.set_target(ManualTarget(yaw=0.5, antennas=(1.0, -1.0)))
    _, _, _ = move.evaluate(0.0)
    head, antennas, _ = move.evaluate(0.05)
    early_yaw = target_from_pose((head, tuple(antennas), 0.0)).yaw
    assert 0.0 < early_yaw < 0.5  # moving, not jumping
    for i in range(2, 200):
        head, antennas, body_yaw = move.evaluate(i * 0.02)
    final = target_from_pose((head, tuple(antennas), body_yaw))
    assert final.yaw == pytest.approx(0.5, abs=1e-3) and final.antennas == pytest.approx(
        (1.0, -1.0), abs=1e-3
    )


def test_manual_move_limits_speed_of_big_jumps():
    move = ManualMove(ManualTarget(body_yaw=0.0))
    move.set_target(ManualTarget(body_yaw=2.8))
    move.evaluate(0.0)
    _, _, body_yaw = move.evaluate(0.1)
    assert 0.0 < body_yaw <= 0.2  # at most ~1.5 rad/s


def test_manual_move_evaluates_to_sdk_shapes():
    head, antennas, body_yaw = ManualMove(ManualTarget()).evaluate(0.0)
    assert (
        head.shape == (4, 4)
        and np.allclose(head, np.eye(4))
        and len(antennas) == 2
        and body_yaw == 0.0
    )


def test_manual_move_can_start_from_where_the_robot_is_when_it_begins():
    """Queued behind an emotion, the hold must ease from where the emotion ended, not from before it."""
    where = {"pose": ManualTarget(yaw=0.0)}
    move = ManualMove(lambda: where["pose"])
    move.set_target(ManualTarget(yaw=0.0))
    where["pose"] = ManualTarget(yaw=0.6)  # the emotion ran in between
    head, _, _ = move.evaluate(10.0)
    assert target_from_pose((head, (0.0, 0.0), 0.0)).yaw == pytest.approx(0.6)
