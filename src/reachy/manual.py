"""Manual control of head, antennas and body from the web UI's digital twin.

The page sends a target; a ManualMove (exclusive, so tracking and speech sway fade out) eases the
robot toward it with a speed limit, so a jump on the controls never snaps the motors.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from reachy_mini.utils import create_head_pose

from reachy.motion import FullBodyPose, Move

# Same ranges as the page's controls (frontend/src/robot/pose.ts). Metres and radians.
LIMITS: dict[str, tuple[float, float]] = {
    "x": (-0.02, 0.02),
    "y": (-0.02, 0.02),
    "z": (-0.04, 0.025),
    "roll": (-0.5, 0.5),
    "pitch": (-0.6, 0.6),
    "yaw": (-1.1, 1.1),
    "antenna": (-2.5, 2.5),
    "yawBody": (-2.8, 2.8),
}

_EASE_S = 0.12  # time constant of the easing toward the target
_MAX_SPEED = {"move": 0.1, "turn": 2.0, "antenna": 6.0, "body": 1.5}  # m/s, rad/s


def _clamp(value: float, key: str) -> float:
    lo, hi = LIMITS[key]
    return min(hi, max(lo, value))


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("not a number")  # noqa: TRY004 - bad input from a page, handled as one kind
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("not finite")
    return number


@dataclass(frozen=True)
class ManualTarget:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    antennas: tuple[float, float] = (0.0, 0.0)  # [right, left], as the SDK orders them
    body_yaw: float = 0.0

    @classmethod
    def from_message(
        cls, message: Mapping[str, Any], base: ManualTarget | None = None
    ) -> ManualTarget:
        """Parse a page's target (camelCase `yawBody`), clamped; fields it leaves out keep `base`'s values."""
        t = base or cls()
        values: dict[str, Any] = {}
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            if key in message:
                values[key] = _clamp(_number(message[key]), key)
        if "antennas" in message:
            pair = message["antennas"]
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("antennas must be two numbers")
            values["antennas"] = (
                _clamp(_number(pair[0]), "antenna"),
                _clamp(_number(pair[1]), "antenna"),
            )
        if "yawBody" in message:
            values["body_yaw"] = _clamp(_number(message["yawBody"]), "yawBody")
        return replace(t, **values)

    def to_message(self) -> dict:
        r = lambda v: round(float(v), 4)
        return {
            "x": r(self.x),
            "y": r(self.y),
            "z": r(self.z),
            "roll": r(self.roll),
            "pitch": r(self.pitch),
            "yaw": r(self.yaw),
            "antennas": [r(self.antennas[0]), r(self.antennas[1])],
            "yawBody": r(self.body_yaw),
        }

    def head_pose(self) -> NDArray[np.float64]:
        return create_head_pose(
            x=self.x,
            y=self.y,
            z=self.z,
            roll=self.roll,
            pitch=self.pitch,
            yaw=self.yaw,
            degrees=False,
            mm=False,
        )


def target_from_pose(pose: FullBodyPose) -> ManualTarget:
    """Split an SDK pose (4x4 head with R = Rz(yaw)·Ry(pitch)·Rx(roll)) into the page's fields."""
    head, antennas, body_yaw = pose
    m = np.asarray(head, dtype=np.float64)
    return ManualTarget(
        x=float(m[0, 3]),
        y=float(m[1, 3]),
        z=float(m[2, 3]),
        roll=float(math.atan2(m[2, 1], m[2, 2])),
        pitch=float(-math.asin(max(-1.0, min(1.0, m[2, 0])))),
        yaw=float(math.atan2(m[1, 0], m[0, 0])),
        antennas=(float(antennas[0]), float(antennas[1])),
        body_yaw=float(body_yaw),
    )


def _step(current: float, goal: float, dt: float, max_speed: float) -> float:
    delta = (goal - current) * (1.0 - math.exp(-dt / _EASE_S))
    limit = max_speed * dt
    return current + max(-limit, min(limit, delta))


class ManualMove(Move):
    """Holds the robot where the page puts it, easing toward each new target; runs until replaced."""

    exclusive = True

    def __init__(self, start: ManualTarget | Callable[[], ManualTarget]) -> None:
        """`start` is the pose to ease from, or a function that reads it when the move begins
        (a hold queued behind an emotion must start from where the emotion ended)."""
        self._lock = threading.Lock()
        self._start_from = start if callable(start) else None
        self._now: ManualTarget | None = None if callable(start) else start
        self._target: ManualTarget | None = self._now
        self._last_t: float | None = None

    @property
    def duration(self) -> float:
        return float("inf")

    def set_target(self, target: ManualTarget) -> None:
        with self._lock:
            self._target = target

    def target(self) -> ManualTarget:
        with self._lock:
            if self._target is not None:
                return self._target
        return self._now if self._now is not None else self._start_from()  # type: ignore[misc]

    def evaluate(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        if self._now is None:
            self._now = self._start_from()  # type: ignore[misc]
        goal, cur = self.target(), self._now
        move, turn = _MAX_SPEED["move"], _MAX_SPEED["turn"]
        self._now = cur = ManualTarget(
            x=_step(cur.x, goal.x, dt, move),
            y=_step(cur.y, goal.y, dt, move),
            z=_step(cur.z, goal.z, dt, move),
            roll=_step(cur.roll, goal.roll, dt, turn),
            pitch=_step(cur.pitch, goal.pitch, dt, turn),
            yaw=_step(cur.yaw, goal.yaw, dt, turn),
            antennas=(
                _step(cur.antennas[0], goal.antennas[0], dt, _MAX_SPEED["antenna"]),
                _step(cur.antennas[1], goal.antennas[1], dt, _MAX_SPEED["antenna"]),
            ),
            body_yaw=_step(cur.body_yaw, goal.body_yaw, dt, _MAX_SPEED["body"]),
        )
        return cur.head_pose(), np.array(cur.antennas, dtype=np.float64), cur.body_yaw
