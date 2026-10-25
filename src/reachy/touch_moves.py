"""Head and antenna moves played when Dobby is petted or tapped."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from reachy_mini.utils import create_head_pose

from reachy.motion import Move

_Pose = tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, float | None]


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class PetMove(Move):
    """Lean into the hand: antennas drop back, head tilts and lifts, antennas sway until released."""

    exclusive = True

    EASE_IN_S = 1.0
    EASE_OUT_S = 1.0
    ANTENNAS = np.array([-3.05, 3.05])  # folded back, as in the SDK sleep pose
    SWAY_RAD = 0.15
    SWAY_HZ = 0.4

    def __init__(self, hold_s: float = 30.0) -> None:
        """`hold_s` caps the pose if `release` never comes."""
        self._release_at = hold_s
        self._started_wall: float | None = None  # wall time of move-local t=0

    @property
    def duration(self) -> float:
        """Run until released, then ease out."""
        return self._release_at + self.EASE_OUT_S

    def release(self, at: float) -> None:
        """Start easing out at move-local time `at`."""
        self._release_at = min(self._release_at, max(0.0, at))

    def release_now(self) -> None:
        """Start easing out now, measured on the move's own clock (it may have waited in queue)."""
        started = self._started_wall
        self.release(0.0 if started is None else time.monotonic() - started)

    def _amount(self, t: float) -> float:
        held = _smooth(min(t, self._release_at) / self.EASE_IN_S)
        if t <= self._release_at:
            return held
        return held * (1 - _smooth((t - self._release_at) / self.EASE_OUT_S))

    def evaluate(self, t: float) -> _Pose:
        """Evaluate the petting pose at time `t`."""
        if self._started_wall is None:
            self._started_wall = time.monotonic() - t
        f = self._amount(t)
        # Sway only inward: the folded pose already sits next to the joint limit.
        phase = 2 * np.pi * self.SWAY_HZ * max(0.0, t - self.EASE_IN_S)
        sway = self.SWAY_RAD * (1 - np.cos(phase)) / 2
        antennas = f * (self.ANTENNAS - np.sign(self.ANTENNAS) * sway)
        head = create_head_pose(z=6 * f, roll=10 * f, pitch=-5 * f, mm=True, degrees=True)
        return head, antennas.astype(np.float64), 0.0


class TapMove(Move):
    """Startle: antennas flick up, head pulls back with a small shake, then back to neutral."""

    exclusive = True

    DURATION_S = 1.2
    ATTACK_S = 0.1
    ANTENNAS = np.array([0.6, -0.6])

    @property
    def duration(self) -> float:
        """Fixed short duration."""
        return self.DURATION_S

    def evaluate(self, t: float) -> _Pose:
        """Evaluate the startle pose at time `t`."""
        if t < self.ATTACK_S:
            f = t / self.ATTACK_S
        else:
            f = 1 - _smooth((t - self.ATTACK_S) / (self.DURATION_S - self.ATTACK_S))
        shake = 6 * np.sin(2 * np.pi * 2 * max(0.0, t - self.ATTACK_S)) * f
        head = create_head_pose(x=-8 * f, yaw=shake, mm=True, degrees=True)
        return head, (f * self.ANTENNAS).astype(np.float64), 0.0
