"""Head and antenna moves played when Dobby is petted or tapped."""

from __future__ import annotations

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

    EASE_IN_S = 0.6
    EASE_OUT_S = 0.8
    ANTENNAS = np.array([-0.9, 0.9])
    SWAY_RAD = 0.15
    SWAY_HZ = 0.4

    def __init__(self, hold_s: float = 30.0) -> None:
        """`hold_s` caps the pose if `release` never comes."""
        self._release_at = hold_s

    @property
    def duration(self) -> float:
        """Run until released, then ease out."""
        return self._release_at + self.EASE_OUT_S

    def release(self, at: float) -> None:
        """Start easing out at move-local time `at`."""
        self._release_at = min(self._release_at, max(0.0, at))

    def _amount(self, t: float) -> float:
        held = _smooth(min(t, self._release_at) / self.EASE_IN_S)
        if t <= self._release_at:
            return held
        return held * (1 - _smooth((t - self._release_at) / self.EASE_OUT_S))

    def evaluate(self, t: float) -> _Pose:
        """Evaluate the petting pose at time `t`."""
        f = self._amount(t)
        sway = self.SWAY_RAD * np.sin(2 * np.pi * self.SWAY_HZ * max(0.0, t - self.EASE_IN_S))
        antennas = f * (self.ANTENNAS + np.array([sway, -sway]))
        head = create_head_pose(z=6 * f, roll=10 * f, pitch=-5 * f, mm=True, degrees=True)
        return head, antennas.astype(np.float64), 0.0


class TapMove(Move):
    """Startle: antennas flick up, head pulls back with a small shake, then back to neutral."""

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
