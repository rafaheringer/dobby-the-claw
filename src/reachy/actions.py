"""Typed Reachy action commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ListeningGestureAction:
    """Command for listening antenna gesture."""

    amplitude_rad: float = 0.35
    duration_s: float = 0.6


@dataclass(frozen=True)
class ThinkGestureAction:
    """Command for think antenna gesture."""

    amplitude_rad: float = 0.35
    duration_s: float = 0.6


@dataclass(frozen=True)
class AntennaWaveGestureAction:
    """Command for generic antenna wave gesture."""

    amplitude_rad: float = 0.35
    duration_s: float = 0.6


@dataclass(frozen=True)
class HeadMoveAction:
    """Command for direct head pose target in degrees."""

    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass(frozen=True)
class GazeLookAtAction:
    """Command for image-space look-at targeting."""

    u: int = 0
    v: int = 0
    duration_s: float = 0.0


@dataclass(frozen=True)
class AntennaCycleGestureAction:
    """Command for multi-cycle antenna gesture."""

    amplitude_rad: float = 0.2
    cycles: int = 2
    duration_s: float = 0.4


@dataclass(frozen=True)
class CameraCaptureSnapshotAction:
    """Command for capturing one camera snapshot."""

    pass


@dataclass(frozen=True)
class PlayRecordedMoveAction:
    """Command for playing a recorded move from a dataset."""

    dataset_name: str
    move_name: str
    play_frequency: float = 100.0
    initial_goto_duration: float = 0.0
    sound: bool = True


ReachyAction = (
    ListeningGestureAction
    | ThinkGestureAction
    | AntennaWaveGestureAction
    | HeadMoveAction
    | GazeLookAtAction
    | AntennaCycleGestureAction
    | CameraCaptureSnapshotAction
    | PlayRecordedMoveAction
)
