"""Robot actions adapter backed by ReachyClient."""

from __future__ import annotations

from reachy.actions import (
    AntennaCycleGestureAction,
    ListeningGestureAction,
    ThinkGestureAction,
)
from reachy.client import ReachyClient

from bridge.runtime.ports import RobotActionsPort


class ReachyRobotActions(RobotActionsPort):
    """Translate runtime action intents into ReachyClient calls."""

    def __init__(self, reachy: ReachyClient) -> None:
        """Store Reachy client used to execute gesture and posture commands."""
        self._reachy = reachy

    def gesture_listening(self) -> None:
        """Trigger listening gesture on Reachy."""
        _ = self._reachy.execute_typed_action(ListeningGestureAction())

    def gesture_think(self) -> None:
        """Trigger thinking gesture on Reachy."""
        _ = self._reachy.execute_typed_action(ThinkGestureAction())

    def gesture_delegating(self) -> None:
        """Trigger delegating gesture used during external task delegation."""
        _ = self._reachy.execute_typed_action(
            AntennaCycleGestureAction(amplitude_rad=0.16, cycles=3, duration_s=0.22)
        )

    def wake_up(self) -> None:
        """Wake Reachy from sleep posture."""
        self._reachy.wake_up()

    def goto_sleep(self) -> None:
        """Send Reachy to sleep posture."""
        self._reachy.goto_sleep()

    def disable_motors(self) -> None:
        """Cut torque on all motors."""
        self._reachy.disable_motors()

    def enable_motors(self) -> None:
        """Re-energize all motors."""
        self._reachy.enable_motors()
