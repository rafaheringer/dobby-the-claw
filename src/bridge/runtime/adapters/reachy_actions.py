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
        self._reachy = reachy

    def gesture_listening(self) -> None:
        _ = self._reachy.execute_typed_action(ListeningGestureAction())

    def gesture_think(self) -> None:
        _ = self._reachy.execute_typed_action(ThinkGestureAction())

    def gesture_delegating(self) -> None:
        _ = self._reachy.execute_typed_action(
            AntennaCycleGestureAction(amplitude_rad=0.16, cycles=3, duration_s=0.22)
        )

    def wake_up(self) -> None:
        self._reachy.wake_up()

    def goto_sleep(self) -> None:
        self._reachy.goto_sleep()
