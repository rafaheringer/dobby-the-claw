"""Robot actions adapter backed by ReachyClient."""

from __future__ import annotations

from bridge.reachy.client import ReachyClient

from bridge.runtime.ports import RobotActionsPort


class ReachyRobotActions(RobotActionsPort):
    """Translate runtime action intents into ReachyClient calls."""

    def __init__(self, reachy: ReachyClient) -> None:
        self._reachy = reachy

    def gesture_listening(self) -> None:
        _ = self._reachy.execute_action({"type": "gesture.listening"})

    def gesture_think(self) -> None:
        _ = self._reachy.execute_action({"type": "gesture.think"})

    def wake_up(self) -> None:
        self._reachy.wake_up()

    def goto_sleep(self) -> None:
        self._reachy.goto_sleep()
