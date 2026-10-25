"""Robot control for the web UI's digital twin: the pose stream and manual mode.

Manual mode puts an exclusive ManualMove at the head of the motion queue, so face tracking and
speech sway fade out while a page drives the head. It ends when the page switches back to auto,
when that page disconnects, when Dobby falls asleep, or after a while without commands.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from reachy.manual import ManualMove, ManualTarget, target_from_pose

logger = logging.getLogger(__name__)

IDLE_RELEASE_S = 300.0  # a forgotten manual session gives the robot back after five minutes

# The page's four emotions, played with Pollen's recorded moves (bridge/tools/emotion.py).
EMOTIONS = {"feliz": "cheerful1", "curioso": "curious1", "sono": "tired1", "surpreso": "surprised1"}

_ASLEEP = "O Dobby está dormindo: acorde-o para mexer nele."
_UNAVAILABLE = "Controle do robô indisponível (sem o SDK do Reachy)."


class RobotControl:
    def __init__(
        self,
        motion: Any,
        run_tool: Callable[[str, dict], Any] | None,
        asleep: Callable[[], bool],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._motion = motion
        self._run_tool = run_tool
        self._asleep = asleep
        self._clock = clock
        self._lock = threading.Lock()
        self._move: ManualMove | None = None
        self._owner: object = None
        self._last_command = 0.0

    # ------------------------------------------------------------------ pose stream

    def snapshot(self) -> dict | None:
        """What pages draw: the pose the robot is being commanded to, and who is in charge."""
        if self._motion is None:
            return None
        pose = target_from_pose(self._motion.commanded_pose())
        with self._lock:
            move = self._move
        return {
            "type": "robot",
            "mode": "manual" if move is not None else "auto",
            "pose": pose.to_message(),
            "target": move.target().to_message() if move is not None else None,
        }

    # ------------------------------------------------------------------ commands

    def set_manual(self, on: bool, owner: object) -> str | None:
        if not on:
            with self._lock:
                self._end_locked()
            return None
        error = self._refusal()
        if error:
            return error
        with self._lock:
            self._begin_locked(owner)
        return None

    def set_target(self, message: Mapping[str, Any], owner: object) -> str | None:
        error = self._refusal()
        if error:
            return error
        with self._lock:
            base = self._move.target() if self._move is not None else self._current()
            try:
                target = ManualTarget.from_message(message, base=base)
            except ValueError:
                return "posição inválida"
            self._begin_locked(owner)
            self._move.set_target(target)
        return None

    def play_emotion(self, name: str, owner: object) -> str | None:
        """Play one of the page's emotions; blocking (loading a recorded move can take a moment)."""
        if name not in EMOTIONS:
            return "emoção desconhecida"
        error = self._refusal()
        if error or self._run_tool is None:
            return error or _UNAVAILABLE
        with self._lock:
            manual = self._move
            if manual is not None:
                self._motion.clear_move_queue()  # the hold never ends by itself: the emotion goes first
            self._last_command = self._clock()
        result = self._run_tool("express_emotion", {"emotion": EMOTIONS[name], "sound": True})
        output = getattr(result, "output", result)  # a ToolExecutionResult, or its output
        if isinstance(output, Mapping) and output.get("ok") is False:
            logger.warning("Twin emotion %s failed: %s", name, output.get("message"))
        with self._lock:
            if self._move is manual and manual is not None:
                # Resume holding the page's target once the emotion is over, easing from wherever it ended.
                resumed = ManualMove(self._current)  # read when it starts, after the emotion
                resumed.set_target(manual.target())
                self._move = resumed
                self._motion.queue_move(resumed)
        return None

    def release(self, owner: object) -> None:
        """A page went away: if it was driving the robot, give the robot back."""
        with self._lock:
            if self._move is not None and self._owner is owner:
                self._end_locked()

    def tick(self) -> None:
        """Safety releases, called a few times a second."""
        with self._lock:
            if self._move is None:
                return
            if self._asleep() or self._clock() - self._last_command > IDLE_RELEASE_S:
                self._end_locked()

    # ------------------------------------------------------------------ internals

    def _refusal(self) -> str | None:
        if self._motion is None:
            return _UNAVAILABLE
        if self._asleep():
            return _ASLEEP
        return None

    def _current(self) -> ManualTarget:
        return target_from_pose(self._motion.commanded_pose())

    def _begin_locked(self, owner: object) -> None:
        self._owner = owner
        self._last_command = self._clock()
        if self._move is not None:
            return
        self._move = ManualMove(self._current())
        self._motion.clear_move_queue()
        self._motion.queue_move(self._move)
        logger.info("Twin: manual control on")

    def _end_locked(self) -> None:
        if self._move is None:
            return
        self._move = None
        self._owner = None
        self._motion.clear_move_queue()
        self._motion.mark_activity()
        logger.info("Twin: manual control off")
