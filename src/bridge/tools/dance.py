from __future__ import annotations

import logging
import random
from typing import Any, Dict

import numpy as np

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult
from reachy.motion import Move, MotionManager

logger = logging.getLogger(__name__)

try:
    from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
    from reachy_mini_dances_library.dance_move import DanceMove

    DANCE_AVAILABLE = True
except Exception as exc:
    logger.warning("Dance library not available: %s", exc)
    AVAILABLE_MOVES = {}
    DanceMove = None
    DANCE_AVAILABLE = False


def _build_move_description(available_moves: Dict[str, Any]) -> str:
    lines = ["Name of the move; use 'random' or omit for random."]
    if not available_moves:
        return " ".join(lines)
    lines.append("Available moves:")
    for name, (_, _, metadata) in sorted(available_moves.items()):
        description = ""
        if isinstance(metadata, dict):
            description = str(metadata.get("description", "")).strip()
        if description:
            lines.append(f"- {name}: {description}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


class DanceQueueMove(Move):
    """Wrapper to adapt library DanceMove to the motion queue protocol."""

    def __init__(self, move_name: str) -> None:
        if not DANCE_AVAILABLE or DanceMove is None:
            raise RuntimeError("Dance library not available")
        self._move_name = move_name
        self._dance_move = DanceMove(move_name)
        self._logged_start = False

    @property
    def duration(self) -> float:
        return float(self._dance_move.duration)

    def evaluate(
        self, t: float
    ) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
        try:
            head_pose, antennas, body_yaw = self._dance_move.evaluate(t)
            if isinstance(antennas, tuple):
                antennas = np.array([antennas[0], antennas[1]], dtype=np.float64)
            # Log once when move starts
            if not self._logged_start and t < 0.1:
                logger.info("Executing dance move '%s' (duration=%.2fs)", self._move_name, self.duration)
                self._logged_start = True
            return head_pose, antennas, body_yaw
        except Exception as exc:
            logger.error("Error evaluating dance move '%s': %s", self._move_name, exc)
            from reachy_mini.utils import create_head_pose

            neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            return neutral_head_pose, np.array([0.0, 0.0], dtype=np.float64), 0.0


class DanceTool:
    """Queue a dance move from the Reachy Mini dance library."""

    def __init__(self, motion_manager: MotionManager) -> None:
        self._motion_manager = motion_manager
        self._available_moves = dict(AVAILABLE_MOVES)
        self._move_names = sorted(self._available_moves.keys())
        self._move_description = _build_move_description(self._available_moves)
        self._move_enum = ["random", *self._move_names] if self._move_names else ["random"]

    def definition(self) -> ToolDefinition:
        parameters = {
            "type": "object",
            "properties": {
                "move": {
                    "type": "string",
                    "description": self._move_description,
                    "enum": self._move_enum,
                },
                "repeat": {
                    "type": "integer",
                    "description": "How many times to repeat the move (default 1).",
                    "minimum": 1,
                },
            },
            "additionalProperties": False,
        }
        return ToolDefinition(
            name="dance",
            description="Play a named or random dance move once (or repeat). Non-blocking.",
            parameters=parameters,
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        if not DANCE_AVAILABLE:
            return ToolExecutionResult(
                output={
                    "ok": False,
                    "message": "Dance library not available; install reachy_mini_dances_library.",
                }
            )
        if not self._move_names:
            return ToolExecutionResult(
                output={
                    "ok": False,
                    "message": "No dance moves are available.",
                }
            )
        if self._motion_manager is None:
            return ToolExecutionResult(
                output={
                    "ok": False,
                    "message": "Motion manager not available; REACHY_BRIDGE_URL must use sdk.",
                }
            )

        move_raw = arguments.get("move")
        move_name = str(move_raw).strip().lower() if move_raw is not None else ""
        if not move_name or move_name == "random":
            move_name = random.choice(self._move_names)
        if move_name not in self._available_moves:
            return ToolExecutionResult(
                output={
                    "ok": False,
                    "message": f"Unknown dance move '{move_name}'.",
                    "available_moves": self._move_names,
                }
            )

        repeat_raw = arguments.get("repeat", 1)
        try:
            repeat = int(repeat_raw)
        except (TypeError, ValueError):
            return ToolExecutionResult(
                output={"ok": False, "message": "Repeat must be an integer."}
            )
        if repeat < 1:
            return ToolExecutionResult(
                output={"ok": False, "message": "Repeat must be at least 1."}
            )

        logger.info("Queueing dance move %s repeat=%d", move_name, repeat)
        for _ in range(repeat):
            self._motion_manager.queue_move(DanceQueueMove(move_name))

        return ToolExecutionResult(
            output={
                "ok": True,
                "message": "Dance move queued",
                "move": move_name,
                "repeat": repeat,
            }
        )
