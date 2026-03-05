"""Tool for expressing robot emotions through recorded move playback."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from reachy.motion import Move, MotionManager

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult


logger = logging.getLogger(__name__)

try:
    from reachy_mini.motion.recorded_move import RecordedMoves

    RECORDED_MOVE_AVAILABLE = True
except Exception as exc:
    logger.warning("Recorded move library not available: %s", exc)
    RecordedMoves = None
    RECORDED_MOVE_AVAILABLE = False


EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"

AVAILABLE_EMOTIONS = (
    "incomprehensible2",
    "scared1",
    "inquiring3",
    "anxiety1",
    "dying1",
    "laughing1",
    "understanding2",
    "welcoming2",
    "dance3",
    "furious1",
    "confused1",
    "sad1",
    "attentive2",
    "laughing2",
    "attentive1",
    "boredom1",
    "curious1",
    "surprised1",
    "no_excited1",
    "displeased2",
    "tired1",
    "yes1",
    "reprimand1",
    "enthusiastic2",
    "resigned1",
    "indifferent1",
    "amazed1",
    "displeased1",
    "dance2",
    "reprimand3",
    "disgusted1",
    "contempt1",
    "success2",
    "helpful1",
    "thoughtful1",
    "helpful2",
    "irritated1",
    "serenity1",
    "no_sad1",
    "relief1",
    "oops1",
    "loving1",
    "frustrated1",
    "calming1",
    "irritated2",
    "proud3",
    "surprised2",
    "dance1",
    "lost1",
    "reprimand2",
    "uncertain1",
    "inquiring1",
    "come1",
    "rage1",
    "yes_sad1",
    "impatient1",
    "success1",
    "exhausted1",
    "proud2",
    "downcast1",
    "shy1",
    "thoughtful2",
    "grateful1",
    "cheerful1",
    "boredom2",
    "impatient2",
    "go_away1",
    "proud1",
    "sad2",
    "inquiring2",
    "enthusiastic1",
    "electric1",
    "uncomfortable1",
    "understanding1",
    "lonely1",
    "welcoming1",
    "no1",
    "fear1",
    "oops2",
    "relief2",
    "sleep1",
)


class EmotionTool:
    """Expose Reachy emotion moves as a callable assistant tool."""

    def __init__(self, motion_manager: MotionManager | None) -> None:
        """Store Reachy/motion dependencies and emotion allowlist."""
        self._motion_manager = motion_manager
        self._available = tuple(sorted(set(AVAILABLE_EMOTIONS)))
        self._available_set = set(self._available)
        if RECORDED_MOVE_AVAILABLE and RecordedMoves is not None:
            self._recorded_moves = RecordedMoves(EMOTIONS_DATASET)
        else:
            self._recorded_moves = None

    def definition(self) -> ToolDefinition:
        """Return OpenAI function schema for emotion playback."""
        return ToolDefinition(
            name="express_emotion",
            description=(
                "Express an emotion using Reachy recorded moves from the emotions dataset. "
                "Use when a brief physical reaction helps communication."
            ),
            runtime_guardrail=(
                "Use `express_emotion` when a short physical emotional cue improves communication. "
                "Pick one emotion matching the answer tone and avoid overusing it (typically once per reply)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "description": "Emotion move name from the available emotions list.",
                        "enum": list(self._available),
                    },
                    "sound": {
                        "type": "boolean",
                        "description": "Play associated move sound when available (default true).",
                    },
                },
                "required": ["emotion"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Validate emotion input and trigger emotion playback."""
        emotion = str(arguments.get("emotion", "")).strip().lower()
        if not emotion:
            return ToolExecutionResult(output={"ok": False, "message": "Missing required argument: emotion"})
        if emotion not in self._available_set:
            return ToolExecutionResult(
                output={
                    "ok": False,
                    "message": f"Unknown emotion '{emotion}'.",
                    "available_emotions": list(self._available),
                }
            )

        sound = bool(arguments.get("sound", True))
        if self._motion_manager is None:
            return ToolExecutionResult(
                output={
                    "ok": False,
                    "message": "Motion manager not available; REACHY_BRIDGE_URL must use sdk.",
                    "emotion": emotion,
                    "dataset": EMOTIONS_DATASET,
                    "sound": sound,
                }
            )
        if self._recorded_moves is None:
            return ToolExecutionResult(
                output={
                    "ok": False,
                    "message": "Recorded move library not available.",
                    "emotion": emotion,
                    "dataset": EMOTIONS_DATASET,
                    "sound": sound,
                }
            )

        try:
            recorded_move = self._recorded_moves.get(emotion)
            self._motion_manager.queue_move(_EmotionQueueMove(recorded_move, emotion))
            return ToolExecutionResult(
                output={
                    "ok": True,
                    "message": f"Emotion queued via motion manager: {emotion}",
                    "emotion": emotion,
                    "dataset": EMOTIONS_DATASET,
                    "sound": sound,
                }
            )
        except Exception as exc:
            return ToolExecutionResult(
                output={
                    "ok": False,
                    "message": f"Failed to queue emotion '{emotion}': {exc}",
                    "emotion": emotion,
                    "dataset": EMOTIONS_DATASET,
                    "sound": sound,
                }
            )


class _EmotionQueueMove(Move):
    """Adapt SDK RecordedMove to MotionManager queue protocol."""

    def __init__(self, recorded_move: Any, emotion_name: str) -> None:
        """Wrap one recorded move with a diagnostic emotion label."""
        self._recorded_move = recorded_move
        self._emotion_name = emotion_name

    @property
    def duration(self) -> float:
        """Return wrapped move duration in seconds."""
        return float(self._recorded_move.duration)

    def evaluate(self, t: float) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
        """Evaluate wrapped recorded move at time `t`."""
        try:
            head_pose, antennas, body_yaw = self._recorded_move.evaluate(t)
            if isinstance(antennas, tuple):
                antennas = np.array([antennas[0], antennas[1]], dtype=np.float64)
            return head_pose, antennas, body_yaw
        except Exception as exc:
            logger.error("Error evaluating emotion move '%s': %s", self._emotion_name, exc)
            return None, np.array([0.0, 0.0], dtype=np.float64), 0.0
