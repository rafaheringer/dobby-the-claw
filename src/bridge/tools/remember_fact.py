"""Tool that lets Dobby save an explicit fact to persistent per-speaker memory."""

from __future__ import annotations

from typing import Any, Dict, Optional

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult


class RememberFactTool:
    """Save an explicit fact about a speaker to persistent memory on demand."""

    def __init__(self, speaker_memory, camera_worker=None) -> None:
        self._speaker_memory = speaker_memory
        self._camera_worker = camera_worker

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="remember_fact",
            description=(
                "Save an explicit fact about a person to persistent memory. "
                "Use when the user shares important information they want you to remember "
                "(preferences, allergies, names of family members, routines, etc). "
                "Facts saved here are injected into future sessions automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to remember, written in third person (e.g. 'Prefere café sem açúcar').",
                    },
                    "speaker": {
                        "type": "string",
                        "description": (
                            "Name of the person the fact is about. "
                            "Defaults to the currently identified speaker if omitted."
                        ),
                    },
                },
                "required": ["fact"],
                "additionalProperties": False,
            },
            runtime_guardrail=(
                "Use remember_fact only for information the user explicitly wants persisted "
                "(they said 'lembra que...', 'anota isso', 'guarda isso', etc). "
                "Do not save trivial or conversational details."
            ),
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        fact = str(arguments.get("fact", "")).strip()
        if not fact:
            return ToolExecutionResult(output={"ok": False, "message": "Nenhum fato informado."})

        speaker = str(arguments.get("speaker", "")).strip()

        if not speaker and self._camera_worker is not None:
            with self._camera_worker._speaker_lock:
                speaker = self._camera_worker._current_speaker or ""

        if not speaker:
            return ToolExecutionResult(
                output={"ok": False, "message": "Não sei quem é o falante atual para salvar o fato."}
            )

        from reachy.face_recognizer import FaceRecognizer
        if speaker == FaceRecognizer.UNKNOWN:
            return ToolExecutionResult(
                output={"ok": False, "message": "Não posso salvar fatos para visitantes não identificados."}
            )

        self._speaker_memory.save_fact(speaker, fact)
        return ToolExecutionResult(output={"ok": True, "message": f"Anotado para {speaker}.", "speaker": speaker})
