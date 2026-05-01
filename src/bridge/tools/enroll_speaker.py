"""Tool that lets Dobby enroll a new speaker face profile during conversation."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult


class EnrollSpeakerTool:
    """Capture camera frames and enroll a face profile for speaker identification."""

    def __init__(self, camera_worker) -> None:
        self._camera_worker = camera_worker

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="enroll_speaker",
            description=(
                "Enrolls the current person's face so Dobby can recognize them in future interactions. "
                "Captures camera frames and saves them under the given name. "
                "Use when asked to remember someone or when talking to an unrecognized person and wanting to learn their name."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name to associate with this person's face.",
                    }
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            runtime_guardrail=(
                "Enroll a speaker only after explicitly learning their name — either because the user asked you "
                "to remember them or because you proactively asked an unrecognized person what to call them. "
                "Never guess a name or enroll without consent."
            ),
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        name = str(arguments.get("name", "")).strip()
        if not name:
            return ToolExecutionResult(output={"ok": False, "message": "Nome não informado."})

        face_recognizer = self._camera_worker._face_recognizer
        if face_recognizer is None or not face_recognizer.available:
            return ToolExecutionResult(
                output={"ok": False, "message": "Reconhecimento facial não disponível."}
            )

        frames = self._capture_frames(count=4, interval_s=0.35)
        if not frames:
            return ToolExecutionResult(
                output={"ok": False, "message": "Câmera não disponível para captura."}
            )

        count = face_recognizer.enroll(name, frames)
        if count > 0:
            profiles = face_recognizer.profile_names()
            return ToolExecutionResult(
                output={
                    "ok": True,
                    "message": f"'{name}' cadastrado(a) com sucesso ({count} amostras aceitas).",
                    "profiles": profiles,
                }
            )

        return ToolExecutionResult(
            output={
                "ok": False,
                "message": (
                    f"Não detectei um rosto claro para cadastrar '{name}'. "
                    "Peça para a pessoa olhar diretamente para a câmera e tente novamente."
                ),
            }
        )

    def _capture_frames(self, count: int, interval_s: float) -> List:
        frames = []
        for _ in range(count):
            frame = self._camera_worker.get_latest_frame()
            if frame is not None:
                frames.append(frame)
            time.sleep(interval_s)
        return frames
