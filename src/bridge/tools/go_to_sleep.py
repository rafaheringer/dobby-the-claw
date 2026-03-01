from __future__ import annotations

from typing import Any, Dict, Iterable

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult


class GoToSleepTool:
    def __init__(self, *, wakeword_enabled: bool, wakeword_aliases: Iterable[str]) -> None:
        self._wakeword_enabled = bool(wakeword_enabled)
        self._wakeword_aliases = tuple(
            alias.strip() for alias in wakeword_aliases if str(alias).strip()
        )

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="go_to_sleep",
            description=(
                "Put Reachy into sleep mode to stop the active session and wait for the wake word."
            ),
            runtime_guardrail=(
                "Use `go_to_sleep` only when the user explicitly asks to sleep, pause listening, or go idle. "
                "After calling, confirm and remind the user to say the wake word to resume."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        if not self._wakeword_enabled:
            return ToolExecutionResult(
                output={
                    "ok": False,
                    "message": "Offline wakeword is disabled; refusing to enter sleep mode.",
                    "wakeword_enabled": False,
                }
            )

        return ToolExecutionResult(
            output={
                "ok": True,
                "message": "Sleep mode requested",
                "wakeword_enabled": True,
                "wakeword_aliases": list(self._wakeword_aliases),
            }
        )
