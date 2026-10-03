"""Tool that reports the system's current date and time."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult

_WEEKDAYS_PT = (
    "segunda-feira",
    "terça-feira",
    "quarta-feira",
    "quinta-feira",
    "sexta-feira",
    "sábado",
    "domingo",
)
_MONTHS_PT = (
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
)


class CurrentTimeTool:
    """Read the current local date and time from the host system clock."""

    def definition(self) -> ToolDefinition:
        """Return OpenAI function schema for the clock lookup."""
        return ToolDefinition(
            name="get_current_time",
            description="Get the current date, time and weekday from the system clock.",
            runtime_guardrail=(
                "Always call `get_current_time` before answering about the current time, date, "
                "weekday, or relative dates (hoje, amanhã, ontem). Never guess or reuse an earlier "
                "answer, because time changes between turns. Never delegate this to `delegate_task`."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Return the current local time in structured and spoken-friendly forms."""
        now = datetime.now().astimezone()
        return ToolExecutionResult(
            output={
                "ok": True,
                "iso": now.isoformat(timespec="seconds"),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M"),
                "weekday": _WEEKDAYS_PT[now.weekday()],
                "spoken_date": f"{now.day} de {_MONTHS_PT[now.month - 1]} de {now.year}",
                "timezone": now.tzname(),
                "utc_offset": now.strftime("%z"),
            }
        )
