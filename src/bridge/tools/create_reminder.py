"""Tool that schedules timed reminders via OpenClaw cron.add."""

from __future__ import annotations

from typing import Any, Dict

from bridge.runtime.adapters.openclaw_gateway import OpenClawGatewayClient
from bridge.tools.contracts import ToolDefinition, ToolExecutionResult


class CreateReminderTool:
    """Schedule a reminder that Dobby delivers proactively after a delay."""

    def __init__(self, client: OpenClawGatewayClient, callback_url: str) -> None:
        self._client = client
        self._callback_url = callback_url

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="create_reminder",
            description=(
                "Schedule a reminder to be spoken proactively to the user. "
                "Supports one-time reminders (delay_seconds), interval recurrence "
                "(repeat_every_seconds), and cron-based recurrence (cron_expression). "
                "Returns a job_id that can be used with cancel_reminder. "
                "Use this instead of delegate_task for any reminder, timer, or alarm."
            ),
            runtime_guardrail=(
                "Use `create_reminder` for any reminder, timer, alarm, or recurring notification — "
                "never delegate_task for these. Pass exactly one of: delay_seconds, "
                "repeat_every_seconds, or cron_expression."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The reminder message to speak to the user when it fires.",
                    },
                    "delay_seconds": {
                        "type": "integer",
                        "description": "One-time: seconds from now until the reminder fires.",
                        "minimum": 1,
                    },
                    "repeat_every_seconds": {
                        "type": "integer",
                        "description": "Recurring: repeat the reminder every N seconds (e.g. 1800 = every 30 min).",
                        "minimum": 1,
                    },
                    "cron_expression": {
                        "type": "string",
                        "description": (
                            "Recurring: standard 5-field cron expression (e.g. '0 9 * * 1-5' = "
                            "weekdays at 9am). Use with timezone."
                        ),
                    },
                    "timezone": {
                        "type": "string",
                        "description": (
                            "IANA timezone for cron_expression (e.g. 'America/Sao_Paulo'). "
                            "Defaults to America/Sao_Paulo when omitted."
                        ),
                    },
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        message = str(arguments.get("message", "")).strip()
        if not message:
            return ToolExecutionResult(output={"ok": False, "message": "message is required"})

        delay_seconds = arguments.get("delay_seconds")
        repeat_every_seconds = arguments.get("repeat_every_seconds")
        cron_expression = arguments.get("cron_expression")
        timezone = str(arguments.get("timezone") or "America/Sao_Paulo").strip()

        provided = sum(x is not None for x in (delay_seconds, repeat_every_seconds, cron_expression))
        if provided == 0:
            return ToolExecutionResult(
                output={"ok": False, "message": "One of delay_seconds, repeat_every_seconds, or cron_expression is required"}
            )
        if provided > 1:
            return ToolExecutionResult(
                output={"ok": False, "message": "Provide only one of delay_seconds, repeat_every_seconds, or cron_expression"}
            )

        try:
            job_id = self._client.schedule_reminder(
                message=message,
                callback_url=self._callback_url,
                delay_seconds=float(delay_seconds) if delay_seconds is not None else None,
                repeat_every_seconds=float(repeat_every_seconds) if repeat_every_seconds is not None else None,
                cron_expression=str(cron_expression).strip() if cron_expression is not None else None,
                timezone=timezone,
            )
        except Exception as exc:
            return ToolExecutionResult(output={"ok": False, "message": f"Reminder scheduling failed: {exc}"})

        kind = "one-time" if delay_seconds is not None else "recurring"
        return ToolExecutionResult(output={"ok": True, "scheduled": True, "kind": kind, "job_id": job_id})
