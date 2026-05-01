"""Tool that cancels a scheduled reminder by job ID."""

from __future__ import annotations

from typing import Any, Dict

from bridge.runtime.adapters.openclaw_gateway import OpenClawGatewayClient
from bridge.tools.contracts import ToolDefinition, ToolExecutionResult


class CancelReminderTool:
    """Cancel a previously scheduled reminder by its job ID."""

    def __init__(self, client: OpenClawGatewayClient) -> None:
        self._client = client

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="cancel_reminder",
            description=(
                "Cancel a previously scheduled reminder using the job_id returned by create_reminder. "
                "Use this when the user asks to cancel or remove a reminder or recurring alarm."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "The job_id returned when the reminder was created.",
                    },
                },
                "required": ["job_id"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        job_id = str(arguments.get("job_id", "")).strip()
        if not job_id:
            return ToolExecutionResult(output={"ok": False, "message": "job_id is required"})

        try:
            self._client.cancel_reminder(job_id)
        except Exception as exc:
            return ToolExecutionResult(output={"ok": False, "message": f"Failed to cancel reminder: {exc}"})

        return ToolExecutionResult(output={"ok": True, "cancelled": True, "job_id": job_id})
