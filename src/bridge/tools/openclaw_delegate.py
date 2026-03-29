"""Tool that delegates requests to the external OpenClaw gateway."""

from __future__ import annotations

from typing import Any, Dict

from bridge.runtime.adapters.openclaw_gateway import OpenClawGatewayClient
from bridge.tools.contracts import ToolDefinition, ToolExecutionResult


class OpenClawDelegateTool:
    """Expose OpenClaw task delegation as a runtime tool."""

    def __init__(self, client: OpenClawGatewayClient, default_language: str) -> None:
        """Initialize delegation tool with gateway client and default language."""
        self._client = client
        self._default_language = default_language

    def definition(self) -> ToolDefinition:
        """Return OpenAI function schema and guardrails for delegation."""
        return ToolDefinition(
            name="delegate_task",
            description=(
                "Delegate any request you cannot complete fully and confidently on your own to an external executor. "
                "Do not answer with inability/capability disclaimers when this tool can handle the request; call it instead. "
                "Before calling this tool, first tell the user you are delegating "
                "and they should wait a moment."
            ),
            runtime_guardrail=(
                "`delegate_task` is mandatory whenever you are not fully certain you can execute the request end-to-end by yourself. "
                "Do not respond with capability disclaimers (for example 'I cannot access', 'I can't do that directly') when delegation is possible. "
                "Before calling `delegate_task`, ask the user to wait briefly; after tool result, provide the final answer naturally."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task or question to delegate.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional extra context to improve the delegated response.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session id for correlation.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Preferred response language.",
                    },
                },
                "required": ["task"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Delegate task/context to OpenClaw and return normalized result."""
        task = str(arguments.get("task", "")).strip()
        if not task:
            return ToolExecutionResult(
                output={"ok": False, "message": "Missing required argument: task"}
            )

        context = str(arguments.get("context", "")).strip()
        session_id = str(arguments.get("session_id", "")).strip() or None
        language = str(arguments.get("language", "")).strip() or self._default_language

        try:
            text = self._client.delegate(
                task=task,
                context=context,
                session_id=session_id,
                language=language,
            )
        except Exception as exc:
            return ToolExecutionResult(
                output={
                    "ok": False,
                    "delegated": True,
                    "message": f"Open Claw delegation failed: {exc}",
                }
            )

        return ToolExecutionResult(
            output={
                "ok": True,
                "delegated": True,
                "source": "openclaw",
                "openclaw_text": text,
            }
        )
