"""Tool runtime registry and protocol contracts for execution."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult


class ToolHandler(Protocol):
    """Protocol implemented by concrete runtime tools."""

    def definition(self) -> ToolDefinition:
        """Return static tool definition metadata and parameter schema."""
        ...

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Execute a tool call and return normalized output payload."""
        ...


class ToolRegistry:
    """In-memory registry that maps tool names to tool handlers."""

    def __init__(self) -> None:
        """Initialize empty registry storage."""
        self._tools: Dict[str, ToolHandler] = {}

    def register(self, tool: ToolHandler) -> None:
        """Register one tool handler keyed by its declared name."""
        spec = tool.definition()
        self._tools[spec.name] = tool

    def names(self) -> List[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def openai_specs(self) -> List[Dict[str, Any]]:
        """Build OpenAI function-call specs from registered tool definitions."""
        specs: List[Dict[str, Any]] = []
        for tool in self._tools.values():
            definition = tool.definition()
            specs.append(
                {
                    "type": "function",
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.parameters,
                }
            )
        return specs

    def runtime_guardrails(self) -> List[str]:
        """Collect runtime guardrail text from registered tools."""
        guardrails: List[str] = []
        for tool in self._tools.values():
            definition = tool.definition()
            runtime_guardrail = definition.runtime_guardrail
            if not isinstance(runtime_guardrail, str):
                continue
            text = runtime_guardrail.strip()
            if not text:
                continue
            guardrails.append(f"[{definition.name}] {text}")
        return guardrails

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Execute named tool or return a normalized unknown-tool error."""
        if name not in self._tools:
            return ToolExecutionResult(output={"ok": False, "error": f"Unknown tool: {name}"})
        return self._tools[name].execute(arguments)
