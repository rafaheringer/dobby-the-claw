from __future__ import annotations

from typing import Any, Dict, List, Protocol

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult


class ToolHandler(Protocol):
    def definition(self) -> ToolDefinition:
        ...

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolHandler] = {}

    def register(self, tool: ToolHandler) -> None:
        spec = tool.definition()
        self._tools[spec.name] = tool

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def openai_specs(self) -> List[Dict[str, Any]]:
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

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        if name not in self._tools:
            return ToolExecutionResult(output={"ok": False, "error": f"Unknown tool: {name}"})
        return self._tools[name].execute(arguments)
