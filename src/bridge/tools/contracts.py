"""Typed contracts for tool definitions and execution results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ToolDefinition:
    """Declarative definition for one callable runtime tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    runtime_guardrail: str | None = None


@dataclass(frozen=True)
class ToolExecutionResult:
    """Normalized tool execution output and optional image payload."""
    output: Dict[str, Any]
    image_base64: str | None = None


@dataclass(frozen=True)
class ToolSet:
    """Collection of tool definitions used for registration or grouping."""
    definitions: List[ToolDefinition]
