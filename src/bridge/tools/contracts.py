from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass(frozen=True)
class ToolExecutionResult:
    output: Dict[str, Any]
    image_base64: str | None = None


@dataclass(frozen=True)
class ToolSet:
    definitions: List[ToolDefinition]
