"""Tool runtime adapter backed by ToolRegistry."""

from __future__ import annotations

from typing import Any, Optional

from bridge.config import BridgeConfig
from reachy.camera_worker import CameraWorker
from reachy.motion import MotionManager

from bridge.runtime.audio_support import build_tool_registry
from bridge.runtime.ports import ToolRuntimePort
from bridge.tools import ToolRegistry


class ToolRegistryRuntime(ToolRuntimePort):
    """Expose ToolRegistry through runtime tool port interface."""

    def __init__(self, registry: ToolRegistry) -> None:
        """Wrap a tool registry behind the runtime port contract."""
        self._registry = registry

    def names(self) -> list[str]:
        """Return registered tool names."""
        return self._registry.names()

    def openai_specs(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function specs for registered tools."""
        return self._registry.openai_specs()

    def runtime_guardrails(self) -> list[str]:
        """Return runtime guardrail statements aggregated from tools."""
        return self._registry.runtime_guardrails()

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool by name with parsed arguments."""
        return self._registry.execute(name, arguments)


def build_tool_runtime(
    config: BridgeConfig,
    camera_worker: Optional[CameraWorker],
    motion_manager: Optional[MotionManager],
    speaker_memory=None,
) -> ToolRuntimePort:
    """Build default tool runtime adapter from runtime config."""
    return ToolRegistryRuntime(build_tool_registry(config, camera_worker, motion_manager, speaker_memory=speaker_memory))
