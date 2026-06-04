"""Runtime adapter implementations for concrete infrastructure integrations."""

from bridge.runtime.adapters.openclaw_gateway import OpenClawGatewayClient, OpenClawGatewayConfig
from bridge.runtime.adapters.reachy_actions import ReachyRobotActions
from bridge.runtime.adapters.reachy_media import ReachySdkMediaIO, build_reachy_media_io
from bridge.runtime.adapters.realtime_session import OpenAIRealtimeSessionFactory
from bridge.runtime.adapters.tool_runtime import ToolRegistryRuntime, build_tool_runtime

__all__ = [
    "OpenClawGatewayClient",
    "OpenClawGatewayConfig",
    "ReachyRobotActions",
    "ReachySdkMediaIO",
    "build_reachy_media_io",
    "OpenAIRealtimeSessionFactory",
    "ToolRegistryRuntime",
    "build_tool_runtime",
]
