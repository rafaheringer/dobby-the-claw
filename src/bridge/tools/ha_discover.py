"""Tool for discovering Home Assistant entities and service capabilities."""

from __future__ import annotations

from typing import Any, Dict

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult
from homeassistant.home_assistant_client import HomeAssistantWsClient


class HomeAssistantDiscoverTool:
    """Expose Home Assistant discovery as an assistant callable tool."""

    def __init__(self, client: HomeAssistantWsClient) -> None:
        """Store Home Assistant websocket client for discovery calls."""
        self._client = client

    def definition(self) -> ToolDefinition:
        """Return OpenAI function schema for capability discovery."""
        return ToolDefinition(
            name="discover_home_devices",
            description=(
                "Fetch current state and detailed attributes for Home Assistant entities. "
                "Use when you need up-to-date state, specific attributes, or available service schemas."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of device domains to filter (e.g. light, fan, switch).",
                    },
                    "include_attributes": {
                        "type": "boolean",
                        "description": "When true, include full entity attributes in discovery results.",
                    },
                    "include_services": {
                        "type": "boolean",
                        "description": "When true, include domain service schemas and fields.",
                    },
                    "max_entities": {
                        "type": "integer",
                        "description": "Maximum number of entities to return (1-500).",
                        "minimum": 1,
                        "maximum": 500,
                    },
                },
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        """Discover entities/services with optional domain and payload controls."""
        domains_raw = arguments.get("domains")
        domains: list[str] = []
        if isinstance(domains_raw, list):
            for item in domains_raw:
                value = str(item).strip().lower()
                if value:
                    domains.append(value)

        include_attributes = bool(arguments.get("include_attributes", False))
        include_services = bool(arguments.get("include_services", True))
        max_entities = int(arguments.get("max_entities", 150))

        try:
            result = self._client.discover_capabilities(
                domains=domains,
                include_attributes=include_attributes,
                include_services=include_services,
                max_entities=max_entities,
            )
        except Exception as exc:
            return ToolExecutionResult(
                output={"ok": False, "message": f"Home Assistant discovery failed: {exc}"}
            )

        return ToolExecutionResult(
            output={
                "ok": True,
                "message": "Home device discovery completed",
                **result,
            }
        )
