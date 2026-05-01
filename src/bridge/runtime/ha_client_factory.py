"""Factory helper for building a Home Assistant websocket client from config."""

from __future__ import annotations

from bridge.config import BridgeConfig, build_home_assistant_websocket_url
from homeassistant.home_assistant_client import HomeAssistantWsClient, HomeAssistantWsConfig


def build_ha_client(config: BridgeConfig) -> "HomeAssistantWsClient | None":
    """Build a Home Assistant websocket client if HA is enabled, else None."""
    if not config.home_assistant_enabled:
        return None
    if not config.home_assistant_token or not config.home_assistant_url:
        return None
    return HomeAssistantWsClient(
        HomeAssistantWsConfig(
            ws_url=build_home_assistant_websocket_url(config.home_assistant_url),
            access_token=config.home_assistant_token,
            timeout_s=config.home_assistant_timeout_s,
        )
    )
