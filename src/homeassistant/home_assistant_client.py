"""Async Home Assistant websocket client with synchronous helper entrypoints."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import websockets


@dataclass(frozen=True)
class HomeAssistantWsConfig:
    """Connection settings for the Home Assistant websocket API."""
    ws_url: str
    access_token: str
    timeout_s: float


class HomeAssistantWsClient:
    """High-level Home Assistant websocket client for discovery and control."""

    def __init__(self, config: HomeAssistantWsConfig) -> None:
        """Store immutable websocket connection configuration."""
        self._config = config

    def discover_capabilities(
        self,
        *,
        domains: list[str] | None,
        include_attributes: bool,
        include_services: bool,
        max_entities: int,
    ) -> dict[str, Any]:
        """Synchronously discover entities and optional domain services."""
        return asyncio.run(
            self._discover_capabilities_async(
                domains=domains,
                include_attributes=include_attributes,
                include_services=include_services,
                max_entities=max_entities,
            )
        )

    def get_catalog(self, domains: list[str] | None = None) -> list[dict[str, Any]]:
        """Return compact entity list (entity_id, friendly_name, state, domain)."""
        return asyncio.run(self._get_catalog_async(domains=domains))

    async def _get_catalog_async(self, domains: list[str] | None) -> list[dict[str, Any]]:
        """Fetch states and return a flat catalog of entities."""
        async with self._connect() as connection:
            response = await self._send_command(connection, {"type": "get_states"})
            states = response.get("result", [])
            normalized_domains = {d.strip().lower() for d in (domains or []) if d.strip()}
            catalog = []
            for state in states:
                if not isinstance(state, dict):
                    continue
                entity_id = str(state.get("entity_id", "")).strip()
                if not entity_id or "." not in entity_id:
                    continue
                domain = entity_id.split(".", 1)[0].lower()
                if normalized_domains and domain not in normalized_domains:
                    continue
                attributes = state.get("attributes") or {}
                catalog.append({
                    "entity_id": entity_id,
                    "domain": domain,
                    "friendly_name": attributes.get("friendly_name") or entity_id,
                    "state": state.get("state", "unknown"),
                })
            return catalog

    def call_service(
        self,
        *,
        domain: str,
        service: str,
        target: dict[str, Any],
        service_data: dict[str, Any],
        return_response: bool,
    ) -> dict[str, Any]:
        """Synchronously call one Home Assistant domain service."""
        return asyncio.run(
            self._call_service_async(
                domain=domain,
                service=service,
                target=target,
                service_data=service_data,
                return_response=return_response,
            )
        )

    async def _discover_capabilities_async(
        self,
        *,
        domains: list[str] | None,
        include_attributes: bool,
        include_services: bool,
        max_entities: int,
    ) -> dict[str, Any]:
        """Fetch states/services and build filtered discovery payload."""
        async with self._connect() as connection:
            response_states = await self._send_command(connection, {"type": "get_states"})
            states = response_states.get("result")
            if not isinstance(states, list):
                raise RuntimeError("Home Assistant get_states returned invalid payload")

            normalized_domains = {item.strip().lower() for item in (domains or []) if item.strip()}

            entities: list[dict[str, Any]] = []
            for state in states:
                if not isinstance(state, dict):
                    continue
                entity_id = str(state.get("entity_id", "")).strip()
                if not entity_id or "." not in entity_id:
                    continue
                domain = entity_id.split(".", 1)[0].strip().lower()
                if normalized_domains and domain not in normalized_domains:
                    continue

                attributes = state.get("attributes")
                if not isinstance(attributes, dict):
                    attributes = {}

                entry: dict[str, Any] = {
                    "entity_id": entity_id,
                    "domain": domain,
                    "state": state.get("state"),
                    "friendly_name": attributes.get("friendly_name"),
                }
                if include_attributes:
                    entry["attributes"] = attributes
                entities.append(entry)

            entities = entities[: max(1, min(max_entities, 500))]

            services_payload: dict[str, Any] = {}
            if include_services:
                response_services = await self._send_command(connection, {"type": "get_services"})
                services = response_services.get("result")
                if not isinstance(services, dict):
                    raise RuntimeError("Home Assistant get_services returned invalid payload")

                for domain, services_for_domain in services.items():
                    domain_name = str(domain).strip().lower()
                    if normalized_domains and domain_name not in normalized_domains:
                        continue
                    if isinstance(services_for_domain, dict):
                        services_payload[domain_name] = services_for_domain

            return {
                "entities": entities,
                "services": services_payload,
                "total_entities": len(entities),
                "total_service_domains": len(services_payload),
            }

    async def _call_service_async(
        self,
        *,
        domain: str,
        service: str,
        target: dict[str, Any],
        service_data: dict[str, Any],
        return_response: bool,
    ) -> dict[str, Any]:
        """Execute one `call_service` command and normalize response shape."""
        async with self._connect() as connection:
            command: dict[str, Any] = {
                "type": "call_service",
                "domain": domain,
                "service": service,
            }
            if target:
                command["target"] = target
            if service_data:
                command["service_data"] = service_data
            if return_response:
                command["return_response"] = True

            response = await self._send_command(connection, command)
            result = response.get("result")
            if isinstance(result, dict):
                return result
            if isinstance(result, list):
                return {"changed_states": result}
            return {"result": result}

    def _connect(self) -> "_WsSession":
        """Create a websocket session context manager for this configuration."""
        return _WsSession(self._config)

    async def _send_command(self, connection: Any, command: dict[str, Any]) -> dict[str, Any]:
        """Send one command frame and return validated result response."""
        request_id = await connection.next_id()
        payload = {"id": request_id, **command}
        await connection.send(payload)
        response = await connection.wait_response(request_id)
        success = bool(response.get("success"))
        if not success:
            error = response.get("error")
            if isinstance(error, dict):
                message = str(error.get("message", "unknown Home Assistant error")).strip()
            else:
                message = "unknown Home Assistant error"
            raise RuntimeError(message)
        return response


class _WsSession:
    """Context-managed low-level websocket session for Home Assistant RPC."""

    def __init__(self, config: HomeAssistantWsConfig) -> None:
        """Initialize session state and request-id counter."""
        self._config = config
        self._connection: Any | None = None
        self._next_request_id = 1

    async def __aenter__(self) -> "_WsSession":
        """Open websocket connection and complete authentication handshake."""
        self._connection = await websockets.connect(
            self._config.ws_url,
            open_timeout=self._config.timeout_s,
            close_timeout=2.0,
            ping_interval=20.0,
            ping_timeout=20.0,
        )
        await self._perform_auth()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close websocket connection when leaving context manager."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def next_id(self) -> int:
        """Return next monotonically increasing request identifier."""
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    async def send(self, payload: dict[str, Any]) -> None:
        """Send one JSON payload over the active websocket connection."""
        if self._connection is None:
            raise RuntimeError("Home Assistant websocket is not connected")
        await self._connection.send(json.dumps(payload))

    async def wait_response(self, request_id: int) -> dict[str, Any]:
        """Wait until a matching `result` frame is received for request id."""
        deadline = asyncio.get_running_loop().time() + self._config.timeout_s
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting Home Assistant response")

            frame = await self._recv(timeout_s=remaining)
            frame_type = str(frame.get("type", "")).strip().lower()
            if frame_type != "result":
                continue
            frame_id = frame.get("id")
            if isinstance(frame_id, int) and frame_id == request_id:
                return frame

    async def _perform_auth(self) -> None:
        """Authenticate websocket connection using long-lived access token."""
        first = await self._recv(timeout_s=min(self._config.timeout_s, 10.0))
        if str(first.get("type", "")).strip().lower() != "auth_required":
            raise RuntimeError("Home Assistant websocket auth_required not received")

        await self.send({"type": "auth", "access_token": self._config.access_token})
        second = await self._recv(timeout_s=min(self._config.timeout_s, 10.0))
        if str(second.get("type", "")).strip().lower() != "auth_ok":
            message = "Home Assistant authentication failed"
            if isinstance(second.get("message"), str):
                message = second["message"].strip() or message
            raise RuntimeError(message)

    async def _recv(self, *, timeout_s: float) -> dict[str, Any]:
        """Receive and parse one websocket frame as a JSON object."""
        if self._connection is None:
            raise RuntimeError("Home Assistant websocket is not connected")
        raw = await asyncio.wait_for(self._connection.recv(), timeout=timeout_s)
        if not isinstance(raw, str):
            raise RuntimeError("Home Assistant frame is not text")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid Home Assistant JSON frame: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("Home Assistant frame must be a JSON object")
        return parsed
