from __future__ import annotations

from typing import Any, Dict

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult
from homeassistant.home_assistant_client import HomeAssistantWsClient


class HomeAssistantExecuteActionTool:
    def __init__(
        self,
        client: HomeAssistantWsClient,
        *,
        sensitive_domains: tuple[str, ...],
    ) -> None:
        self._client = client
        self._sensitive_domains = {item.strip().lower() for item in sensitive_domains if item.strip()}

    def definition(self) -> ToolDefinition:
        sensitive_domain_list = ", ".join(sorted(self._sensitive_domains)) or "none"
        return ToolDefinition(
            name="control_home_device",
            description=(
                "Control a home device by executing an action on a selected domain/entity, such as turning a "
                "light on, setting fan speed, or running another available device action."
            ),
            runtime_guardrail=(
                "Before calling `control_home_device`, prefer using `discover_home_devices` when uncertain about "
                "entity_id, domain or service. For sensitive domains, require explicit confirmation from the user. "
                f"Sensitive domains: {sensitive_domain_list}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "Device domain (e.g. light, fan, switch, lock).",
                    },
                    "service": {
                        "type": "string",
                        "description": "Service name to execute (e.g. turn_on, turn_off, set_percentage).",
                    },
                    "target": {
                        "type": "object",
                        "description": (
                            "Target selector with fields such as entity_id, device_id, area_id. "
                            "entity_id can be string or array."
                        ),
                    },
                    "service_data": {
                        "type": "object",
                        "description": "Optional service parameters.",
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "Set true only after explicit user confirmation for sensitive actions.",
                    },
                    "return_response": {
                        "type": "boolean",
                        "description": "When true, asks Home Assistant to return detailed response payload.",
                    },
                },
                "required": ["domain", "service"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        domain = str(arguments.get("domain", "")).strip().lower()
        service = str(arguments.get("service", "")).strip().lower()
        if not domain or not service:
            return ToolExecutionResult(
                output={"ok": False, "message": "Missing required arguments: domain and service"}
            )

        confirmed = bool(arguments.get("confirmed", False))
        if domain in self._sensitive_domains and not confirmed:
            return ToolExecutionResult(
                output={
                    "ok": False,
                    "requires_confirmation": True,
                    "message": (
                        f"Action '{domain}.{service}' requires explicit confirmation. "
                        "Ask the user and retry with confirmed=true."
                    ),
                }
            )

        target_raw = arguments.get("target", {})
        target = target_raw if isinstance(target_raw, dict) else {}
        service_data_raw = arguments.get("service_data", {})
        service_data = service_data_raw if isinstance(service_data_raw, dict) else {}
        return_response = bool(arguments.get("return_response", False))

        try:
            response = self._client.call_service(
                domain=domain,
                service=service,
                target=target,
                service_data=service_data,
                return_response=return_response,
            )
        except Exception as exc:
            return ToolExecutionResult(
                output={
                    "ok": False,
                    "message": f"Home Assistant action failed: {exc}",
                    "domain": domain,
                    "service": service,
                }
            )

        return ToolExecutionResult(
            output={
                "ok": True,
                "message": "Home device action executed",
                "domain": domain,
                "service": service,
                "target": target,
                "response": response,
            }
        )
