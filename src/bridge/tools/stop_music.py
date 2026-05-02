"""Tool for stopping music on Alexa media players via Home Assistant."""

from __future__ import annotations

from typing import Any, Dict

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult
from homeassistant.home_assistant_client import HomeAssistantWsClient

_ALEXA_ENTITIES: list[str] = [
    "media_player.alexa_do_escritorio",
    "media_player.alexa_da_sala",
    "media_player.alexa_do_quarto",
]

_ROOM_MAP: dict[str, str] = {
    "escritório": "media_player.alexa_do_escritorio",
    "escritorio": "media_player.alexa_do_escritorio",
    "sala": "media_player.alexa_da_sala",
    "quarto": "media_player.alexa_do_quarto",
}

_ACTIVE_STATES = {"playing", "paused"}


class StopMusicTool:
    def __init__(self, client: HomeAssistantWsClient) -> None:
        self._client = client

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="stop_music",
            description=(
                "Para a música tocando nas caixas Alexa. Detecta automaticamente qual está "
                "tocando e para. Use quando o usuário pedir para parar, pausar ou silenciar a música."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "room": {
                        "type": "string",
                        "enum": ["escritório", "sala", "quarto"],
                        "description": (
                            "Cômodo específico para parar. Omitir detecta automaticamente "
                            "qual Alexa está tocando."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        room = (arguments.get("room") or "").strip().lower()

        if room:
            entity_id = _ROOM_MAP.get(room)
            if not entity_id:
                return ToolExecutionResult(
                    output={"ok": False, "error": f"Cômodo desconhecido: {room}"}
                )
            return self._stop_one(entity_id, room)

        # Auto-detect: query current states of all Alexas
        try:
            catalog = self._client.get_catalog(domains=["media_player"])
        except Exception as exc:
            return ToolExecutionResult(
                output={"ok": False, "error": f"Falha ao consultar dispositivos: {exc}"}
            )

        alexa_set = set(_ALEXA_ENTITIES)
        active = [e for e in catalog if e["entity_id"] in alexa_set and e.get("state") in _ACTIVE_STATES]

        if not active:
            return ToolExecutionResult(
                output={"ok": False, "nothing_playing": True, "message": "Nenhuma Alexa está tocando música."}
            )

        stopped, errors = [], []
        for entity in active:
            try:
                self._client.call_service(
                    domain="media_player",
                    service="media_stop",
                    target={"entity_id": entity["entity_id"]},
                    service_data={},
                    return_response=False,
                )
                stopped.append(entity.get("friendly_name") or entity["entity_id"])
            except Exception as exc:
                errors.append(str(exc))

        return ToolExecutionResult(
            output={
                "ok": len(stopped) > 0,
                "stopped": stopped,
                **({"errors": errors} if errors else {}),
            }
        )

    def _stop_one(self, entity_id: str, room_label: str) -> ToolExecutionResult:
        try:
            self._client.call_service(
                domain="media_player",
                service="media_stop",
                target={"entity_id": entity_id},
                service_data={},
                return_response=False,
            )
        except Exception as exc:
            return ToolExecutionResult(output={"ok": False, "error": str(exc)})
        return ToolExecutionResult(output={"ok": True, "stopped": [room_label]})
