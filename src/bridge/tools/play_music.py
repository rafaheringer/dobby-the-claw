"""Tool for playing music via Alexa through a Home Assistant script."""

from __future__ import annotations

from typing import Any, Dict

from bridge.tools.contracts import ToolDefinition, ToolExecutionResult
from homeassistant.home_assistant_client import HomeAssistantWsClient

_ROOM_MAP: dict[str, str] = {
    "escritório": "media_player.alexa_do_escritorio",
    "escritorio": "media_player.alexa_do_escritorio",
    "sala": "media_player.alexa_da_sala",
    "quarto": "media_player.alexa_do_quarto",
}


class PlayMusicTool:
    def __init__(self, client: HomeAssistantWsClient) -> None:
        self._client = client

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="play_music",
            description=(
                "Toca música no Spotify via Alexa. Use para pedidos como tocar um artista, "
                "música, álbum, playlist ou gênero musical."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "what_to_play": {
                        "type": "string",
                        "description": (
                            "O que tocar: artista, música, álbum, playlist ou gênero. "
                            "Ex: 'Pitty', 'rock dos anos 80', 'playlist de jazz'."
                        ),
                    },
                    "room": {
                        "type": "string",
                        "enum": ["escritório", "sala", "quarto"],
                        "description": "Cômodo onde tocar. Padrão: escritório.",
                    },
                },
                "required": ["what_to_play"],
                "additionalProperties": False,
            },
        )

    def execute(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        what = (arguments.get("what_to_play") or "").strip()
        if not what:
            return ToolExecutionResult(output={"ok": False, "error": "Nenhuma música especificada."})

        room = (arguments.get("room") or "").strip().lower()
        service_data: dict[str, Any] = {"what_to_play": what}
        if room:
            entity_id = _ROOM_MAP.get(room)
            if entity_id:
                service_data["where_to_play"] = entity_id

        try:
            self._client.call_service(
                domain="script",
                service="play_music",
                target={},
                service_data=service_data,
                return_response=False,
            )
        except Exception as exc:
            return ToolExecutionResult(output={"ok": False, "error": str(exc)})

        return ToolExecutionResult(
            output={"ok": True, "playing": what, "room": room or "escritório"}
        )
