from typing import Any, Dict


class OpenClawClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def send_text(self, text: str, session_id: str) -> Dict[str, Any]:
        # TODO: Implement HTTP/WebSocket client to OpenClaw API.
        raise NotImplementedError("OpenClaw client not implemented yet")
