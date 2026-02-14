from typing import Any, Dict, Optional

import requests


class OpenClawClient:
    def __init__(self, base_url: str, intent_path: str = "/v1/intent", timeout_s: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.intent_path = intent_path
        self.timeout_s = timeout_s

    def send_text(self, text: str, session_id: str, language: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "text": text,
        }
        if language:
            payload["language"] = language

        response = requests.post(
            f"{self.base_url}{self.intent_path}",
            json=payload,
            timeout=self.timeout_s,
        )

        if not response.ok:
            raise RuntimeError(
                f"OpenClaw request failed: {response.status_code} {response.text}"
            )

        return response.json()
