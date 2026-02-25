from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


SYSTEM_PROMPT = (
    "You are Reachy, a helpful physical robot. "
    "Keep replies concise. Use light sarcasm for normal responses, "
    "minimal sarcasm for safety-critical topics, and be clear in errors."
)


class DirectLLMClient:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        timeout_s: int = 20,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def generate_reply(
        self,
        text: str,
        session_id: str,
        language: Optional[str] = None,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if language:
            text = f"[Language: {language}]\n{text}"
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": text}]

        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "metadata": {"session_id": session_id},
        }

        response = requests.post(
            f"{self.api_base}/responses",
            headers=headers,
            json=payload,
            timeout=self.timeout_s,
        )

        if not response.ok:
            raise RuntimeError(
                f"LLM request failed: {response.status_code} {response.text}"
            )

        data = response.json()
        return _extract_output_text(data)


def _extract_output_text(data: Dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    outputs = data.get("output", [])
    if isinstance(outputs, list):
        for item in outputs:
            content = item.get("content", []) if isinstance(item, dict) else []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "output_text":
                    text = block.get("text", "")
                    if isinstance(text, str) and text.strip():
                        return text.strip()

    raise RuntimeError("LLM response missing output text")
