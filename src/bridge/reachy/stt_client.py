"""OpenAI STT client for transcribing captured microphone audio."""

import io
from typing import Optional

import requests


class OpenAISttClient:
    """Client wrapper around OpenAI audio transcription endpoint."""

    def __init__(self, api_base: str, api_key: str, model: str, timeout_s: int = 20) -> None:
        """Initialize STT client configuration."""
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def transcribe(self, wav_bytes: bytes, language: Optional[str] = None) -> str:
        """Transcribe WAV audio bytes to text using OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model,
        }
        if language:
            data["language"] = language

        files = {
            "file": ("speech.wav", io.BytesIO(wav_bytes), "audio/wav"),
        }

        response = requests.post(
            f"{self.api_base}/audio/transcriptions",
            headers=headers,
            data=data,
            files=files,
            timeout=self.timeout_s,
        )
        if not response.ok:
            raise RuntimeError(f"STT request failed: {response.status_code} {response.text}")

        payload = response.json()
        text = str(payload.get("text", "")).strip()
        return text
