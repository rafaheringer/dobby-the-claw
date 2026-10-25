"""Speaker and microphone volume, and where sound comes from, through the Reachy daemon's REST API.

The same endpoints the official Reachy Mini app uses (/api/volume/*, /api/state/doa), so the page's
sliders behave like the app's: setting the speaker plays the daemon's test sound.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_TIMEOUT_S = 2.0
_PATHS = {"speaker": "/api/volume", "mic": "/api/volume/microphone"}


class DaemonAudio:
    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        self.volumes: dict[str, dict | None] = {"speaker": None, "mic": None}

    async def _json(self, method: str, path: str, body: dict | None = None) -> dict | None:
        # Plain blocking HTTP on a worker thread, not aiohttp: on Windows the daemon resets its
        # connections and asyncio's proactor then logs an error for every one of them.
        return await asyncio.get_running_loop().run_in_executor(
            None, self._request, method, path, body
        )

    def _request(self, method: str, path: str, body: dict | None) -> dict | None:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self._base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
                parsed = json.loads(response.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            logger.debug("Daemon %s %s failed: %s", method, path, exc)
            return None
        return parsed if isinstance(parsed, dict) else None

    @staticmethod
    def _volume(data: dict | None) -> dict | None:
        if data is None or not isinstance(data.get("volume"), int):
            return None
        return {
            "volume": data["volume"],
            "device": str(data.get("device") or ""),
            "platform": str(data.get("platform") or ""),
        }

    async def refresh(self) -> dict[str, dict | None]:
        for kind, path in _PATHS.items():
            self.volumes[kind] = self._volume(await self._json("GET", path + "/current"))
        return self.volumes

    async def set_volume(self, kind: str, volume: int) -> str | None:
        """Set one device's volume (0-100); returns an error text for the page, or None."""
        if kind not in _PATHS:
            return "dispositivo inválido"
        data = self._volume(
            await self._json("POST", _PATHS[kind] + "/set", {"volume": max(0, min(100, volume))})
        )
        if data is None:
            await self.refresh()  # the page snaps back to what the daemon really has
            return "não foi possível mudar o volume"
        self.volumes[kind] = data
        return None

    async def doa(self) -> dict | None:
        """Direction of arrival: angle in radians (0 left, pi/2 front, pi right) and whether it hears speech."""
        data = await self._json("GET", "/api/state/doa")
        if data is None or not isinstance(data.get("angle"), (int, float)):
            return None
        return {
            "angle": round(float(data["angle"]), 3),
            "speech": bool(data.get("speech_detected")),
        }
