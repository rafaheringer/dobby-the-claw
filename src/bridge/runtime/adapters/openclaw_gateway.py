"""OpenClaw Gateway websocket client adapter."""

from __future__ import annotations

import asyncio
import json
import platform
import time
import uuid
from dataclasses import dataclass
from typing import Any

import websockets

OPENCLAW_PROTOCOL_VERSION = 3
OPENCLAW_CLIENT_ID = "gateway-client"
OPENCLAW_CLIENT_MODE = "backend"
OPENCLAW_ROLE = "operator"
OPENCLAW_SCOPES = ["operator.admin"]


@dataclass(frozen=True)
class OpenClawGatewayConfig:
    """Connection parameters for the OpenClaw Gateway websocket client."""
    ws_url: str
    bearer_token: str
    timeout_s: float
    default_language: str
    delegate_model: str = "anthropic/claude-sonnet-4-6"


class OpenClawGatewayClient:
    """Websocket-only RPC client to delegate tasks to OpenClaw Gateway."""

    def __init__(self, config: OpenClawGatewayConfig) -> None:
        """Initialize the gateway client with static connection configuration."""
        self._config = config

    def delegate(
        self,
        *,
        task: str,
        context: str = "",
        session_id: str | None = None,
        language: str | None = None,
    ) -> str:
        """Delegate a task synchronously through the OpenClaw Gateway."""
        return asyncio.run(
            self._delegate_async(
                task=task,
                context=context,
                session_id=session_id,
                language=language,
            )
        )

    def schedule_reminder(
        self,
        *,
        message: str,
        callback_url: str,
        job_name: str = "dobby-reminder",
        delay_seconds: float | None = None,
        repeat_every_seconds: float | None = None,
        cron_expression: str | None = None,
        timezone: str | None = None,
    ) -> str:
        """Schedule a reminder cron job and return its job ID."""
        return asyncio.run(
            self._schedule_reminder_async(
                message=message,
                callback_url=callback_url,
                job_name=job_name,
                delay_seconds=delay_seconds,
                repeat_every_seconds=repeat_every_seconds,
                cron_expression=cron_expression,
                timezone=timezone,
            )
        )

    def cancel_reminder(self, job_id: str) -> None:
        """Remove a previously scheduled cron job by ID."""
        asyncio.run(self._cancel_reminder_async(job_id))

    async def _schedule_reminder_async(
        self,
        *,
        message: str,
        callback_url: str,
        job_name: str,
        delay_seconds: float | None,
        repeat_every_seconds: float | None,
        cron_expression: str | None,
        timezone: str | None,
    ) -> str:
        """Create a cron.add job with webhook delivery. Returns the job ID."""
        import datetime as _dt

        if delay_seconds is not None:
            fire_at = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=delay_seconds)
            schedule: dict[str, Any] = {"kind": "at", "at": fire_at.strftime("%Y-%m-%dT%H:%M:%SZ")}
            delete_after_run = True
        elif repeat_every_seconds is not None:
            schedule = {"kind": "every", "everyMs": int(repeat_every_seconds * 1000)}
            delete_after_run = False
        elif cron_expression is not None:
            schedule = {"kind": "cron", "expr": cron_expression.strip()}
            if timezone:
                schedule["tz"] = timezone.strip()
            delete_after_run = False
        else:
            raise ValueError("One of delay_seconds, repeat_every_seconds, or cron_expression is required")

        async with websockets.connect(
            self._config.ws_url,
            open_timeout=self._config.timeout_s,
            close_timeout=2.0,
            ping_interval=20.0,
            ping_timeout=20.0,
        ) as connection:
            await self._perform_connect_handshake(connection)
            payload = await self._rpc_request(
                connection,
                method="cron.add",
                params={
                    "name": job_name,
                    "sessionTarget": "main",
                    "wakeMode": "now",
                    "deleteAfterRun": delete_after_run,
                    "schedule": schedule,
                    "payload": {"kind": "systemEvent", "text": message},
                    "delivery": {"mode": "webhook", "to": callback_url},
                },
                expect_final=False,
            )

        if not isinstance(payload, dict):
            raise RuntimeError("cron.add returned unexpected payload")
        job_id = str(payload.get("id", "")).strip()
        if not job_id:
            raise RuntimeError("cron.add returned no job ID")
        return job_id

    async def _cancel_reminder_async(self, job_id: str) -> None:
        """Remove a cron job via cron.remove RPC."""
        async with websockets.connect(
            self._config.ws_url,
            open_timeout=self._config.timeout_s,
            close_timeout=2.0,
            ping_interval=20.0,
            ping_timeout=20.0,
        ) as connection:
            await self._perform_connect_handshake(connection)
            await self._rpc_request(
                connection,
                method="cron.remove",
                params={"id": job_id},
                expect_final=False,
            )

    async def _delegate_async(
        self,
        *,
        task: str,
        context: str,
        session_id: str | None,
        language: str | None,
    ) -> str:
        """Send task/context to gateway and wait for assistant text completion."""
        session_key = (session_id or "agent:main:main").strip() or "agent:main:main"
        prompt = task.strip()
        if context.strip():
            prompt = f"{task.strip()}\n\nContexto adicional:\n{context.strip()}"
        selected_language = (language or self._config.default_language or "pt").strip() or "pt"
        prompt = f"[Responder em: {selected_language}]\n{prompt}"

        async with websockets.connect(
            self._config.ws_url,
            open_timeout=self._config.timeout_s,
            close_timeout=2.0,
            ping_interval=20.0,
            ping_timeout=20.0,
        ) as connection:
            await self._perform_connect_handshake(connection)

            start_ms = int(time.time() * 1000)
            send_payload = await self._rpc_request(
                connection,
                method="chat.send",
                params={
                    "sessionKey": session_key,
                    "idempotencyKey": f"bridge-{uuid.uuid4()}",
                    "message": prompt,
                    "model": self._config.delegate_model,
                },
                expect_final=True,
            )
            if not isinstance(send_payload, dict):
                raise RuntimeError("Invalid chat.send response payload")

            deadline = time.monotonic() + self._config.timeout_s
            while time.monotonic() < deadline:
                history = await self._rpc_request(
                    connection,
                    method="chat.history",
                    params={"sessionKey": session_key, "limit": 20},
                    expect_final=False,
                )
                if not isinstance(history, dict):
                    raise RuntimeError("Invalid chat.history response payload")

                text = self._extract_latest_assistant_text(history=history, min_timestamp_ms=start_ms)
                if text:
                    return text
                await asyncio.sleep(0.5)

            raise TimeoutError("Timed out waiting for OpenClaw response")

    async def _perform_connect_handshake(self, connection: Any) -> None:
        """Complete gateway connect challenge and protocol handshake."""
        _ = await self._wait_for_connect_challenge(connection)
        connect_params = {
            "minProtocol": OPENCLAW_PROTOCOL_VERSION,
            "maxProtocol": OPENCLAW_PROTOCOL_VERSION,
            "client": {
                "id": OPENCLAW_CLIENT_ID,
                "displayName": "dobby-bridge",
                "version": "0.0.1",
                "platform": platform.system().lower(),
                "mode": OPENCLAW_CLIENT_MODE,
                "instanceId": f"dobby-bridge-{uuid.uuid4()}",
            },
            "caps": [],
            "role": OPENCLAW_ROLE,
            "scopes": OPENCLAW_SCOPES,
            "auth": {
                "token": self._config.bearer_token,
            }
            if self._config.bearer_token
            else None,
        }
        if connect_params["auth"] is None:
            connect_params.pop("auth")

        _ = await self._rpc_request(
            connection,
            method="connect",
            params=connect_params,
            expect_final=False,
        )

    async def _wait_for_connect_challenge(self, connection: Any) -> str:
        """Wait for and validate the initial `connect.challenge` event."""
        deadline = time.monotonic() + min(self._config.timeout_s, 10.0)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("OpenClaw connect challenge timeout")

            frame = await self._recv_frame(connection, timeout_s=remaining)
            if frame.get("type") != "event":
                continue
            if str(frame.get("event", "")).strip() != "connect.challenge":
                continue

            payload = frame.get("payload")
            if not isinstance(payload, dict):
                raise RuntimeError("Invalid connect.challenge payload")
            nonce = payload.get("nonce")
            if not isinstance(nonce, str) or not nonce.strip():
                raise RuntimeError("OpenClaw connect challenge missing nonce")
            return nonce.strip()

    async def _rpc_request(
        self,
        connection: Any,
        *,
        method: str,
        params: dict[str, Any],
        expect_final: bool,
    ) -> Any:
        """Send one RPC request frame and return the validated response payload."""
        request_id = str(uuid.uuid4())
        frame = {
            "type": "req",
            "id": request_id,
            "method": method,
            "params": params,
        }
        await connection.send(json.dumps(frame))

        deadline = time.monotonic() + self._config.timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timeout waiting response for method={method}")

            incoming = await self._recv_frame(connection, timeout_s=remaining)
            incoming_type = str(incoming.get("type", "")).strip()
            if incoming_type == "event":
                continue
            if incoming_type != "res":
                continue
            if str(incoming.get("id", "")).strip() != request_id:
                continue

            ok = bool(incoming.get("ok"))
            if not ok:
                error = incoming.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message", "unknown gateway error")).strip()
                else:
                    message = "unknown gateway error"
                raise RuntimeError(message)

            payload = incoming.get("payload")
            if expect_final and isinstance(payload, dict) and str(payload.get("status", "")).strip() == "accepted":
                continue
            return payload

    @staticmethod
    async def _recv_frame(connection: Any, *, timeout_s: float) -> dict[str, Any]:
        """Receive one gateway frame and parse it as a JSON object."""
        raw = await asyncio.wait_for(connection.recv(), timeout=timeout_s)
        if not isinstance(raw, str):
            raise RuntimeError("OpenClaw frame is not text")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid OpenClaw JSON frame: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("OpenClaw frame must be a JSON object")
        return parsed

    @staticmethod
    def _extract_latest_assistant_text(*, history: dict[str, Any], min_timestamp_ms: int) -> str:
        """Extract newest assistant text blocks from chat history payload."""
        messages = history.get("messages")
        if not isinstance(messages, list):
            return ""

        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).strip().lower() != "assistant":
                continue
            timestamp = message.get("timestamp")
            if isinstance(timestamp, (int, float)) and int(timestamp) < min_timestamp_ms:
                continue

            content = message.get("content")
            if not isinstance(content, list):
                continue

            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if str(block.get("type", "")).strip().lower() != "text":
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            if parts:
                return "\n".join(parts)

        return ""
