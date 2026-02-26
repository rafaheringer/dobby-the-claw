"""OpenAI Realtime streaming client for low-latency microphone conversations."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np
from numpy.typing import NDArray
from openai import AsyncOpenAI

from bridge.tools.contracts import ToolExecutionResult


logger = logging.getLogger(__name__)

NOISY_DELTA_EVENTS = {
    "conversation.item.input_audio_transcription.delta",
    "response.output_audio_transcript.delta",
    "response.audio_transcript.delta",
    "response.output_audio.delta",
    "response.audio.delta",
}


class OpenAIRealtimeSession:
    """Maintain a background OpenAI Realtime connection and stream audio frames."""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str,
        instructions: str,
        language: str,
        transcription_model: str,
        vad_silence_ms: int,
        vad_prefix_padding_ms: int,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_user_transcript: Optional[Callable[[str], None]] = None,
        on_assistant_text: Optional[Callable[[str], None]] = None,
        on_assistant_audio_chunk: Optional[Callable[[NDArray[np.float32]], None]] = None,
        on_assistant_audio_done: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        tool_specs: Optional[List[Dict[str, Any]]] = None,
        on_tool_call: Optional[Callable[[str, Dict[str, Any]], ToolExecutionResult]] = None,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.instructions = instructions
        self.language = language
        self.transcription_model = transcription_model
        self.vad_silence_ms = vad_silence_ms
        self.vad_prefix_padding_ms = vad_prefix_padding_ms

        self.on_speech_start = on_speech_start
        self.on_user_transcript = on_user_transcript
        self.on_assistant_text = on_assistant_text
        self.on_assistant_audio_chunk = on_assistant_audio_chunk
        self.on_assistant_audio_done = on_assistant_audio_done
        self.on_error = on_error
        self.tool_specs = tool_specs or []
        self.on_tool_call = on_tool_call

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connection = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._last_assistant_text = ""
        self._last_assistant_ts = 0.0
        self._start_ts = time.monotonic()

    def _elapsed_ms(self) -> int:
        """Return elapsed time since session object creation in milliseconds."""
        return int((time.monotonic() - self._start_ts) * 1000)

    def _safe_call(self, label: str, callback: Optional[Callable], *args) -> None:
        """Run callbacks defensively so one callback failure doesn't kill realtime loop."""
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as exc:
            logger.exception("[%dms] Callback '%s' failed: %s", self._elapsed_ms(), label, exc)

    def start(self) -> None:
        """Start realtime connection in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._run_thread, daemon=True)
        self._thread.start()
        logger.info("[%dms] Realtime thread started", self._elapsed_ms())

    def wait_until_ready(self, timeout_s: float = 8.0) -> bool:
        """Wait until realtime session is connected and updated."""
        return self._ready.wait(timeout=timeout_s)

    def stop(self) -> None:
        """Stop background realtime connection."""
        self._stop.set()
        logger.info("[%dms] Realtime stop requested", self._elapsed_ms())
        if self._loop is not None and self._connection is not None:
            future = asyncio.run_coroutine_threadsafe(self._connection.close(), self._loop)
            try:
                future.result(timeout=2.0)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def feed_audio(self, sample_rate: int, sample: NDArray[np.float32]) -> None:
        """Feed one microphone frame to realtime input buffer."""
        if not self._ready.is_set() or self._loop is None or self._connection is None:
            return

        mono = self._to_float32_mono(sample)
        if mono.size == 0:
            return
        pcm24k = self._resample_to_24k(mono, sample_rate)
        pcm16 = np.clip(pcm24k, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        encoded = base64.b64encode(pcm16.tobytes()).decode("utf-8")

        future = asyncio.run_coroutine_threadsafe(
            self._connection.input_audio_buffer.append(audio=encoded),
            self._loop,
        )
        try:
            future.add_done_callback(lambda f: f.exception())
        except Exception:
            pass

    def _run_thread(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_base)
        logger.info(
            "[%dms] Connecting realtime session model=%s language=%s vad_silence=%sms vad_padding=%sms",
            self._elapsed_ms(),
            self.model,
            self.language,
            self.vad_silence_ms,
            self.vad_prefix_padding_ms,
        )
        try:
            async with client.realtime.connect(model=self.model) as connection:
                self._connection = connection
                session_payload: Dict[str, Any] = {
                    "type": "realtime",
                    "instructions": self.instructions,
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": 24000},
                            "transcription": {
                                "model": self.transcription_model,
                                "language": self.language,
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "interrupt_response": True,
                                "create_response": True,
                                "silence_duration_ms": self.vad_silence_ms,
                                "prefix_padding_ms": self.vad_prefix_padding_ms,
                            },
                        },
                        "output": {
                            "format": {
                                "type": "audio/pcm",
                            },
                            "voice": "ballad",
                        },
                    },
                }
                if self.tool_specs:
                    session_payload["tools"] = self.tool_specs
                    session_payload["tool_choice"] = "auto"

                await connection.session.update(session=cast(Any, session_payload))
                self._ready.set()
                logger.info("[%dms] Realtime session ready", self._elapsed_ms())

                async for event in connection:
                    if self._stop.is_set():
                        break
                    event_type = getattr(event, "type", "")
                    if event_type not in NOISY_DELTA_EVENTS:
                        logger.debug("[%dms] Realtime event: %s", self._elapsed_ms(), event_type)

                    if event_type == "input_audio_buffer.speech_started":
                        logger.debug("[%dms] speech_started", self._elapsed_ms())
                        self._safe_call("on_speech_start", self.on_speech_start)
                        continue

                    if event_type == "input_audio_buffer.speech_stopped":
                        logger.debug("[%dms] speech_stopped", self._elapsed_ms())
                        continue

                    if event_type == "conversation.item.input_audio_transcription.completed":
                        transcript = str(getattr(event, "transcript", "")).strip()
                        if transcript:
                            logger.info("[%dms] User transcript: %s", self._elapsed_ms(), transcript)
                            self._safe_call("on_user_transcript", self.on_user_transcript, transcript)
                        continue

                    if event_type in {
                        "response.audio_transcript.done",
                        "response.output_audio_transcript.done",
                        "response.output_text.done",
                        "response.text.done",
                    }:
                        text = str(getattr(event, "transcript", "") or getattr(event, "text", "")).strip()
                        self._emit_assistant_text(text)
                        continue

                    if event_type in {"response.audio.delta", "response.output_audio.delta"}:
                        delta = getattr(event, "delta", None)
                        if isinstance(delta, str) and self.on_assistant_audio_chunk is not None:
                            try:
                                pcm16 = np.frombuffer(base64.b64decode(delta), dtype=np.int16)
                                if pcm16.size > 0:
                                    audio = (pcm16.astype(np.float32) / 32767.0).astype(np.float32)
                                    self._safe_call("on_assistant_audio_chunk", self.on_assistant_audio_chunk, audio)
                            except Exception as exc:
                                logger.debug("[%dms] Failed to decode audio delta: %s", self._elapsed_ms(), exc)
                        continue

                    if event_type in {
                        "response.audio.done",
                        "response.output_audio.done",
                        "response.audio.completed",
                    }:
                        logger.debug("[%dms] Assistant audio done", self._elapsed_ms())
                        self._safe_call("on_assistant_audio_done", self.on_assistant_audio_done)
                        continue

                    if event_type == "response.done":
                        response = getattr(event, "response", None)
                        text = self._extract_response_text(response)
                        logger.debug("[%dms] response.done text_len=%s", self._elapsed_ms(), len(text))
                        self._emit_assistant_text(text)
                        continue

                    if event_type == "response.function_call_arguments.done":
                        await self._handle_tool_call_event(event)
                        continue

                    if event_type == "error":
                        err = getattr(event, "error", None)
                        message = str(getattr(err, "message", err or "unknown realtime error"))
                        logger.warning("[%dms] Realtime API error: %s", self._elapsed_ms(), message)
                        self._safe_call("on_error", self.on_error, message)
        except Exception as exc:
            logger.exception("[%dms] Realtime session failed: %s", self._elapsed_ms(), exc)
            self._safe_call("on_error", self.on_error, str(exc))
        finally:
            self._ready.clear()
            self._connection = None
            logger.info("[%dms] Realtime session closed", self._elapsed_ms())

    async def _handle_tool_call_event(self, event: Any) -> None:
        """Execute a realtime function call and feed results back to the model."""
        tool_name = getattr(event, "name", None)
        arguments_raw = getattr(event, "arguments", None)
        call_id = getattr(event, "call_id", None)

        if not isinstance(tool_name, str) or not isinstance(call_id, str):
            logger.warning("[%dms] Invalid tool call event payload", self._elapsed_ms())
            return

        parsed_args: Dict[str, Any] = {}
        if isinstance(arguments_raw, str) and arguments_raw.strip():
            try:
                obj = json.loads(arguments_raw)
                if isinstance(obj, dict):
                    parsed_args = obj
            except json.JSONDecodeError:
                parsed_args = {}

        logger.info("[%dms] Tool call: %s args=%s", self._elapsed_ms(), tool_name, parsed_args)

        if self.on_tool_call is None:
            result = ToolExecutionResult(output={"ok": False, "message": "Tooling not configured"})
        else:
            try:
                result = self.on_tool_call(tool_name, parsed_args)
            except Exception as exc:
                logger.exception("[%dms] Tool execution failed: %s", self._elapsed_ms(), exc)
                result = ToolExecutionResult(output={"ok": False, "message": str(exc)})

        if self._connection is None:
            return

        await self._connection.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(result.output),
            }
        )

        if isinstance(result.image_base64, str) and result.image_base64:
            await self._connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{result.image_base64}",
                        }
                    ],
                }
            )

        await self._connection.response.create(
            response={
                "instructions": "Use tool results (and image when provided) to answer the user naturally.",
            }
        )

    def _extract_response_text(self, response: object) -> str:
        """Extract assistant text from response payload as a fallback path."""
        if response is None:
            return ""
        outputs = getattr(response, "output", None)
        if not isinstance(outputs, list):
            return ""
        for item in outputs:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for block in content:
                block_type = getattr(block, "type", "")
                if block_type in {"output_text", "text"}:
                    text = str(getattr(block, "text", "")).strip()
                    if text:
                        return text
        return ""

    def _emit_assistant_text(self, text: str) -> None:
        """Emit assistant text callback while suppressing near-duplicate events."""
        text = text.strip()
        if not text or self.on_assistant_text is None:
            return
        now = time.monotonic()
        if text == self._last_assistant_text and (now - self._last_assistant_ts) < 2.0:
            logger.debug("[%dms] Suppressed duplicate assistant text", self._elapsed_ms())
            return
        self._last_assistant_text = text
        self._last_assistant_ts = now
        logger.info("[%dms] Assistant text: %s", self._elapsed_ms(), text)
        self._safe_call("on_assistant_text", self.on_assistant_text, text)

    def _to_float32_mono(self, sample: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert incoming sample to mono float32 vector."""
        arr = np.asarray(sample)
        if arr.ndim == 1:
            return arr.astype(np.float32, copy=False)
        if arr.ndim == 2:
            if arr.shape[0] <= 8 and arr.shape[0] <= arr.shape[1]:
                return np.mean(arr, axis=0).astype(np.float32, copy=False)
            return np.mean(arr, axis=1).astype(np.float32, copy=False)
        return np.mean(arr.reshape(arr.shape[0], -1), axis=0).astype(np.float32, copy=False)

    def _resample_to_24k(self, mono: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        """Resample input audio to 24kHz using linear interpolation."""
        if sample_rate == 24000:
            return mono
        if mono.size < 2:
            return mono
        duration = mono.size / float(sample_rate)
        target_size = max(1, int(duration * 24000.0))
        source_x = np.linspace(0.0, 1.0, num=mono.size, dtype=np.float32)
        target_x = np.linspace(0.0, 1.0, num=target_size, dtype=np.float32)
        return np.interp(target_x, source_x, mono).astype(np.float32)
