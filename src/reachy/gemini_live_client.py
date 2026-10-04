"""Gemini Live streaming client implementing the same session interface as the OpenAI one."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import threading
import time
from concurrent.futures import CancelledError
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from google import genai
from google.genai import types

from bridge.tools.contracts import ToolExecutionResult

logger = logging.getLogger(__name__)

INPUT_RATE = 16000
MAX_RECONNECTS = 5
# Transcript markers emitted when proactive audio decides not to speak.
_SILENCE_MARKERS = re.compile(r"<no speech>|\{pause\}")


def tool_specs_to_declarations(tool_specs: List[Dict[str, Any]]) -> List[types.FunctionDeclaration]:
    """Convert OpenAI-style function specs to Gemini function declarations."""
    declarations: List[types.FunctionDeclaration] = []
    for spec in tool_specs:
        name = spec.get("name")
        if not name:
            continue
        params = spec.get("parameters")
        has_props = isinstance(params, dict) and bool(params.get("properties"))
        declarations.append(
            types.FunctionDeclaration(
                name=name,
                description=spec.get("description") or "",
                parameters_json_schema=params if has_props else None,
            )
        )
    return declarations


class GeminiLiveSession:
    """Maintain a background Gemini Live connection and stream audio frames."""

    def __init__(
        self,
        api_key: str,
        model: str,
        instructions: str,
        thinking_level: str,
        voice: str,
        voice_style: str,
        proactive_audio: bool,
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
        """Store config and callbacks; the connection starts in `start()`."""
        self.api_key = api_key
        self.model = model
        self.instructions = instructions
        self.thinking_level = thinking_level
        self.voice = voice
        self.voice_style = voice_style
        self.proactive_audio = proactive_audio
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
        self._session: Any = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._resume_handle: Optional[str] = None
        self._start_ts = time.monotonic()
        self._user_text = ""
        self._assistant_text = ""
        self._user_speaking = False
        self._tasks: set[asyncio.Task] = set()

    def _elapsed_ms(self) -> int:
        return int((time.monotonic() - self._start_ts) * 1000)

    def _safe_call(self, label: str, callback: Optional[Callable], *args) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as exc:
            logger.exception("[%dms] Callback '%s' failed: %s", self._elapsed_ms(), label, exc)

    def start(self) -> None:
        """Start the Live connection in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)
        self._thread.start()
        logger.info("[%dms] Gemini Live thread started", self._elapsed_ms())

    def wait_until_ready(self, timeout_s: float = 8.0) -> bool:
        """Wait until the session is connected."""
        return self._ready.wait(timeout=timeout_s)

    def stop(self) -> None:
        """Stop the background session."""
        self._stop.set()
        logger.info("[%dms] Gemini Live stop requested", self._elapsed_ms())
        loop = self._loop
        if loop is not None and loop.is_running():

            def _cancel_all() -> None:
                for task in asyncio.all_tasks(loop):
                    task.cancel()

            loop.call_soon_threadsafe(_cancel_all)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _build_config(self) -> types.LiveConnectConfig:
        instructions = (
            "Responda SEMPRE em português do Brasil, independentemente do idioma percebido.\n\n"
            + self.instructions
        )
        cfg: Dict[str, Any] = {
            "response_modalities": ["AUDIO"],
            "system_instruction": instructions,
            "input_audio_transcription": {},
            "output_audio_transcription": {},
            "context_window_compression": {"sliding_window": {}},
            "session_resumption": ({"handle": self._resume_handle} if self._resume_handle else {}),
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "silence_duration_ms": self.vad_silence_ms,
                    "prefix_padding_ms": self.vad_prefix_padding_ms,
                },
            },
        }
        if self.voice:
            cfg["speech_config"] = {
                "voice_config": {"prebuilt_voice_config": {"voice_name": self.voice}}
            }
        if self.proactive_audio:
            cfg["proactivity"] = {"proactive_audio": True}
        if "extended" in self.model:
            cfg["thinking_config"] = {"thinking_level": self.thinking_level}
        declarations = tool_specs_to_declarations(self.tool_specs)
        if declarations:
            cfg["tools"] = [types.Tool(function_declarations=declarations)]
        return types.LiveConnectConfig(**cfg)

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        # Proactive audio is only available on the v1alpha Live API.
        http_options = {"api_version": "v1alpha"} if self.proactive_audio else None
        client = genai.Client(api_key=self.api_key, http_options=http_options)
        attempts = 0
        try:
            while not self._stop.is_set():
                try:
                    logger.info("[%dms] Connecting Gemini Live model=%s", self._elapsed_ms(), self.model)
                    async with client.aio.live.connect(
                        model=self.model, config=self._build_config()
                    ) as session:
                        self._session = session
                        self._ready.set()
                        attempts = 0
                        logger.info("[%dms] Gemini Live session ready", self._elapsed_ms())
                        while not self._stop.is_set():
                            async for message in session.receive():
                                await self._handle_message(message)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if self._stop.is_set():
                        break
                    attempts += 1
                    logger.warning(
                        "[%dms] Gemini Live connection lost (%d): %r", self._elapsed_ms(), attempts, exc
                    )
                    if attempts > MAX_RECONNECTS:
                        self._safe_call("on_error", self.on_error, str(exc) or repr(exc))
                        break
                    await asyncio.sleep(min(2.0 * attempts, 8.0))
                finally:
                    self._ready.clear()
                    self._session = None
        except asyncio.CancelledError:
            pass
        finally:
            for task in list(self._tasks):
                task.cancel()
            logger.info("[%dms] Gemini Live session closed", self._elapsed_ms())

    def _flush_user_text(self) -> None:
        text = self._user_text.strip()
        self._user_text = ""
        self._user_speaking = False
        if text:
            logger.info("[%dms] User transcript: %s", self._elapsed_ms(), text)
            self._safe_call("on_user_transcript", self.on_user_transcript, text)

    def _flush_assistant_text(self) -> None:
        text = _SILENCE_MARKERS.sub("", self._assistant_text).strip()
        self._assistant_text = ""
        if text:
            logger.info("[%dms] Assistant text: %s", self._elapsed_ms(), text)
            self._safe_call("on_assistant_text", self.on_assistant_text, text)

    async def _handle_message(self, message: Any) -> None:
        update = getattr(message, "session_resumption_update", None)
        if update is not None and getattr(update, "resumable", False) and update.new_handle:
            self._resume_handle = update.new_handle

        if getattr(message, "go_away", None) is not None:
            logger.info("[%dms] Gemini Live GoAway; reconnecting with resumption", self._elapsed_ms())
            raise ConnectionError("GoAway")

        tool_call = getattr(message, "tool_call", None)
        if tool_call is not None and tool_call.function_calls:
            self._flush_user_text()
            task = asyncio.create_task(self._handle_tool_calls(tool_call.function_calls))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        sc = getattr(message, "server_content", None)
        if sc is None:
            return

        if getattr(sc, "interrupted", False):
            logger.debug("[%dms] interrupted", self._elapsed_ms())
            self._assistant_text = ""
            self._user_speaking = True
            self._safe_call("on_speech_start", self.on_speech_start)

        it = getattr(sc, "input_transcription", None)
        if it is not None and it.text:
            if not self._user_speaking:
                self._user_speaking = True
                self._safe_call("on_speech_start", self.on_speech_start)
            self._user_text += it.text

        if sc.model_turn is not None:
            self._flush_user_text()
            for part in sc.model_turn.parts or []:
                data = part.inline_data.data if part.inline_data else None
                if data and self.on_assistant_audio_chunk is not None:
                    pcm16 = np.frombuffer(data, dtype=np.int16)
                    if pcm16.size:
                        audio = (pcm16.astype(np.float32) / 32767.0).astype(np.float32)
                        self._safe_call("on_assistant_audio_chunk", self.on_assistant_audio_chunk, audio)

        ot = getattr(sc, "output_transcription", None)
        if ot is not None and ot.text:
            self._flush_user_text()
            self._assistant_text += ot.text

        if getattr(sc, "generation_complete", False):
            self._flush_assistant_text()
            logger.debug("[%dms] Assistant audio done", self._elapsed_ms())
            self._safe_call("on_assistant_audio_done", self.on_assistant_audio_done)

        if getattr(sc, "turn_complete", False):
            self._flush_assistant_text()
            self._flush_user_text()

    async def _handle_tool_calls(self, calls: Any) -> None:
        responses: List[types.FunctionResponse] = []
        images: List[str] = []
        for call in calls:
            args = dict(call.args or {})
            logger.info("[%dms] Tool call: %s args=%s", self._elapsed_ms(), call.name, args)
            if self.on_tool_call is None:
                result = ToolExecutionResult(output={"ok": False, "message": "Tooling not configured"})
            else:
                try:
                    result = await asyncio.to_thread(self.on_tool_call, call.name, args)
                except Exception as exc:
                    logger.exception("[%dms] Tool execution failed: %s", self._elapsed_ms(), exc)
                    result = ToolExecutionResult(output={"ok": False, "message": str(exc)})
            output = result.output if isinstance(result.output, dict) else {"result": result.output}
            responses.append(
                types.FunctionResponse(
                    id=call.id,
                    name=call.name,
                    response=json.loads(json.dumps(output, default=str)),
                )
            )
            if isinstance(result.image_base64, str) and result.image_base64:
                images.append(result.image_base64)

        session = self._session
        if session is None:
            return
        try:
            for image_b64 in images:
                await session.send_realtime_input(
                    video=types.Blob(data=base64.b64decode(image_b64), mime_type="image/jpeg")
                )
            await session.send_tool_response(function_responses=responses)
        except Exception as exc:
            logger.warning("[%dms] Failed to send tool response: %r", self._elapsed_ms(), exc)

    def feed_audio(self, sample_rate: int, sample: NDArray[np.float32]) -> None:
        """Feed one microphone chunk to the live session (16 kHz PCM16)."""
        session, loop = self._session, self._loop
        if not self._ready.is_set() or session is None or loop is None:
            return
        arr = np.asarray(sample)
        if arr.ndim == 2:
            axis = 0 if (arr.shape[0] <= 8 and arr.shape[0] <= arr.shape[1]) else 1
            arr = arr.mean(axis=axis)
        mono = arr.reshape(-1).astype(np.float32, copy=False)
        if mono.size == 0:
            return
        if sample_rate != INPUT_RATE and mono.size >= 2:
            target = max(1, int(mono.size * INPUT_RATE / float(sample_rate)))
            mono = np.interp(
                np.linspace(0.0, 1.0, num=target, dtype=np.float32),
                np.linspace(0.0, 1.0, num=mono.size, dtype=np.float32),
                mono,
            ).astype(np.float32)
        pcm16 = (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)
        future = asyncio.run_coroutine_threadsafe(
            session.send_realtime_input(
                audio=types.Blob(data=pcm16.tobytes(), mime_type=f"audio/pcm;rate={INPUT_RATE}")
            ),
            loop,
        )
        future.add_done_callback(self._consume_future_exception)

    @staticmethod
    def _consume_future_exception(future: Any) -> None:
        try:
            future.exception()
        except (CancelledError, Exception):
            return

    def send_text(self, text: str, respond: bool = True) -> bool:
        """Send one text turn. With respond=False the text is only added as context."""
        text = text.strip()
        session, loop = self._session, self._loop
        if not text or not self._ready.is_set() or session is None or loop is None:
            return False
        if respond:
            coro = session.send_realtime_input(text=text)
        else:
            coro = session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=text)]),
                turn_complete=False,
            )
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            future.result(timeout=4.0)
            return True
        except Exception as exc:
            logger.warning("[%dms] Failed to send text turn: %r", self._elapsed_ms(), exc)
            return False
