"""Web UI server: live status, chat, logs and camera for a browser or phone (PWA).

Runs inside the bridge process on its own thread and event loop, listening on localhost only.
Remote access goes through the Cloudflare tunnel, guarded by Cloudflare Access (see access.py).
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Awaitable, Optional

import cv2
from aiohttp import WSMsgType, web

from bridge.status_checks import check_service_up, get_pi_stats
from web_ui.audio_control import DaemonAudio
from web_ui.access import AccessPolicy, AccessVerifier, make_guard
from web_ui.frontend_routes import FRONTEND_DIST, add_frontend_routes, warn_if_missing
from web_ui.logbus import CATEGORIES, LogLine, redact
from web_ui.robot_control import RobotControl

if TYPE_CHECKING:
    from bridge.config import BridgeConfig
    from bridge.runtime.orchestrator import RuntimeOrchestrator

logger = logging.getLogger(__name__)

_LOG_BACKLOG = 3000
_MAX_CHAT_CHARS = 2000
_MAX_STREAMS = 8  # viewers; dead ones are reaped by the transport check in the stream loop
_WRITE_TIMEOUT_S = 5.0
_VIDEO_PERIOD_S = 0.125
_VIDEO_MAX_WIDTH = 1280
_USER_TEXT = re.compile(r"User (said|typed): (.*)", re.S)
_LOG_LEVELS = {"DEBUG": logging.DEBUG, "INFO": logging.INFO}
_ROBOT_PERIOD_S = 1 / 15  # pose stream to the 3D twin
_AUDIO_PERIOD_S = 0.1  # mic level and sound direction, for the home page's waveform
_VOLUME_POLL_S = 5.0  # volumes changed elsewhere (the official app, the OS) show up this late


class Hub:
    """State shared by every connected page: log/chat history, status snapshot and commands."""

    def __init__(self, orchestrator: "RuntimeOrchestrator", config: "BridgeConfig", log_queue: "queue.Queue[LogLine]") -> None:
        self.orch = orchestrator
        self.camera = orchestrator.camera_worker
        self.config = config
        self._log_queue = log_queue
        self.clients: set[web.WebSocketResponse] = set()
        self.logs: deque[dict] = deque(maxlen=_LOG_BACKLOG)
        self.chat: deque[dict] = deque(maxlen=500)
        self.services: dict[str, Any] = {}
        self.video_enabled = True
        self.streams = 0
        self._recording = False
        self.tasks: set[asyncio.Task] = set()
        self._jpeg_lock = threading.Lock()
        self._jpeg_cache: dict[tuple, tuple[float, bytes]] = {}
        self._frames = 0
        self._fps_since = time.monotonic()
        self.fps = 0.0
        self._seq = 0
        self.robot = RobotControl(
            orchestrator.motion_manager,
            orchestrator.tool_runtime.execute,
            lambda: bool(orchestrator.sleeping or orchestrator.transitioning),
        )
        self.audio = DaemonAudio(config.reachy_daemon_url)
        provider = config.conversation_provider
        self.model = f"{provider} · {config.gemini_live_model if provider == 'gemini' else config.realtime_model}"

    # ------------------------------------------------------------------ snapshots

    def status(self) -> dict:
        orch, cam = self.orch, self.camera
        wake = orch.wakeword
        people = []
        speaker = None
        if cam is not None:
            people = [{"id": pid, "name": name} for pid, name, _ in cam.get_presence()]
            speaker = cam.get_current_speaker()
        return {
            "type": "status",
            "state": orch.state_machine.state.value,
            "sleeping": orch.sleeping,
            "transition": orch.transitioning,
            "model": self.model,
            "people": people,
            "speaker": speaker,
            "camera": cam is not None,
            "head_tracking": bool(cam.is_head_tracking_enabled) if cam is not None else False,
            "video_enabled": self.video_enabled,
            "fps": round(self.fps, 1),
            "mic": {"rms": round(wake.last_rms, 5)} if wake.enabled else None,
            "wake": {"heard": wake.heard_recently()} if wake.enabled else None,
            "services": self.services,
            "log_level": logging.getLevelName(logging.getLogger().level),
        }

    def hello(self) -> dict:
        return {
            "type": "hello",
            "categories": list(CATEGORIES),
            "logs": list(self.logs)[-800:],
            "chat": list(self.chat),
            "status": self.status(),
            "robot": self.robot.snapshot(),
            "volumes": self.audio.volumes,
        }

    # ------------------------------------------------------------------ background work

    async def pump(self) -> None:
        """Move new log lines into the history, route chat turns out of them and push both to the pages."""
        while True:
            new_logs, new_chat = [], []
            for _ in range(1000):
                try:
                    line = self._log_queue.get_nowait()
                except queue.Empty:
                    break
                self._seq += 1
                entry = {"id": self._seq, "t": line.created, "lv": line.level, "cat": line.category, "msg": redact(line.message)}
                self.logs.append(entry)
                new_logs.append(entry)
                chat = self._chat_from(line)
                if chat:
                    self.chat.append(chat)
                    new_chat.append(chat)
            if new_logs:
                await self.broadcast({"type": "logs", "lines": new_logs})
            for chat in new_chat:
                await self.broadcast({"type": "chat", "entry": chat})
            await asyncio.sleep(0.15)

    @staticmethod
    def _chat_from(line: LogLine) -> Optional[dict]:
        if line.source == "bridge.conversation" and line.message.startswith("Dobby ❯ "):
            return {"t": line.created, "who": "dobby", "text": redact(line.message[len("Dobby ❯ "):])}
        found = _USER_TEXT.search(line.message) if line.source == "root" else None
        if found:
            return {"t": line.created, "who": "you", "text": redact(found.group(2))}
        return None

    async def status_loop(self) -> None:
        while True:
            if self.clients:
                await self.broadcast(self.status())
            await asyncio.sleep(1.0)

    async def robot_loop(self) -> None:
        """Stream the robot's pose to the pages (the 3D twin) and apply manual-mode safety releases."""
        while True:
            try:
                self.robot.tick()
                if self.clients:
                    snapshot = self.robot.snapshot()
                    if snapshot is not None:
                        await self.broadcast(snapshot)
            except Exception as exc:
                logger.debug("Web UI robot stream failed: %s", exc)
            await asyncio.sleep(_ROBOT_PERIOD_S)

    async def audio_loop(self) -> None:
        """Stream the mic level and the sound direction while a page is open."""
        wake = self.orch.wakeword
        while True:
            started = time.monotonic()
            if self.clients:
                try:
                    rms = round(wake.last_rms, 5) if wake.enabled else None
                    await self.broadcast({"type": "audio", "rms": rms, "doa": await self._doa()})
                except Exception as exc:
                    logger.debug("Web UI audio stream failed: %s", exc)
            await asyncio.sleep(max(0.02, _AUDIO_PERIOD_S - (time.monotonic() - started)))

    async def _doa(self) -> Optional[dict]:
        # The bridge reads the mic array itself when it has the camera worker; the daemon then has nothing.
        if self.camera is not None:
            reading = self.camera.latest_doa()
            return None if reading is None else {"angle": round(reading[0], 3), "speech": reading[1]}
        return await self.audio.doa()

    async def volume_loop(self) -> None:
        while True:
            if self.clients:
                before = dict(self.audio.volumes)
                if await self.audio.refresh() != before:
                    await self.broadcast({"type": "volumes", **self.audio.volumes})
            await asyncio.sleep(_VOLUME_POLL_S)

    async def volume_command(self, message: dict, ws: web.WebSocketResponse) -> None:
        value = message.get("volume")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            error = "volume inválido"
        else:
            error = await self.audio.set_volume(str(message.get("device", "")), int(value))
        if error and not ws.closed:
            await ws.send_str(json.dumps({"type": "error", "text": error}, ensure_ascii=False))
        await self.broadcast({"type": "volumes", **self.audio.volumes})

    async def robot_command(self, message: dict, ws: web.WebSocketResponse) -> Optional[str]:
        """Manual control from the twin; `ws` owns manual mode until it switches back or disconnects."""
        kind = message.get("type")
        if kind == "robot_mode":
            return self.robot.set_manual(bool(message.get("manual")), ws)
        if kind == "robot_target" and isinstance(message.get("target"), dict):
            return self.robot.set_target(message["target"], ws)
        if kind == "robot_emotion":
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.robot.play_emotion, str(message.get("name", "")), ws)
        return "mensagem inválida"

    async def services_loop(self) -> None:
        cfg = self.config
        while True:
            try:
                jobs = {}
                if cfg.home_assistant_enabled and cfg.home_assistant_url:
                    jobs["ha"] = check_service_up(cfg.home_assistant_url)
                if cfg.openclaw_enabled and cfg.openclaw_ws_url:
                    jobs["oc"] = check_service_up(cfg.openclaw_ws_url)
                if cfg.pi_ssh_host:
                    jobs["pi"] = get_pi_stats(host=cfg.pi_ssh_host, user=cfg.pi_ssh_user)
                values = await asyncio.gather(*jobs.values(), return_exceptions=True)
                self.services = {k: (None if isinstance(v, Exception) else v) for k, v in zip(jobs, values)}
            except Exception as exc:
                logger.debug("Web UI services poll failed: %s", exc)
            await asyncio.sleep(5.0)

    async def broadcast(self, message: dict) -> None:
        if not self.clients:
            return
        data = json.dumps(message, ensure_ascii=False)
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except Exception:
                self.clients.discard(ws)

    # ------------------------------------------------------------------ commands from pages

    def handle(self, message: dict, identity: str) -> Optional[str]:
        """Apply one command from a page; returns an error text for the page, or None."""
        kind = message.get("type")
        if kind == "chat":
            text = str(message.get("text", "")).strip()
            if not text:
                return None
            if len(text) > _MAX_CHAT_CHARS:
                return "mensagem longa demais"
            if text.startswith("/"):
                return "comandos com / não são aceitos aqui"  # "/quit" etc. are orchestrator controls
            if self.orch.sleeping:
                return "o Dobby está dormindo: acorde-o primeiro"
            logger.info("Web UI chat from %s", identity)
            self.orch.text_queue.put(text)
        elif kind == "cmd" and message.get("name") in ("wake", "sleep"):
            self.orch.text_queue.put("/" + message["name"])
        elif kind == "head_tracking" and self.camera is not None:
            self.camera.set_head_tracking_enabled(bool(message.get("enabled")))
        elif kind == "video_enabled":
            self.video_enabled = bool(message.get("enabled"))
        elif kind == "log_level" and message.get("level") in _LOG_LEVELS:
            logging.getLogger().setLevel(_LOG_LEVELS[message["level"]])
        return None

    # ------------------------------------------------------------------ video

    def render_jpeg(self, people: bool, voice: bool, tracking: bool, width: int) -> Optional[bytes]:
        """JPEG of the annotated frame; viewers with the same options share one render per frame period."""
        key = (people, voice, tracking, width)
        with self._jpeg_lock:
            cached = self._jpeg_cache.get(key)
            if cached is not None and time.monotonic() - cached[0] < _VIDEO_PERIOD_S * 0.8:
                return cached[1]
            jpeg = self._render_jpeg(people, voice, tracking, width)
            if jpeg is not None:
                self._jpeg_cache[key] = (time.monotonic(), jpeg)
            return jpeg

    def _render_jpeg(self, people: bool, voice: bool, tracking: bool, width: int) -> Optional[bytes]:
        if self.camera is None:
            return None
        frame = self.camera.render_debug_frame(show_people=people, show_tracking=tracking, show_voice=voice)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, max(1, int(h * width / w))), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        return buffer.tobytes() if ok else None

    # ------------------------------------------------------------------ wakeword test

    async def wake_command(self, message: dict, ws: web.WebSocketResponse) -> None:
        """Record a short clip and report what the wakeword detector hears in it."""
        send = lambda payload: ws.send_str(json.dumps(payload, ensure_ascii=False))  # noqa: E731
        wake = self.orch.wakeword
        if message.get("type") != "wake_test":
            return
        if not wake.enabled:
            await send({"type": "error", "text": "O detector de nome está desligado (OFFLINE_WAKEWORD_ENABLED=0)."})
            return
        if self._recording:
            return
        seconds = 3.0
        self._recording = True
        wake.muted_until = time.monotonic() + seconds + 5.0  # the test must not wake the robot
        loop = asyncio.get_running_loop()
        try:
            await send({"type": "wake_state", "recording": True, "seconds": seconds})
            audio = await loop.run_in_executor(None, self.orch.record_mic_clip, seconds)
            await send({"type": "wake_state", "recording": False, "thinking": True})
            text = await loop.run_in_executor(None, wake.hear_clip, audio)
            if text is None:
                await send({"type": "wake_result", "text": "", "hit": False, "silent": True})
                return
            await send({"type": "wake_result", "text": text, "hit": wake.names_robot(text), "silent": False})
        except Exception:
            logger.exception("Wakeword test failed")
            await send({"type": "error", "text": "O teste falhou. Tente de novo."})
        finally:
            self._recording = False
            wake.muted_until = time.monotonic() + 1.5
            await send({"type": "wake_state", "recording": False})

    def count_frame(self) -> None:
        self._frames += 1
        elapsed = time.monotonic() - self._fps_since
        if elapsed >= 2.0:
            self.fps, self._frames, self._fps_since = self._frames / elapsed, 0, time.monotonic()


def _flag(request: web.Request, name: str, default: bool = True) -> bool:
    value = request.query.get(name)
    return default if value is None else value not in ("0", "false")


async def _ws(request: web.Request) -> web.StreamResponse:
    hub: Hub = request.app["hub"]
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=64 * 1024)
    await ws.prepare(request)
    hub.clients.add(ws)
    identity = request["identity"]
    allowance, last = 10.0, time.monotonic()  # token bucket: 10 commands burst, 5 per second
    drive, drive_last = 40.0, time.monotonic()  # the twin's controls send up to ~30 targets a second
    try:
        await ws.send_str(json.dumps(hub.hello(), ensure_ascii=False))
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            now = time.monotonic()
            try:
                payload = json.loads(msg.data)
                if isinstance(payload, dict) and str(payload.get("type", "")).startswith("robot_"):
                    drive = min(40.0, drive + (now - drive_last) * 30.0)
                    drive_last = now
                    if drive < 1.0:
                        continue
                    drive -= 1.0
                    if payload.get("type") == "robot_emotion":
                        task = asyncio.create_task(_reply_error(ws, hub.robot_command(payload, ws)))
                        hub.tasks.add(task)
                        task.add_done_callback(hub.tasks.discard)
                        continue
                    error = await hub.robot_command(payload, ws)
                    if error:
                        await ws.send_str(json.dumps({"type": "error", "text": error}, ensure_ascii=False))
                    continue
                allowance = min(10.0, allowance + (now - last) * 5.0)
                last = now
                if allowance < 1.0:
                    continue
                allowance -= 1.0
                if isinstance(payload, dict) and payload.get("type") == "volume":
                    task = asyncio.create_task(hub.volume_command(payload, ws))
                    hub.tasks.add(task)
                    task.add_done_callback(hub.tasks.discard)
                    continue
                if isinstance(payload, dict) and str(payload.get("type", "")).startswith("wake_"):
                    task = asyncio.create_task(hub.wake_command(payload, ws))  # recording takes seconds: don't hold the socket
                    hub.tasks.add(task)  # the loop only keeps weak references to tasks
                    task.add_done_callback(hub.tasks.discard)
                    continue
                error = hub.handle(payload, identity)
            except (ValueError, TypeError):
                error = "mensagem inválida"
            if error:
                await ws.send_str(json.dumps({"type": "error", "text": error}, ensure_ascii=False))
    finally:
        hub.clients.discard(ws)
        hub.robot.release(ws)
    return ws


async def _reply_error(ws: web.WebSocketResponse, pending: Awaitable[Optional[str]]) -> None:
    error = await pending
    if error and not ws.closed:
        await ws.send_str(json.dumps({"type": "error", "text": error}, ensure_ascii=False))


async def _video(request: web.Request) -> web.StreamResponse:
    hub: Hub = request.app["hub"]
    if hub.camera is None or not hub.video_enabled:
        raise web.HTTPNotFound(text="câmera indisponível")
    if hub.streams >= _MAX_STREAMS:
        raise web.HTTPTooManyRequests(text="muitas transmissões abertas")
    people, voice, tracking = (_flag(request, name) for name in ("people", "voice", "tracking"))
    try:
        width = max(320, min(_VIDEO_MAX_WIDTH, int(request.query.get("w", "960"))))
    except ValueError:
        width = 960
    response = web.StreamResponse(headers={"Content-Type": "multipart/x-mixed-replace; boundary=frame", "Cache-Control": "no-store"})
    await response.prepare(request)
    loop = asyncio.get_running_loop()
    hub.streams += 1
    try:
        while hub.video_enabled:
            transport = request.transport
            if transport is None or transport.is_closing():
                break  # the viewer is gone (aiohttp does not cancel the handler on disconnect)
            started = time.monotonic()
            jpeg = await loop.run_in_executor(None, hub.render_jpeg, people, voice, tracking, width)
            if jpeg:
                await asyncio.wait_for(
                    response.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg) + jpeg + b"\r\n"),
                    _WRITE_TIMEOUT_S,
                )
                hub.count_frame()
            await asyncio.sleep(max(0.02, _VIDEO_PERIOD_S - (time.monotonic() - started)))
    except (OSError, asyncio.TimeoutError, asyncio.CancelledError):
        pass  # the viewer left (phone locked, tunnel dropped): not an error
    finally:
        hub.streams -= 1
    return response


def build_app(hub: Hub, policy: AccessPolicy) -> web.Application:
    app = web.Application(middlewares=[make_guard(AccessVerifier(policy))])
    app["hub"] = hub
    app.router.add_get("/ws", _ws)
    app.router.add_get("/api/video", _video)
    add_frontend_routes(app, FRONTEND_DIST)  # last: it catches every other path
    return app


def start_web_ui(orchestrator: "RuntimeOrchestrator", config: "BridgeConfig", log_queue: "queue.Queue[LogLine]") -> threading.Thread:
    """Start the web UI on its own thread; returns the thread (daemon, ends with the process)."""
    policy = AccessPolicy(
        team_domain=config.web_ui_access_team_domain,
        audience=config.web_ui_access_audience,
        allowed_emails=config.web_ui_allowed_emails,
        frame_ancestors=config.web_ui_frame_ancestors,
    )
    hub = Hub(orchestrator, config, log_queue)
    app = build_app(hub, policy)
    warn_if_missing(FRONTEND_DIST)

    async def serve() -> None:
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, config.web_ui_host, config.web_ui_port).start()
        logger.info(
            "Web UI on http://%s:%d (remote access %s)",
            config.web_ui_host, config.web_ui_port, "guarded by Cloudflare Access" if policy.configured else "OFF: Access not configured",
        )
        await hub.audio.refresh()
        loops = (hub.pump(), hub.status_loop(), hub.services_loop(), hub.robot_loop(), hub.audio_loop(), hub.volume_loop())
        tasks = [asyncio.create_task(coro) for coro in loops]
        try:
            await asyncio.Event().wait()
        finally:
            for task in tasks:
                task.cancel()
            await runner.cleanup()

    def run() -> None:
        try:
            asyncio.run(serve())
        except Exception:
            logger.exception("Web UI stopped")

    thread = threading.Thread(target=run, name="web-ui", daemon=True)
    thread.start()
    return thread
