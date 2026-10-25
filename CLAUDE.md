# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**dobby-the-claw** is an orchestration bridge for an embodied AI assistant running on a Reachy Mini robot. It connects:
- **OpenAI Realtime API** for low-latency voice conversation and tool calling
- **Reachy Mini SDK** for physical embodiment (gestures, head movement, camera, speaker)
- **OpenClaw** (optional) for delegating complex/long-running tasks
- **Home Assistant** (optional) for smart home control

The AI persona is "Dobby" — responds exclusively in Brazilian Portuguese by default, is sarcastic and irreverent but helpful, and keeps responses under 25 words when possible.

## Commands

### Run

```bash
# Realtime voice (requires mic + API key for the conversation provider). The web UI (status, chat,
# camera with detection overlays, filtered logs, controls) is served on http://localhost:18900.
python -m bridge.main

# Optional flags
--no-headtracking          # Disable head tracking
--idle-sleep-timeout-s 300 # Override idle timeout (<=0 disables idle sleep)
```

Run from the `src/` directory, or set `PYTHONPATH=src`.

### Web UI frontend

The page is a React + Vite app in `src/web_ui/frontend/`, built to `frontend/dist/` (git-ignored) and
served by the bridge. `scripts/run-windows.ps1` runs `npm ci` / `npm run build` when needed.

```bash
cd src/web_ui/frontend
npm ci && npm run build   # tsc + vitest + vite build
npm run dev               # hot reload on :5173, proxies /ws and /api to the bridge on :18900
```

### Docker (development hosts)

```bash
cp .env.example .env
# Edit .env with your credentials
docker compose up --build
```

> **Current setup (2026-10):** the Reachy Mini is plugged into the Windows PC, which runs the
> Reachy daemon + bridge (`powershell -File scripts/run-windows.ps1`). The Raspberry Pi only runs
> OpenClaw (:18789) and Home Assistant (:8123), reached through an SSH tunnel (18789, 8123 and the
> reverse 18800). `dobby-bridge`/`reachy-mini-daemon` are disabled on the Pi. Run from `src/`
> (relative `../.dobby` and `../models` paths). First OpenClaw delegation can take ~30 s (cold agent).
>
> Older note — on the **Raspberry Pi the bridge ran natively** (systemd `dobby-bridge.service` in the
> `reachy_mini_env` virtualenv), not in Docker — Docker's CPU/memory overhead and device
> passthrough are too costly on the throttled Pi. See [`docs/raspberry-integration.md`](docs/raspberry-integration.md#deploying-the-bridge-native).
> The `docker-compose.rpi.yml` override is kept minimal; the base + override must not both
> declare `group_add` (Compose concatenates the list, failing validation).

### SSH Tunnels (local development)

Forwards Reachy SDK, WebRTC signaling, OpenClaw and Home Assistant from the Raspberry Pi:

```bash
./scripts/tunnel.sh        # open all tunnels
./scripts/tunnel.sh stop   # close
```

### Lint / Format

```bash
ruff check .    # Lint
ruff format .   # Format (100-char line limit)
```

There is no automated test suite for the bridge; validation is manual/integration-based. The web UI
has unit tests (`npm test` in `src/web_ui/frontend/`) and `tests/web_ui/` (pytest) covers how the
built page is served.

## Architecture

Full architecture reference (components, ports & adapters, state machine, data flow, tool list): [`docs/architecture.md`](docs/architecture.md).

### Ports & Adapters

The orchestrator (`orchestrator.py`) depends only on ports (`ports.py`), never on concrete infra. Ports defined there:
- `ConversationSessionPort` — start/stop/feed-audio for a realtime session
- `ConversationSessionFactoryPort` — creates sessions
- `RobotActionsPort` — gestures, wake/sleep, motor control
- `ToolRuntimePort` — list, spec, and execute tools
- `MediaIOPort` — microphone input and speaker output

New integrations require a new port + adapter pair in `adapters/`.

### State Machine

States: `IDLE → LISTENING → THINKING → EXECUTING → IDLE` (with branches for `DELEGATING`, `CONFIRMING`, `ERROR`). All transitions go through `StateMachine.transition(Event)` — never mutate `.state` directly. Apply transitions via `apply_event()` in `common.py`.

### Tool System

Each tool is a class implementing `ToolHandler` (from `bridge/tools/runtime.py`):
- `definition() → ToolDefinition` — declares name, description, JSON schema, and optional `runtime_guardrail`
- `execute(arguments) → ToolExecutionResult` — runs the tool and returns typed output

All tools are registered in `build_tool_registry()` in `src/bridge/runtime/audio_support.py`. To add a new tool: create a file in `src/bridge/tools/`, implement `ToolHandler`, and call `tool_registry.register(MyTool(...))` in `build_tool_registry`. Export from `src/bridge/tools/__init__.py` if needed.

### Runtime Modes

There is one mode: realtime voice. `realtime_mode.py` is a thin composition layer that builds the `RuntimeOrchestrator`; shared behavior belongs there. Unless `WEB_UI_ENABLED=0`, `main.py` also installs a log queue and starts the web UI (`src/web_ui/`: aiohttp server serving the React PWA from `frontend/dist`, with an SPA fallback so every route survives reload) on its own thread. The 3D digital twin uses Pollen's Reachy Mini model behind the `RobotLink` interface (`frontend/src/robot/`); today it is a mock (`MockRobotLink`) that sends nothing to the robot. The pages feed typed turns and `/wake`, `/sleep` through `orchestrator.text_queue` (the hub refuses `/quit` and any other `/` text from the network). The camera stream with detection overlays comes from `CameraWorker.render_debug_frame()`.

### Web UI security

The server binds to `127.0.0.1` only. Requests that arrive through the Cloudflare tunnel (they carry `Cf-*`/`X-Forwarded-*` headers or a non-local Host) must present a valid Cloudflare Access JWT (`WEB_UI_ACCESS_TEAM_DOMAIN` + `WEB_UI_ACCESS_AUDIENCE`); without that config remote access is refused. Cross-origin writes and WebSocket upgrades are rejected, and secrets are masked in the logs sent to pages. Never bind it to `0.0.0.0` or expose the port without Access in front. The CSP allows only `'self'`, plus `'wasm-unsafe-eval'` and `blob:` workers for the Draco decoder of the 3D model: no CDNs, self-hosted fonts.

### REACHY_BRIDGE_URL

If `REACHY_BRIDGE_URL` starts with `sdk`, the runtime uses the Reachy Mini SDK directly (activates `CameraWorker`, `MotionManager`, SDK media). Otherwise it's a stub HTTP client — SDK features are unavailable.

### Audio Backend

On Linux with `arecord`/`aplay` available, the bridge uses direct ALSA audio by default (`ReachyDirectAlsaMediaIO`). Override with `REACHY_DIRECT_ALSA_AUDIO=0` to use the Reachy SDK GStreamer pipeline instead.

### Realtime Callbacks

No blocking work in any callback. Audio playback runs via `audio_queue`; motion gestures run on the MotionManager thread. Session recovery is automatic (session is recreated on disconnect or error).

## Key Conventions

- **Language:** All AI responses default to Brazilian Portuguese (pt-BR) per `src/prompts/identity.md`
- **Typed results:** Use `ReachyActionResult` (robot actions) and `ToolExecutionResult` (tools) for all action returns
- **Delegation:** When uncertain, the model uses `delegate_task` (OpenClaw) rather than refusing
- **Home Assistant sensitive domains** (`lock`, `alarm_control_panel`, `cover`, `button`) require explicit user confirmation before executing
- **Refactoring:** Prefer root-cause fixes over compatibility shims; remove dead code when safe
- **Feature docs:** [`docs/features.md`](docs/features.md) lists everything Dobby does, in plain English from the user's point of view (no jargon). Whenever you add, change or remove a user-facing feature (tool, behavior, mode, integration), update that file in the same change.

## Deployment

For full Raspberry Pi setup (hardware requirements, OpenClaw, Home Assistant, Reachy daemon, systemd service), see [`docs/raspberry-integration.md`](docs/raspberry-integration.md).

Custom OpenClaw skills live in `openclaw/skills/` and are copied to `~/.openclaw/workspace/skills/` on the Pi (e.g. `compras`: shopping with 1Password-backed secrets). The repo is public — keep personal data (address, phone, card) in the Pi's workspace `USER.md`, never in the skill.

## Configuration

All runtime config is via environment variables (see `.env.example`). Key variables:

| Variable | Default | Notes |
|---|---|---|
| `REACHY_BRIDGE_URL` | `http://reachy-bridge:8001` | Set to `sdk` to enable Reachy SDK mode |
| `OPENAI_API_KEY` | — | Read via `LLM_API_KEY_ENV`. Optional with `CONVERSATION_PROVIDER=gemini` (speaker memory then uses Gemini too) |
| `CONVERSATION_PROVIDER` | `openai` | `openai` (Realtime API) or `gemini` (Gemini Live) |
| `REALTIME_MODEL` | `gpt-realtime` | OpenAI Realtime API model |
| `GEMINI_API_KEY` | — | Required when `CONVERSATION_PROVIDER=gemini` |
| `GEMINI_LIVE_MODEL` | `gemini-3.8-live` | Or `gemini-3.8-live-extended-thinking` (adds ~2.5 s before tool calls) |
| `GEMINI_THINKING_LEVEL` | `low` | Only for the extended-thinking model (`low`/`medium`/`high`) |
| `GEMINI_VOICE` | — | Optional prebuilt Gemini voice name (for example `Algenib`) |
| `GEMINI_VOICE_STYLE` | — | Free-text delivery style added to the system prompt (tone, pace, mood) |
| `GEMINI_PROACTIVE_AUDIO` | `1` | Gemini decides whether speech is addressed to Dobby and stays silent otherwise (rules in `src/prompts/addressing.md`; uses the v1alpha Live API) |
| `STT_LANGUAGE` | `pt` | Speech recognition language |
| `BRIDGE_LOG_LEVEL` | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARNING`) |
| `OFFLINE_WAKEWORD_ENABLED` | `1` | Enable offline wakeword detection |
| `OFFLINE_WAKEWORD_ALIASES` | `reachy,dobby` | Names that wake Dobby (speech bursts are transcribed locally with faster-whisper) |
| `OFFLINE_WAKEWORD_MODEL` | `base` | Whisper model for the wake word (`base` or `small`) |
| `REACHY_DIRECT_ALSA_AUDIO` | `1` | Use `arecord`/`aplay` instead of SDK GStreamer |
| `OPENCLAW_ENABLED` | `1` | Enable OpenClaw task delegation |
| `OPENCLAW_WS_URL` | `ws://127.0.0.1:18789` | OpenClaw WebSocket gateway URL |
| `OPENCLAW_BEARER_TOKEN` | — | Auth token for OpenClaw |
| `HOME_ASSISTANT_ENABLED` | `0` | Enable Home Assistant integration |
| `HOME_ASSISTANT_URL` | `http://127.0.0.1:8123` | HA base URL (HTTP or WS scheme) |
| `HOME_ASSISTANT_TOKEN` | — | Long-lived HA access token |
| `SPEAKER_ID_ENABLED` | `1` | Enable face-based speaker identification |
| `VOICE_ID_ENABLED` | `1` | Also identify speakers by voice (sherpa-onnx), learned automatically from face matches |
| `PRESENCE_ENABLED` | `1` | Person detection: follow distant people, choose whom to look at, presence notices for greetings (rules in `src/prompts/presence.md`) |
| `SPEAKER_MEMORY_ENABLED` | `0` | Persist per-speaker conversation summaries |
| `NOTIFICATION_SERVER_PORT` | `18800` | HTTP port for injecting async notifications |
| `WEB_UI_ENABLED` | `1` | Serve the web UI (PWA) on `WEB_UI_HOST:WEB_UI_PORT` (`127.0.0.1:18900`) |
| `WEB_UI_ACCESS_TEAM_DOMAIN` / `WEB_UI_ACCESS_AUDIENCE` | — | Cloudflare Access team domain and application AUD; required for any non-local access |
