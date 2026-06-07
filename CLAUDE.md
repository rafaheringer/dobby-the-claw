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
# Realtime voice mode (requires mic + OpenAI API key)
python -m bridge.main --mode realtime

# Terminal chat mode (no audio hardware needed)
python -m bridge.main --mode chat --no-headtracking

# Rich TUI mode (Textual-based interface)
python -m bridge.main --mode tui

# Optional flags
--no-headtracking          # Disable head tracking
--idle-sleep-timeout-s 300 # Override idle timeout (<=0 disables idle sleep)
```

Run from the `src/` directory, or set `PYTHONPATH=src`.

### Docker (development hosts)

```bash
cp .env.example .env
# Edit .env with your credentials
docker compose up --build
```

> On the **Raspberry Pi the bridge runs natively** (systemd `dobby-bridge.service` in the
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

There is no automated test suite; validation is manual/integration-based.

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

`realtime_mode.py` and `chat_mode.py` are thin composition layers that build a `RuntimeOrchestrator`. Shared behavior belongs in `RuntimeOrchestrator`. The `tui_mode.py` wraps the orchestrator in a Textual-based UI.

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

## Deployment

For full Raspberry Pi setup (hardware requirements, OpenClaw, Home Assistant, Reachy daemon, systemd service), see [`docs/raspberry-integration.md`](docs/raspberry-integration.md).

## Configuration

All runtime config is via environment variables (see `.env.example`). Key variables:

| Variable | Default | Notes |
|---|---|---|
| `REACHY_BRIDGE_URL` | `http://reachy-bridge:8001` | Set to `sdk` to enable Reachy SDK mode |
| `OPENAI_API_KEY` | — | Read via `LLM_API_KEY_ENV` (default: `OPENAI_API_KEY`) |
| `REALTIME_MODEL` | `gpt-realtime` | OpenAI Realtime API model |
| `STT_LANGUAGE` | `pt` | Speech recognition language |
| `BRIDGE_LOG_LEVEL` | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARNING`) |
| `OFFLINE_WAKEWORD_ENABLED` | `1` | Enable offline wakeword detection |
| `OFFLINE_WAKEWORD_ALIASES` | `reachy,dobby` | Comma-separated wakeword names |
| `REACHY_DIRECT_ALSA_AUDIO` | `1` | Use `arecord`/`aplay` instead of SDK GStreamer |
| `OPENCLAW_ENABLED` | `1` | Enable OpenClaw task delegation |
| `OPENCLAW_WS_URL` | `ws://127.0.0.1:18789` | OpenClaw WebSocket gateway URL |
| `OPENCLAW_BEARER_TOKEN` | — | Auth token for OpenClaw |
| `HOME_ASSISTANT_ENABLED` | `0` | Enable Home Assistant integration |
| `HOME_ASSISTANT_URL` | `http://127.0.0.1:8123` | HA base URL (HTTP or WS scheme) |
| `HOME_ASSISTANT_TOKEN` | — | Long-lived HA access token |
| `SPEAKER_ID_ENABLED` | `1` | Enable face-based speaker identification |
| `SPEAKER_MEMORY_ENABLED` | `0` | Persist per-speaker conversation summaries |
| `NOTIFICATION_SERVER_PORT` | `18800` | HTTP port for injecting async notifications |
