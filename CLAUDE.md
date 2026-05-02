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

# Optional flags
--no-headtracking          # Disable head tracking
--idle-sleep-timeout-s 300 # Override idle timeout
```

Run from the `src/` directory, or set `PYTHONPATH=src`.

### Docker

```bash
cp .env.example .env
# Edit .env with your credentials
docker compose up --build

# Raspberry Pi
docker compose -f docker-compose.yml -f docker-compose.rpi.yml up --build
```

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

Key constraints to enforce:
- **Ports & Adapters:** orchestrator (`orchestrator.py`) depends only on ports (`ports.py`), never on concrete infra. New integrations require a new port + adapter pair in `adapters/`.
- **State machine:** all transitions go through `StateMachine` events — never mutate state directly.
- **Realtime callbacks:** no blocking work; audio and motion run on separate queues/threads.
- **Modes:** `realtime_mode.py` and `chat_mode.py` are thin composition layers — shared behavior belongs in `RuntimeOrchestrator`.

## Key Conventions

- **Language:** All AI responses default to Brazilian Portuguese (pt-BR) per `src/prompts/identity.md`
- **Typed results:** Use `ReachyActionResult` and `ToolExecutionResult` for all action returns
- **Delegation:** When uncertain, the model uses `delegate_task` (OpenClaw) rather than refusing
- **Home Assistant sensitive domains** (`lock`, `alarm_control_panel`, `cover`, `button`) require explicit user confirmation before executing
- **Refactoring:** Prefer root-cause fixes over compatibility shims; remove dead code when safe

## Deployment

For full Raspberry Pi setup (hardware requirements, OpenClaw, Home Assistant, Reachy daemon, systemd service), see [`docs/raspberry-integration.md`](docs/raspberry-integration.md).

## Configuration

All runtime config is via environment variables (see `.env.example`). Key variables:
- `REACHY_BRIDGE_URL` — Reachy SDK WebSocket endpoint
- `OPENAI_API_KEY` / `OPENAI_REALTIME_MODEL` — OpenAI credentials and model
- `STT_LANGUAGE` — defaults to `pt` (Brazilian Portuguese)
- `OPENCLAW_GATEWAY_URL` / `OPENCLAW_BEARER_TOKEN` — task delegation
- `HOME_ASSISTANT_URL` / `HOME_ASSISTANT_TOKEN` — Home Assistant base URL and token
- `WAKEWORD_ENABLED` / `WAKEWORD_ALIASES` — offline wake word detection
