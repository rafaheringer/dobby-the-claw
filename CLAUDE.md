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

### Lint / Format

```bash
ruff check .    # Lint
ruff format .   # Format (100-char line limit)
```

There is no automated test suite; validation is manual/integration-based.

## Architecture

### Ports & Adapters (enforced pattern)

The runtime is strictly decoupled:
- **Ports** (`src/bridge/runtime/ports.py`) — abstract interfaces: `ConversationSessionPort`, `RobotActionsPort`, `ToolRuntimePort`, `MediaIOPort`
- **Adapters** (`src/bridge/runtime/adapters/`) — concrete implementations wiring to OpenAI, Reachy SDK, OpenClaw
- **Orchestrator** (`src/bridge/runtime/orchestrator.py`) — depends only on ports, never on concrete infra

Never add direct infra calls inside the orchestrator or state machine. New integrations go through a new port + adapter pair.

### State Machine

`src/bridge/state_machine.py` governs all interaction flow:

```
IDLE → LISTENING → THINKING → EXECUTING → IDLE
                             ↘ DELEGATING ↗
```

All state transitions happen exclusively through `StateMachine` events (e.g., `WAKE_WORD`, `STT_RECEIVED`, `RESPONSE_READY`, `DELEGATION_STARTED`). Never mutate state directly.

### Modes

- `realtime_mode.py` — thin composition layer for OpenAI Realtime voice pipeline
- `chat_mode.py` — thin composition layer for terminal text input
- Shared behavior lives in `RuntimeOrchestrator`

### Tools

`src/bridge/tools/` — each tool implements `definition()` (OpenAI JSON schema) and `execute()`. Registered in a `ToolRegistry`. Built-in tools: `camera_snapshot`, `dance`, `express_emotion`, `go_to_sleep`, `openclaw_delegate`, `discover_home_devices`, `control_home_device`.

### Reachy Integration

`src/reachy/` contains:
- `client.py` — typed action executor
- `motion.py` — motion command queue (worker thread; don't block realtime callbacks)
- `camera_worker.py` — MediaPipe hand/finger tracking in a background thread
- `finger_antenna_controller.py` / `head_roll_controller.py` — peripheral controllers

Audio queue and motion command queue run on separate threads. Never perform blocking work in realtime WebSocket callbacks.

## Key Conventions

- **Language:** All AI responses default to Brazilian Portuguese (pt-BR) per `src/prompts/identity.txt`
- **Typed results:** Use `ReachyActionResult` and `ToolExecutionResult` for all action returns
- **Delegation:** When uncertain, the model uses `delegate_task` (OpenClaw) rather than refusing
- **Home Assistant sensitive domains** (`lock`, `alarm_control_panel`, `cover`, `button`) require explicit user confirmation before executing
- **Refactoring:** Prefer root-cause fixes over compatibility shims; remove dead code when safe

## Configuration

All runtime config is via environment variables (see `.env.example`). Key variables:
- `REACHY_BRIDGE_URL` — Reachy SDK WebSocket endpoint
- `OPENAI_API_KEY` / `OPENAI_REALTIME_MODEL` — OpenAI credentials and model
- `STT_LANGUAGE` — defaults to `pt` (Brazilian Portuguese)
- `OPENCLAW_GATEWAY_URL` / `OPENCLAW_BEARER_TOKEN` — task delegation
- `HA_URL` / `HA_TOKEN` — Home Assistant WebSocket
- `WAKEWORD_ENABLED` / `WAKEWORD_ALIASES` — offline wake word detection
