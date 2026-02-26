# Copilot Instructions for dobby-the-claw

## Runtime truth (read this first)
- Current production path is OpenAI Realtime + Reachy SDK/bridge, not OpenClaw intent routing.
- OpenClaw integration is planned only (`docs/api-contract.md` is future-facing).
- There is no separate STT service in runtime; transcription comes from Realtime events.

## Big-picture architecture
- Entry point: `python -m bridge.main` (`src/bridge/main.py`).
- Main loop in realtime mode: `_run_realtime_loop(...)`.
- Core boundaries:
  - `bridge.reachy.realtime_client.OpenAIRealtimeSession`: websocket session, audio in/out, tool-call handling.
  - `bridge.state_machine.StateMachine`: finite-state transitions (`IDLE/LISTENING/THINKING/EXECUTING/CONFIRMING/ERROR`).
  - `bridge.reachy.client.ReachyClient`: action executor; SDK path active when `REACHY_BRIDGE_URL=sdk`.
  - `bridge.reachy.motion.MotionManager` + `bridge.reachy.camera_worker.CameraWorker`: physical behavior loop and tracking.

## Realtime event-to-state mapping
- Speech start callback triggers `Event.WAKE_WORD` and listening gesture.
- User transcript callback triggers `Event.STT_RECEIVED` and think gesture.
- Assistant audio done triggers `Event.RESPONSE_READY`.
- Keep this mapping consistent when changing callbacks in `main.py`.

## Tooling pattern (important)
- Register tools through `ToolRegistry` (`src/bridge/tools/runtime.py`).
- Tool schema must be OpenAI function-compatible (`definition()` returning JSON schema-like parameters).
- Tool execution returns `ToolExecutionResult`; include `image_base64` when sending visual context (see `camera_snapshot`).
- Add new tools under `src/bridge/tools/` and register from `_run_realtime_loop`.

## Developer workflows
- Local run (venv): `python -m bridge.main --mode realtime`
- Docker run: `docker compose up --build`
- Raspberry profile: `docker compose -f docker-compose.yml -f docker-compose.rpi.yml up --build`
- Primary config comes from `.env` / `.env.example` via `BridgeConfig.from_env()` (`src/bridge/config.py`).

## Project-specific conventions
- Prefer explicit typed/dataclass structures when available (`BridgeConfig`, `ToolDefinition`, `ToolExecutionResult`).
- Keep thread/queue boundaries intact (audio queue in `main.py`, command queue in `MotionManager`).
- Preserve low-latency behavior: avoid blocking work in realtime callbacks.
- Follow existing logging style (`logging` with compact state/latency context).

## Integration notes
- OpenAI dependency: `openai>=2.1.0`; Realtime model defaults to `gpt-realtime`.
- Reachy dependency: `reachy-mini>=1.3.1`.
- `REACHY_BRIDGE_URL=sdk` is the implemented path today; non-SDK HTTP client is still TODO in `ReachyClient`.

## Docs consistency guidance
- Use `README.md` + `docs/architecture.md` as the current architecture source.
- Treat `docs/behavior-spec-v1.md` and OpenClaw contract sections as partially future/planned where they conflict with runtime code.
- If changing runtime behavior, update docs in the same PR to keep these files aligned.
