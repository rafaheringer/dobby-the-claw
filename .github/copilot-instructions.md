# Copilot Instructions for dobby-the-claw

## Runtime truth (read this first)
- Current production path is OpenAI Realtime + Reachy SDK/bridge, not OpenClaw intent routing.
- There is no separate STT service in runtime; transcription comes from Realtime events.

## Big-picture architecture
- Entry point: `python -m bridge.main` (`src/bridge/main.py`).
- Runtime orchestration core: `RuntimeOrchestrator` (`src/bridge/runtime/orchestrator.py`).
- Core boundaries:
  - `reachy.realtime_client.OpenAIRealtimeSession`: websocket session, audio in/out, tool-call handling.
  - `bridge.state_machine.StateMachine`: finite-state transitions (`IDLE/LISTENING/THINKING/EXECUTING/CONFIRMING/ERROR`).
  - `reachy.client.ReachyClient`: action executor; SDK path active when `REACHY_BRIDGE_URL=sdk`.
  - `reachy.motion.MotionManager` + `reachy.camera_worker.CameraWorker`: physical behavior loop and tracking.

## Required architecture standards (always)
- Keep runtime core decoupled via **Ports & Adapters**.
  - Ports live in `src/bridge/runtime/ports.py`.
  - Concrete adapters live in `src/bridge/runtime/adapters/`.
  - `RuntimeOrchestrator` must depend on ports, never on concrete infra clients directly.
- Keep chat/realtime modes thin.
  - `chat_mode.py` and `realtime_mode.py` are composition/bootstrapping entrypoints only.
  - Shared behavior belongs in `RuntimeOrchestrator`.
- State transitions must go through `StateMachine` events.
  - Do not mutate state directly from runtime loops.
  - Preserve callback mapping: speech_start -> `WAKE_WORD`, transcript -> `STT_RECEIVED`, assistant_audio_done -> `RESPONSE_READY`.
- Use typed commands/results for Reachy actions.
  - Action contracts are defined in `src/reachy/actions.py`.
  - Action execution result contract is `ReachyActionResult` in `src/reachy/results.py`.
  - Avoid reintroducing dict-based action APIs unless explicitly requested.
- Keep latency-safe boundaries.
  - No blocking/heavy work inside realtime callbacks.
  - Respect queue/thread boundaries (audio queue, motion command queue).

## Refactor and maintenance policy
- Prefer root-cause refactors over additive compatibility layers.
- Do not keep legacy compatibility paths unless explicitly requested.
- When replacing an API internally, remove unused legacy code in the same PR when safe.
- Keep imports, type hints, and naming consistent after refactors.
- Validate changes with at least compile/syntax checks before finalizing.

## Realtime event-to-state mapping
- Speech start callback triggers `Event.WAKE_WORD` and listening gesture.
- User transcript callback triggers `Event.STT_RECEIVED` and think gesture.
- Assistant audio done triggers `Event.RESPONSE_READY`.
- Keep this mapping consistent when changing callbacks in `main.py`.

## Tooling pattern (important)
- Register tools through `ToolRegistry` (`src/bridge/tools/runtime.py`).
- Tool schema must be OpenAI function-compatible (`definition()` returning JSON schema-like parameters).
- Tool execution returns `ToolExecutionResult`; include `image_base64` when sending visual context (see `camera_snapshot`).
- Add new tools under `src/bridge/tools/` and register in runtime composition path.

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
- If changing runtime behavior, update docs in the same PR to keep these files aligned.
- If changing runtime boundaries or contracts, also update this file (`.github/copilot-instructions.md`).
