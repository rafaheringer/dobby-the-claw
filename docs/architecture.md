# Architecture

This document captures the current runtime architecture for Reachy Mini + Bridge + OpenAI Realtime + OpenClaw delegation + Home Assistant control.

## Goals

- Keep Bridge as the single user-facing interface.
- Use OpenAI Realtime for low-latency voice IO and assistant generation.
- Use Reachy Mini as embodiment (motion + speaker + mic via SDK).
- Delegate complex/long-running tasks to OpenClaw through a tool call.
- Control home devices through Home Assistant tool calls.

## Components

- **OpenAI Realtime API**: live transcription, model responses, streamed assistant audio.
- **Bridge Runtime** (Python): state machine, orchestration, tool routing, wakeword/idle logic.
- **Reachy SDK** (active path): motion, gestures, camera/audio media.
- **CameraWorker**: continuous frame loop for face tracking, head steering, and speaker identification. Provides index-finger-based antenna override (1 finger → both antennas, 2 fingers → left/right split). AudioDoA is lazy-initialized on first use to avoid GStreamer pipeline conflicts.
- **FaceRecognizer** (InsightFace `buffalo_s`): enrolls and identifies speakers by face embedding. Profiles stored as `.npy` files under `SPEAKER_ID_PROFILES_DIR`.
- **SpeakerMemory** (mem0 + Qdrant local + SQLite): extracts lasting facts from each session using an LLM and persists them under `SPEAKER_MEMORY_DIR`. Facts are injected into the session identity prompt on the next wakeup.
- **NotificationServer**: lightweight HTTP server that receives async external notifications and delivers them into the active realtime session.
- **LocalScheduler**: native in-process scheduler for reminders, timers, interval jobs, and cron-based recurring notifications. Persists jobs to disk and restores them on startup.
- **OpenClaw Gateway** (WebSocket): delegated task execution for complex or long-running tasks.
- **Home Assistant** (WebSocket API): entity discovery and service execution.

## Runtime Core Boundaries

- `src/bridge/runtime/orchestrator.py`: application service orchestrating sessions, state, media and tools.
- `src/bridge/runtime/ports.py`: abstractions for sessions, robot actions, tools and media.
- `src/bridge/runtime/adapters/`: concrete adapters (`realtime_session`, `reachy_actions`, `reachy_media`, `tool_runtime`, `openclaw_gateway`).
- `src/bridge/tools/`: tool implementations and contracts (`camera_snapshot`, `delegate_task`, `go_to_sleep`, `dance`, `express_emotion`, Home Assistant tools).
- `ToolDefinition.runtime_guardrail`: optional per-tool runtime policy text aggregated by tool runtime and appended to session instructions.

## Data Flow (Simplified)

1. **Startup / wakeup**: orchestrator runs `identify_current_speaker()` before starting the session. If a known face is detected, their name and any persisted memories (from `SpeakerMemory.load()`) are injected into the session identity prompt.
2. Audio input enters wakeword/voice pipeline.
3. On `speech_start`: a background thread calls `identify_current_speaker()` via AudioDoA steering + face recognition. If the speaker changed since the last turn, a SYSTEM NOTICE is injected into the realtime session.
4. Realtime callbacks transition state (`WAKE_WORD`, `STT_RECEIVED`, `RESPONSE_READY`).
5. Model may call tools:
   - `delegate_task`: Bridge enters `DELEGATING`, calls OpenClaw Gateway over WebSocket RPC, waits for result.
   - `discover_home_devices` / `control_home_device`: HA entity lookup and service execution. Sensitive domains require explicit confirmation.
   - `go_to_sleep`: runtime enters sleep mode, saves session to SpeakerMemory, waits for offline wake word.
   - `express_emotion`: Reachy plays a recorded move from `pollen-robotics/reachy-mini-emotions-library`.
   - `enroll_speaker`: captures 4 camera frames, registers face embedding under the given name.
   - `remember_fact`: saves an explicit user-stated fact to SpeakerMemory immediately (background thread).
   - `create_reminder` / `cancel_reminder`: schedules/cancels native local reminder jobs, including one-time, interval, and cron-based recurrence.
   - `take_photo`: captures a frame and encodes it for the model.
   - `dance`: plays a choreography from the dances library.
   - `get_weather`: fetches current conditions and daily forecast from Open-Meteo for a given city (default: Rio de Janeiro). No API key required.
6. Realtime model produces final user-facing response.
7. Assistant audio is streamed to Reachy speaker.
8. On sleep entry: `SpeakerMemory.save_async()` extracts lasting facts from the session conversation and persists them.

## State Machine

States: `IDLE`, `LISTENING`, `THINKING`, `DELEGATING`, `EXECUTING`, `CONFIRMING`, `ERROR`.

Important transitions:

- `THINKING` + `DELEGATION_STARTED` -> `DELEGATING`
- `DELEGATING` + `DELEGATION_DONE` -> `THINKING`
- Existing realtime transitions remain intact (`WAKE_WORD`, `STT_RECEIVED`, `RESPONSE_READY`, etc.).

## Configuration

Primary runtime configuration comes from `.env` via `BridgeConfig.from_env()`.


## Implementation Status

- Current production path: Bridge + OpenAI Realtime + Reachy SDK + OpenClaw delegation tool.
- Non-SDK Reachy HTTP path remains TODO.

## Prompt/Policy Composition

- Base assistant identity prompt is loaded from `src/prompts/identity.md`.
- Tool runtime aggregates per-tool `runtime_guardrail` entries and appends them as `## RUNTIME TOOL GUARDRAILS` in the session instructions.
- Orchestrator appends a `## FALANTE ATUAL` section when a speaker is identified, and a `## O QUE VOCÊ JÁ SABE SOBRE [NAME]` section when memories exist.
- Home Assistant entity catalog is appended as `## DISPOSITIVOS HOME ASSISTANT DISPONÍVEIS` when HA is enabled.
