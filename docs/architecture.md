# Architecture (Draft)

This document captures the high-level architecture for the Reachy Mini + OpenClaw bridge.

## Goals

- OpenAI Realtime is the active cognitive runtime today.
- Reachy Mini executes physical actions and TTS.
- The bridge coordinates state, safety policy, and IO.
- Runtime communication uses OpenAI Realtime (WebSocket) plus local Reachy SDK integration.
- OpenClaw integration remains a planned next phase.

## Components

- OpenAI Realtime API: live speech transcription + LLM response + streamed audio output.
- Bridge (Python): state machine, realtime IO orchestration, tool routing.
- Reachy SDK (active path): physical motion, gestures, camera/audio media.
- Reachy Bridge HTTP API (planned/non-SDK path): not the active runtime path today.
- Optional future component: OpenClaw API (HTTP/WebSocket) for intent/planning.

## Runtime Core Boundaries

- `src/bridge/runtime/orchestrator.py` is the runtime application service and orchestration core.
- `src/bridge/runtime/ports.py` defines runtime ports (conversation session, robot actions, tools, media IO).
- `src/bridge/runtime/adapters/` contains concrete implementations for current infrastructure:
	- `realtime_session.py` (OpenAI Realtime session factory)
	- `reachy_actions.py` (ReachyClient action mapping)
	- `tool_runtime.py` (ToolRegistry-backed tool execution)
	- `reachy_media.py` (Reachy SDK media input/output mapping)
- `chat_mode.py` and `realtime_mode.py` are thin adapters that configure and run the orchestrator.

## Data Flow (Simplified)

1. Audio input -> Wake word detection.
2. If wake word, start LISTENING.
3. Audio stream is sent to OpenAI Realtime.
4. Realtime returns user transcript + assistant response.
5. Bridge executes tools/actions as needed.
6. Bridge calls Reachy SDK for gestures and motion (active path via `REACHY_BRIDGE_URL=sdk`).
7. Assistant audio is streamed back to Reachy speaker.

## State Machine

States: IDLE, LISTENING, THINKING, EXECUTING, CONFIRMING, ERROR.

Current behavior: transitions are event-driven by realtime callbacks (`WAKE_WORD`, `STT_RECEIVED`, `RESPONSE_READY`, etc.).
Note: explicit LISTENING/CONFIRMING timers are not currently implemented as active runtime timers.
Separate from state transitions, idle sleep is configurable via `IDLE_SLEEP_TIMEOUT_S`.

Note: these states are currently driven by realtime callbacks/events, not by OpenClaw intent responses.

## Safety and Memory

- Never store secrets (passwords, tokens, keys, 2FA codes).
- Store only preferences, config, and summarized action history.
- Retention: audit log 90 days, conversation 7 days optional.

## Deployment

- Docker Compose for local dev on Linux.
- Optional Raspberry profile with CPU/RAM limits.
- Volumes for logs and optional memory store.

## Implementation Status

- Current production path: Bridge + OpenAI Realtime + Reachy SDK (`REACHY_BRIDGE_URL=sdk`).
- Non-SDK Reachy HTTP client path remains TODO.
- Planned path: add OpenClaw intent/planning API and route cognition through it.
