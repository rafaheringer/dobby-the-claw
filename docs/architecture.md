# Architecture (Draft)

This document captures the high-level architecture for the Reachy Mini + OpenClaw bridge.

## Goals

- OpenAI Realtime is the active cognitive runtime today.
- Reachy Mini executes physical actions and TTS.
- The bridge coordinates state, safety policy, and IO.
- Local API is used for all inter-process communication.
- OpenClaw integration remains a planned next phase.

## Components

- OpenAI Realtime API: live speech transcription + LLM response + streamed audio output.
- Bridge (Python): state machine, realtime IO orchestration, tool routing.
- Reachy Bridge API: physical motion, gestures, face tracking.
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
6. Bridge calls Reachy Bridge / SDK for gestures and motion.
7. Assistant audio is streamed back to Reachy speaker.

## State Machine

States: IDLE, LISTENING, THINKING, EXECUTING, CONFIRMING, ERROR.
Timeouts: LISTENING=10s, CONFIRMING=15s.

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

- Current production path: Bridge + OpenAI Realtime + Reachy Bridge/SDK.
- Planned path: add OpenClaw intent/planning API and route cognition through it.
