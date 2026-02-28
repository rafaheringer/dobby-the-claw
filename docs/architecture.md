# Architecture

This document captures the current runtime architecture for Reachy Mini + Bridge + OpenAI Realtime + OpenClaw delegation.

## Goals

- Keep Bridge as the single user-facing interface.
- Use OpenAI Realtime for low-latency voice IO and assistant generation.
- Use Reachy Mini as embodiment (motion + speaker + mic via SDK).
- Delegate complex/long-running tasks to OpenClaw through a tool call.

## Components

- OpenAI Realtime API: live transcription, model responses, streamed assistant audio.
- Bridge Runtime (Python): state machine, orchestration, tool routing, wakeword/idle logic.
- Reachy SDK (active path): motion, gestures, camera/audio media.
- OpenClaw Gateway (WebSocket): delegated task execution.

## Runtime Core Boundaries

- `src/bridge/runtime/orchestrator.py`: application service orchestrating sessions, state, media and tools.
- `src/bridge/runtime/ports.py`: abstractions for sessions, robot actions, tools and media.
- `src/bridge/runtime/adapters/`: concrete adapters (`realtime_session`, `reachy_actions`, `reachy_media`, `tool_runtime`, `openclaw_gateway`).
- `src/bridge/tools/`: tool implementations and contracts (`camera_snapshot`, `delegate_task`).
- `ToolDefinition.runtime_guardrail`: optional per-tool runtime policy text aggregated by tool runtime and appended to session instructions.

## Data Flow (Simplified)

1. Audio input enters wakeword/voice pipeline.
2. Realtime callbacks transition state (`WAKE_WORD`, `STT_RECEIVED`, `RESPONSE_READY`).
3. Model may call tools.
4. If `delegate_task` is called:
   - Bridge informs user to wait (instruction-level behavior in identity prompt).
   - Bridge enters `DELEGATING` and triggers waiting gesture.
   - Tool calls OpenClaw Gateway over WebSocket RPC (`connect.challenge` + `connect`, then `chat.send`/`chat.history`) and waits for final text.
   - Tool result returns to Realtime function output.
5. Realtime model produces final user-facing response (can paraphrase OpenClaw output).
6. Assistant audio is streamed to Reachy speaker.

## State Machine

States: `IDLE`, `LISTENING`, `THINKING`, `DELEGATING`, `EXECUTING`, `CONFIRMING`, `ERROR`.

Important transitions:

- `THINKING` + `DELEGATION_STARTED` -> `DELEGATING`
- `DELEGATING` + `DELEGATION_DONE` -> `THINKING`
- Existing realtime transitions remain intact (`WAKE_WORD`, `STT_RECEIVED`, `RESPONSE_READY`, etc.).

## Configuration

Primary runtime configuration comes from `.env` via `BridgeConfig.from_env()`.

OpenClaw-specific values:

- `OPENCLAW_ENABLED`
- `OPENCLAW_WS_URL`
- `OPENCLAW_BEARER_TOKEN`
- `OPENCLAW_TIMEOUT_S`

## Implementation Status

- Current production path: Bridge + OpenAI Realtime + Reachy SDK + OpenClaw delegation tool.
- Non-SDK Reachy HTTP path remains TODO.

## Prompt/Policy Composition

- Base assistant identity prompt is loaded from `src/prompts/identity.txt`.
- Tool runtime aggregates per-tool `runtime_guardrail` entries.
- Orchestrator appends aggregated tool guardrails to session instructions generically (without hard-coding specific tool names).
