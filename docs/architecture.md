# Architecture

This document captures the current runtime architecture for Reachy Mini + Bridge + OpenAI Realtime + OpenClaw delegation + Home Assistant control.

## Goals

- Keep Bridge as the single user-facing interface.
- Use OpenAI Realtime for low-latency voice IO and assistant generation.
- Use Reachy Mini as embodiment (motion + speaker + mic via SDK).
- Delegate complex/long-running tasks to OpenClaw through a tool call.
- Control home devices through Home Assistant tool calls.

## Components

- OpenAI Realtime API: live transcription, model responses, streamed assistant audio.
- Bridge Runtime (Python): state machine, orchestration, tool routing, wakeword/idle logic.
- Reachy SDK (active path): motion, gestures, camera/audio media.
- Camera worker can provide index-finger-based antenna override (1 finger -> both antennas, 2 fingers -> left/right split), temporarily overriding antenna breathing sway while active.
- OpenClaw Gateway (WebSocket): delegated task execution.
- Home Assistant (WebSocket API): entity discovery and service execution.

## Runtime Core Boundaries

- `src/bridge/runtime/orchestrator.py`: application service orchestrating sessions, state, media and tools.
- `src/bridge/runtime/ports.py`: abstractions for sessions, robot actions, tools and media.
- `src/bridge/runtime/adapters/`: concrete adapters (`realtime_session`, `reachy_actions`, `reachy_media`, `tool_runtime`, `openclaw_gateway`).
- `src/bridge/tools/`: tool implementations and contracts (`camera_snapshot`, `delegate_task`, `go_to_sleep`, Home Assistant tools).
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
5. If Home Assistant tools are called:
   - `discover_home_devices` fetches entities (`get_states`) and service schemas (`get_services`).
   - `control_home_device` performs `call_service` for the selected target.
   - Sensitive domains require explicit confirmation via tool argument.
6. If `go_to_sleep` is called, the runtime enters sleep mode and waits for the offline wake word.
7. Realtime model produces final user-facing response.
8. Assistant audio is streamed to Reachy speaker.

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

- Base assistant identity prompt is loaded from `src/prompts/identity.txt`.
- Tool runtime aggregates per-tool `runtime_guardrail` entries.
- Orchestrator appends aggregated tool guardrails to session instructions generically (without hard-coding specific tool names).
