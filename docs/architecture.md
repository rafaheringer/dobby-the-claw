# Architecture (Draft)

This document captures the high-level architecture for the Reachy Mini + OpenClaw bridge.

## Goals

- OpenClaw is the only cognitive decision maker.
- Reachy Mini executes physical actions and TTS.
- The bridge coordinates state, safety policy, and IO.
- Local API is used for all inter-process communication.

## Components

- OpenClaw API (HTTP/WebSocket): cognition, planning, intents.
- Bridge (Python): state machine, policy enforcement, IO orchestration.
- Reachy Bridge API: physical motion, gestures, face tracking.
- STT/TTS (OpenAI): speech to text, text to speech.

## Data Flow (Simplified)

1. Audio input -> Wake word detection.
2. If wake word, start LISTENING.
3. STT -> text.
4. Bridge sends user text to OpenClaw API.
5. OpenClaw returns intent + action proposal.
6. Bridge enforces action class policy and confirmation.
7. Bridge calls Reachy Bridge for gestures/motion.
8. Bridge calls TTS for speech output.

## State Machine

States: IDLE, LISTENING, THINKING, EXECUTING, CONFIRMING, ERROR.
Timeouts: LISTENING=10s, CONFIRMING=15s.

## Safety and Memory

- Never store secrets (passwords, tokens, keys, 2FA codes).
- Store only preferences, config, and summarized action history.
- Retention: audit log 90 days, conversation 7 days optional.

## Deployment

- Docker Compose for local dev on Linux.
- Optional Raspberry profile with CPU/RAM limits.
- Volumes for logs and optional memory store.
