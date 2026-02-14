# Bridge API Contract (Draft)

This contract defines the message exchange between the Bridge and OpenClaw.

## OpenClaw HTTP API (Bridge -> OpenClaw)

### POST /v1/intent
Send user text and receive an intent + action proposal.

Request body:
- session_id: string
- text: string
- language: string (optional)

Success response (200):
- type: openclaw.intent
- payload: see openclaw.intent

Error response (4xx/5xx):
- type: openclaw.error
- payload: code, message

## Core Envelope

Fields used by all messages:

- id: unique id (string)
- ts: ISO-8601 timestamp (string)
- type: message type (string)
- session_id: session id (string)
- payload: object

## Message Types

### user.text
User text ready (from STT).

Payload:
- text: string
- language: string

### system.wake_word
Wake word detected.

Payload:
- keyword: string

### system.presence
Presence detected or lost.

Payload:
- status: present|lost

### openclaw.intent
OpenClaw intent and action proposal.

Payload:
- intent: string
- action_class: A|B|C|D
- summary: string
- actions: list

### openclaw.error
OpenClaw error.

Payload:
- code: string
- message: string

### bridge.confirmation
Request confirmation from user.

Payload:
- action_class: B|C
- summary: string
- token: string

### bridge.confirmation_response
User confirmation response.

Payload:
- accepted: true|false
- token: string

### bridge.action_result
Result of execution.

Payload:
- ok: true|false
- message: string
