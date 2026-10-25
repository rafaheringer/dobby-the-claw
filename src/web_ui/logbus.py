"""Log records for the web UI: tagged with a category the page can filter on, with secrets masked."""

from __future__ import annotations

import logging
import queue
import re
from dataclasses import dataclass

# First matching prefix wins. Logger names come from the module that logs; the orchestrator
# logs through the root logger, so "root" is its category.
_CATEGORIES: tuple[tuple[str, str], ...] = (
    ("bridge.conversation", "Conversa"),
    ("reachy.gemini_live_client", "Conversa"),
    ("reachy.realtime_client", "Conversa"),
    ("reachy.camera_worker", "Visão"),
    ("reachy.person_detector", "Visão"),
    ("reachy.face_recognizer", "Visão"),
    ("reachy.presence", "Visão"),
    ("reachy.finger_antenna_controller", "Visão"),
    ("bridge.runtime.speaker_id", "Visão"),
    ("bridge.runtime.presence_notices", "Visão"),
    ("reachy.voice_recognizer", "Áudio"),
    ("reachy.speech_tapper", "Áudio"),
    ("bridge.runtime.wakeword", "Áudio"),
    ("reachy_mini.media", "Áudio"),
    ("reachy.motion", "Movimento"),
    ("reachy.head_roll_controller", "Movimento"),
    ("reachy_mini", "Movimento"),
    ("bridge.tools", "Ferramentas"),
    ("homeassistant", "Ferramentas"),
    ("bridge.speaker_memory", "Ferramentas"),
    ("root", "Sistema"),
)

CATEGORIES: tuple[str, ...] = ("Conversa", "Visão", "Áudio", "Movimento", "Ferramentas", "Sistema", "Outros")

# Logs travel to a browser, possibly over the internet: never let credentials through.
_SECRETS = re.compile(
    r"(AIza[0-9A-Za-z_\-]{20,}"
    r"|sk-[A-Za-z0-9_\-]{16,}"
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"
    r"|(?i:bearer)\s+[A-Za-z0-9._\-]{12,}"
    r"|(?i:(?:api[_-]?key|token|secret|password|passwd)[\"']?\s*[=:]\s*)[\"']?[^\s\"',;&]{6,})"
)


def redact(text: str) -> str:
    return _SECRETS.sub("«oculto»", text)


def categorize(logger_name: str) -> str:
    for prefix, category in _CATEGORIES:
        if logger_name == prefix or logger_name.startswith(prefix + "."):
            return category
    return "Outros"


@dataclass(frozen=True)
class LogLine:
    created: float
    level: int
    category: str
    source: str
    message: str


class _QueueHandler(logging.Handler):
    def __init__(self, sink: "queue.Queue[LogLine]") -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._sink.put_nowait(
                LogLine(record.created, record.levelno, categorize(record.name), record.name, record.getMessage())
            )
        except Exception:
            pass  # logging must never raise into the caller


def install_log_queue() -> "queue.Queue[LogLine]":
    """Route every log record into a queue the web UI drains (call before the runtime starts)."""
    sink: "queue.Queue[LogLine]" = queue.Queue(maxsize=20_000)
    handler = _QueueHandler(sink)
    handler.setLevel(logging.DEBUG)  # the root level decides what exists; the page filters the rest
    logging.getLogger().addHandler(handler)
    return sink
