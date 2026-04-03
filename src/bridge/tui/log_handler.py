"""Logging handler that routes records into a thread-safe queue for the TUI."""

from __future__ import annotations

import logging
import queue


class TUILogHandler(logging.Handler):
    """Push log records into a Queue so the TUI can drain them on its own timer."""

    def __init__(self, log_queue: "queue.Queue[logging.LogRecord]") -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put_nowait(record)
        except Exception:
            pass
