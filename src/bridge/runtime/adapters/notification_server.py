"""HTTP server that receives async webhook notifications on POST /notify."""

from __future__ import annotations

import json
import logging
import queue
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

logger = logging.getLogger(__name__)


class NotificationServer:
    """Receive webhook deliveries on POST /notify and enqueue the summary."""

    def __init__(self, host: str, port: int, notification_queue: Optional[queue.Queue[str]] = None) -> None:
        self._host = host
        self._port = port
        self.queue: queue.Queue[str] = notification_queue if notification_queue is not None else queue.Queue()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start HTTP server in a daemon thread."""
        notification_queue = self.queue

        class _ReusePortHTTPServer(HTTPServer):
            def server_bind(self) -> None:
                try:
                    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except (AttributeError, OSError):
                    pass
                super().server_bind()

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path != "/notify":
                    self.send_response(404)
                    self.end_headers()
                    return
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    payload = json.loads(body)
                    summary = str(payload.get("summary", "")).strip()
                    if summary:
                        notification_queue.put(summary)
                        logger.info("Notification enqueued: %.80s", summary)
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"ok":true}')
                except Exception as exc:
                    logger.warning("Notification parse error: %s", exc)
                    self.send_response(400)
                    self.end_headers()

            def log_message(self, *_: object) -> None:
                pass

        self._server = _ReusePortHTTPServer((self._host, self._port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="notification-server",
        )
        self._thread.start()
        logger.info("Notification server listening on %s:%d", self._host, self._port)

    def stop(self) -> None:
        """Shutdown HTTP server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        logger.info("Notification server stopped")
