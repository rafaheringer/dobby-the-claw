"""Native local scheduler — fires reminders directly into the notification queue."""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class LocalScheduler:
    """Schedule timed reminders without any external network calls."""

    def __init__(self, enqueue: Callable[[str], None]) -> None:
        self._enqueue = enqueue
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def schedule(
        self,
        message: str,
        *,
        delay_seconds: Optional[float] = None,
        repeat_every_seconds: Optional[float] = None,
        cron_expression: Optional[str] = None,
        timezone: str = "America/Sao_Paulo",
    ) -> str:
        """Schedule a reminder and return a job_id."""
        job_id = str(uuid.uuid4())
        if delay_seconds is not None:
            self._arm(job_id, message, float(delay_seconds), repeat=None, cron=None, tz=timezone)
        elif repeat_every_seconds is not None:
            interval = float(repeat_every_seconds)
            self._arm(job_id, message, interval, repeat=interval, cron=None, tz=timezone)
        elif cron_expression is not None:
            delay = self._cron_delay(cron_expression, timezone)
            self._arm(job_id, message, delay, repeat=None, cron=cron_expression, tz=timezone)
        return job_id

    def cancel(self, job_id: str) -> bool:
        """Cancel a scheduled reminder. Returns True if found and cancelled."""
        with self._lock:
            timer = self._timers.pop(job_id, None)
        if timer is not None:
            timer.cancel()
            logger.info("Reminder cancelled: %s", job_id)
            return True
        return False

    # ------------------------------------------------------------------ #

    def _arm(
        self,
        job_id: str,
        message: str,
        delay: float,
        *,
        repeat: Optional[float],
        cron: Optional[str],
        tz: str,
    ) -> None:
        def _fire() -> None:
            with self._lock:
                if job_id not in self._timers:
                    return  # cancelled between arm and fire
                del self._timers[job_id]

            logger.info("Reminder fired job=%s: %.80s", job_id, message)
            self._enqueue(message)

            if repeat is not None:
                self._arm(job_id, message, repeat, repeat=repeat, cron=None, tz=tz)
            elif cron is not None:
                try:
                    next_delay = self._cron_delay(cron, tz)
                    self._arm(job_id, message, next_delay, repeat=None, cron=cron, tz=tz)
                except Exception as exc:
                    logger.error("Cron reschedule failed job=%s: %s", job_id, exc)

        t = threading.Timer(delay, _fire)
        t.daemon = True
        with self._lock:
            self._timers[job_id] = t
        t.start()
        logger.info("Reminder armed job=%s delay=%.0fs", job_id, delay)

    @staticmethod
    def _cron_delay(cron_expr: str, timezone: str) -> float:
        try:
            from croniter import croniter  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError("croniter is required for cron-based reminders: pip install croniter") from exc
        import datetime
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]
        tz = ZoneInfo(timezone)
        now = datetime.datetime.now(tz)
        it = croniter(cron_expr, now)
        next_dt: datetime.datetime = it.get_next(datetime.datetime)
        return (next_dt - now).total_seconds()
