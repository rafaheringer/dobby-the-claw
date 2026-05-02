"""Native local scheduler — fires reminders directly into the notification queue."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


class LocalScheduler:
    """Schedule timed reminders without any external network calls.

    Jobs are persisted to a JSON file so they survive bridge restarts.
    Call restore() once during startup to reload and re-arm persisted jobs.
    """

    def __init__(self, enqueue: Callable[[str], None], jobs_file: Optional[str] = None) -> None:
        self._enqueue = enqueue
        self._jobs_file = jobs_file
        self._timers: Dict[str, threading.Timer] = {}
        self._jobs: Dict[str, dict] = {}  # persistent state, keyed by job_id
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public API

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
            delay = float(delay_seconds)
            self._jobs[job_id] = {"message": message, "kind": "once", "fire_at": time.time() + delay}
            self._save()
            self._arm(job_id, message, delay, repeat=None, cron=None, tz=timezone)
        elif repeat_every_seconds is not None:
            interval = float(repeat_every_seconds)
            self._jobs[job_id] = {"message": message, "kind": "interval", "interval": interval, "tz": timezone, "fire_at": time.time() + interval}
            self._save()
            self._arm(job_id, message, interval, repeat=interval, cron=None, tz=timezone)
        elif cron_expression is not None:
            delay = self._cron_delay(cron_expression, timezone)
            self._jobs[job_id] = {"message": message, "kind": "cron", "cron": cron_expression, "tz": timezone}
            self._save()
            self._arm(job_id, message, delay, repeat=None, cron=cron_expression, tz=timezone)
        return job_id

    def cancel(self, job_id: str) -> bool:
        """Cancel a scheduled reminder. Returns True if found and cancelled."""
        with self._lock:
            timer = self._timers.pop(job_id, None)
            self._jobs.pop(job_id, None)
        if timer is not None:
            timer.cancel()
            logger.info("Reminder cancelled: %s", job_id)
        self._save()
        return timer is not None

    def restore(self) -> None:
        """Load persisted jobs and re-arm them. Call once during startup."""
        if not self._jobs_file:
            return
        saved = self._load_file()
        if not saved:
            return
        now = time.time()
        count = 0
        for job_id, state in saved.items():
            try:
                message = state["message"]
                kind = state["kind"]
                if kind == "once":
                    remaining = max(1.0, state["fire_at"] - now)
                    self._jobs[job_id] = state
                    self._arm(job_id, message, remaining, repeat=None, cron=None, tz="America/Sao_Paulo")
                elif kind == "interval":
                    interval = state["interval"]
                    remaining = max(1.0, state["fire_at"] - now)
                    self._jobs[job_id] = state
                    self._arm(job_id, message, remaining, repeat=interval, cron=None, tz=state.get("tz", "America/Sao_Paulo"))
                elif kind == "cron":
                    tz = state.get("tz", "America/Sao_Paulo")
                    delay = self._cron_delay(state["cron"], tz)
                    self._jobs[job_id] = state
                    self._arm(job_id, message, delay, repeat=None, cron=state["cron"], tz=tz)
                count += 1
            except Exception as exc:
                logger.warning("Could not restore job %s: %s", job_id, exc)
        if count:
            logger.info("Restored %d reminder(s) from %s", count, self._jobs_file)

    # ------------------------------------------------------------------ #
    # Internal

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
                with self._lock:
                    if job_id in self._jobs:
                        self._jobs[job_id]["fire_at"] = time.time() + repeat
                self._save()
                self._arm(job_id, message, repeat, repeat=repeat, cron=None, tz=tz)
            elif cron is not None:
                try:
                    next_delay = self._cron_delay(cron, tz)
                    self._arm(job_id, message, next_delay, repeat=None, cron=cron, tz=tz)
                except Exception as exc:
                    logger.error("Cron reschedule failed job=%s: %s", job_id, exc)
                    with self._lock:
                        self._jobs.pop(job_id, None)
                    self._save()
            else:
                with self._lock:
                    self._jobs.pop(job_id, None)
                self._save()

        t = threading.Timer(delay, _fire)
        t.daemon = True
        with self._lock:
            self._timers[job_id] = t
        t.start()
        logger.info("Reminder armed job=%s delay=%.0fs", job_id, delay)

    def _save(self) -> None:
        if not self._jobs_file:
            return
        with self._lock:
            data = dict(self._jobs)
        try:
            path = os.path.expanduser(self._jobs_file)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as exc:
            logger.error("Failed to save scheduler jobs to %s: %s", self._jobs_file, exc)

    def _load_file(self) -> dict:
        path = os.path.expanduser(self._jobs_file)  # type: ignore[arg-type]
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as exc:
            logger.warning("Failed to load scheduler jobs from %s: %s", path, exc)
            return {}

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
