"""Async helpers for checking service health and Raspberry Pi stats."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


async def check_service_up(ws_url: str, timeout: float = 2.0) -> bool:
    """Return True if the host:port behind ws_url accepts a TCP connection."""
    try:
        parsed = urlparse(ws_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme in ("wss", "https") else 80)
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except Exception:
        return False


async def get_pi_stats(
    host: str = "raspberrypi.local",
    user: str = "dobby",
) -> dict | None:
    """Return {cpu_load, mem_used_gb, mem_total_gb} from the Pi via SSH, or None."""
    if not host:
        return None
    # /proc/meminfo values are in kB; /proc/loadavg gives 1-min load average
    cmd = (
        "awk '/^MemTotal/{t=$2}/^MemAvailable/{a=$2}END{print t-a,t}' /proc/meminfo "
        "&& awk '{print $1}' /proc/loadavg"
    )
    try:
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        "ssh",
                        "-o", "ConnectTimeout=3",
                        "-o", "BatchMode=yes",
                        "-o", "StrictHostKeyChecking=no",
                        f"{user}@{host}",
                        cmd,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=8,
                ),
            ),
            timeout=10.0,
        )
        if result.returncode != 0:
            logger.debug("Pi SSH failed (rc=%d): %s", result.returncode, result.stderr.strip())
            return None
        lines = result.stdout.strip().splitlines()
        if len(lines) < 2:
            logger.debug("Pi SSH unexpected output: %r", result.stdout)
            return None
        used_kb, total_kb = map(int, lines[0].split())
        load = float(lines[1])
        return {
            "cpu_load": load,
            "mem_used_gb": used_kb / (1024 ** 2),
            "mem_total_gb": total_kb / (1024 ** 2),
        }
    except Exception as exc:
        logger.debug("Pi SSH exception: %s", exc)
        return None
