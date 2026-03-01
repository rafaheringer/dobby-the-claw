"""Typed result contracts for Reachy action execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReachyActionResult:
    """Structured outcome for one Reachy action execution."""

    ok: bool
    message: str
    path: str | None = None
