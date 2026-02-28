"""Shared runtime helpers for bridge execution modes."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from bridge.reachy.motion import MotionManager
from bridge.state_machine import Event, StateMachine


def apply_event(
    state_machine: StateMachine,
    event: Event,
    motion_manager: Optional[MotionManager],
):
    """Apply a state transition and mirror state to motion manager."""
    state = state_machine.transition(event)
    if motion_manager is not None:
        motion_manager.set_state(state)
    return state


def resample_audio_chunk(chunk: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample a mono float32 audio chunk from `src_rate` to `dst_rate`."""
    if src_rate == dst_rate:
        return chunk
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.size < 2:
        return chunk
    duration = chunk.size / float(src_rate)
    target_size = max(1, int(duration * dst_rate))
    source_x = np.linspace(0.0, 1.0, num=chunk.size, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, num=target_size, dtype=np.float32)
    return np.interp(target_x, source_x, chunk).astype(np.float32)
