"""Robot actions adapter backed by ReachyClient."""

from __future__ import annotations

import logging
import time
import wave
from pathlib import Path

import numpy as np

from reachy.actions import (
    AntennaCycleGestureAction,
    ListeningGestureAction,
    ThinkGestureAction,
)
from reachy.client import ReachyClient
from reachy.motion import MotionManager
from reachy.touch_moves import PetMove, TapMove

from bridge.runtime.ports import RobotActionsPort

logger = logging.getLogger(__name__)

_EMOTIONS_DATASET = "pollen-robotics/reachy-mini-emotions-library"
# (kind, asleep) -> emotion sound; asleep sounds are played quieter.
_TOUCH_SOUNDS = {
    ("pet", False): "loving1",
    ("tap", False): "surprised1",
    ("pet", True): "relief1",
    ("tap", True): "oops1",
}
_ASLEEP_GAIN = 0.35
_QUIET_DIR = Path("~/.dobby/touch_sounds").expanduser()


class ReachyRobotActions(RobotActionsPort):
    """Translate runtime action intents into ReachyClient calls."""

    def __init__(self, reachy: ReachyClient, motion_manager: MotionManager | None = None) -> None:
        """Store Reachy client (and motion manager for touch poses)."""
        self._reachy = reachy
        self._motion_manager = motion_manager
        self._pet_move: PetMove | None = None
        self._pet_move_started = 0.0
        self._sound_dir: Path | None = None

    def gesture_listening(self) -> None:
        """Trigger listening gesture on Reachy."""
        _ = self._reachy.execute_typed_action(ListeningGestureAction())

    def gesture_think(self) -> None:
        """Trigger thinking gesture on Reachy."""
        _ = self._reachy.execute_typed_action(ThinkGestureAction())

    def gesture_delegating(self) -> None:
        """Trigger delegating gesture used during external task delegation."""
        _ = self._reachy.execute_typed_action(
            AntennaCycleGestureAction(amplitude_rad=0.16, cycles=3, duration_s=0.22)
        )

    def wake_up(self) -> None:
        """Wake Reachy from sleep posture."""
        self._reachy.wake_up()

    def goto_sleep(self) -> None:
        """Send Reachy to sleep posture."""
        self._reachy.goto_sleep()

    def disable_motors(self) -> None:
        """Cut torque on all motors."""
        self._reachy.disable_motors()

    def enable_motors(self) -> None:
        """Re-energize all motors."""
        self._reachy.enable_motors()

    def react_touch(self, kind: str, phase: str, asleep: bool) -> float:
        """React to a pet/tap; return the length of the sound started (0.0 if none)."""
        try:
            if phase == "end":
                if kind == "pet" and self._pet_move is not None:
                    self._pet_move.release(time.monotonic() - self._pet_move_started)
                    self._pet_move = None
                return 0.0
            if not asleep and self._motion_manager is not None:
                if kind == "pet":
                    self._pet_move = PetMove()
                    self._pet_move_started = time.monotonic()
                    self._motion_manager.queue_move(self._pet_move)
                else:
                    self._motion_manager.queue_move(TapMove())
            return self._play_touch_sound(kind, asleep)
        except Exception as exc:
            logger.warning("Touch reaction %s/%s failed: %s", kind, phase, exc)
            return 0.0

    def _play_touch_sound(self, kind: str, asleep: bool) -> float:
        media = self._reachy.get_sdk_instance().media
        path = self._touch_sound_path(_TOUCH_SOUNDS[(kind, asleep)], asleep)
        with wave.open(str(path)) as wav:
            duration = wav.getnframes() / wav.getframerate()
        media.play_sound(str(path))
        return duration

    def _touch_sound_path(self, name: str, asleep: bool) -> Path:
        if self._sound_dir is None:
            from huggingface_hub import snapshot_download

            self._sound_dir = Path(
                snapshot_download(_EMOTIONS_DATASET, repo_type="dataset", local_files_only=True)
            )
        source = self._sound_dir / f"{name}.wav"
        if not asleep:
            return source
        quiet = _QUIET_DIR / f"{name}.wav"
        if not quiet.exists():
            _write_scaled_wav(source, quiet, _ASLEEP_GAIN)
        return quiet


def _write_scaled_wav(source: Path, target: Path, gain: float) -> None:
    """Write a copy of a 16-bit PCM wav with its amplitude scaled by `gain`."""
    with wave.open(str(source)) as wav:
        params = wav.getparams()
        pcm = np.frombuffer(wav.readframes(params.nframes), dtype=np.int16)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as out:
        out.setparams(params)
        out.writeframes((pcm * gain).astype(np.int16).tobytes())
