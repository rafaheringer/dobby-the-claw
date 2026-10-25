"""Probe recordings through the real detector, gate and orchestrator wiring (robot and model faked)."""

from pathlib import Path

import numpy as np

from bridge.runtime.orchestrator import RuntimeOrchestrator
from bridge.runtime.touch import TouchDetector, TouchSpeechGate

FIXTURES = Path(__file__).parent / "fixtures" / "touch"
RATE = 16000


class _Actions:
    def __init__(self, sound_s):
        self.calls, self.sound_s = [], sound_s

    def react_touch(self, kind, phase, asleep):
        self.calls.append((kind, phase, asleep))
        return self.sound_s if phase == "start" else 0.0


class _Realtime:
    def __init__(self):
        self.sent = []

    def send_text(self, text, respond=True):
        self.sent.append(text)
        return True


def _run(name, sleeping=False, sound_s=5.3, tail_s=3.0):
    o = object.__new__(RuntimeOrchestrator)
    o._touch = TouchDetector()
    o._touch_gate = TouchSpeechGate()
    o._touch_deaf_until = o._touch_quiet_until = 0.0
    o.robot_actions = _Actions(sound_s)
    o.realtime = None if sleeping else _Realtime()
    o.sleeping = sleeping
    o.playback_started = False
    o._playback_end_ts = 0.0
    o.input_sample_rate = RATE
    o.last_user_activity = 0.0
    o._elapsed_ms = lambda: 0
    audio = np.load(FIXTURES / f"{name}.npy").astype(np.float32) / 32767
    audio = np.concatenate([audio, np.zeros(int(tail_s * RATE), dtype=np.float32)])
    for start in range(0, len(audio), 160):
        o._observe_touch(audio[start : start + 160], (start + 160) / RATE)
    return o


def test_petting_awake_reacts_once_and_asks_for_more():
    o = _run("carinho")
    assert o.robot_actions.calls == [("pet", "start", False), ("pet", "end", False)]
    assert len(o.realtime.sent) == 1 and o.realtime.sent[0].startswith("[toque] carinho de")


def test_tap_awake_startles_and_complains():
    o = _run("batidinha_noagc", sound_s=3.1)
    calls = o.robot_actions.calls
    assert calls and set(calls) == {("tap", "start", False)}  # later series fall in the sound
    assert o.realtime.sent == ["[toque] levou batidinhas na cabeça"]  # then the cooldown


def test_barely_touching_never_startles():
    o = _run("carinho_leve_g2", tail_s=6.0)
    assert "tap" not in [c[0] for c in o.robot_actions.calls]


def test_asleep_petting_only_makes_a_sound():
    o = _run("carinho", sleeping=True)
    assert o.robot_actions.calls == [("pet", "start", True), ("pet", "end", True)]
    assert o.last_user_activity == 0.0


def test_speech_and_silence_do_nothing():
    for name in ("fala", "silencio"):
        o = _run(name)
        assert o.robot_actions.calls == [] and o.realtime.sent == []
