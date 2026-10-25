"""Orchestrator touch wiring: who reacts, who speaks, and when the detector is deaf."""

from types import SimpleNamespace

from bridge.runtime.orchestrator import RuntimeOrchestrator
from bridge.runtime.touch import TouchEvent, TouchKind, TouchSpeechGate


class _Detector:
    def __init__(self, events):
        self.events = events
        self.deaf_calls = []

    def observe(self, sample_rate, sample, now, deaf=False, muted=False):
        self.deaf_calls.append((deaf, muted))
        return self.events if not (deaf or muted) else []


class _Actions:
    def __init__(self):
        self.calls = []

    def react_touch(self, kind, phase, asleep):
        self.calls.append((kind, phase, asleep))
        return 2.0


class _Realtime:
    def __init__(self):
        self.sent = []

    def send_text(self, text, respond=True):
        self.sent.append((text, respond))
        return True


def _orch(events, sleeping=False):
    o = object.__new__(RuntimeOrchestrator)
    o._touch = _Detector(events)
    o._touch_gate = TouchSpeechGate()
    o._touch_deaf_until = 0.0
    o._touch_quiet_until = 0.0
    o.robot_actions = _Actions()
    o.realtime = None if sleeping else _Realtime()
    o.sleeping = sleeping
    o.playback_started = False
    o._playback_end_ts = 0.0
    o.input_sample_rate = 16000
    o.last_user_activity = 0.0
    o._elapsed_ms = lambda: 0
    return o


def test_awake_pet_end_reacts_and_speaks_once():
    o = _orch([TouchEvent(TouchKind.PET_END, 10.0, 6.0)])
    o._observe_touch(SimpleNamespace(), 10.0)
    assert o.robot_actions.calls == [("pet", "end", False)]
    assert o.realtime.sent == [("[toque] carinho de 6 s terminou", True)]
    assert o.last_user_activity == 10.0


def test_asleep_reacts_without_speaking_or_activity():
    o = _orch([TouchEvent(TouchKind.TAP, 5.0)], sleeping=True)
    o._observe_touch(SimpleNamespace(), 5.0)
    assert o.robot_actions.calls == [("tap", "start", True)]
    assert o.last_user_activity == 0.0


def test_deaf_while_playing_and_muted_during_reaction_sound():
    o = _orch([TouchEvent(TouchKind.PET_START, 1.0)])
    o.playback_started = True
    o._observe_touch(SimpleNamespace(), 1.0)
    o.playback_started = False
    o._playback_end_ts = 1.5
    o._observe_touch(SimpleNamespace(), 2.0)  # inside the 0.8 s tail
    o._observe_touch(SimpleNamespace(), 2.5)  # hears PET_START, sound lasts 2.0 s
    o._observe_touch(SimpleNamespace(), 4.0)  # reaction sound still playing
    assert o._touch.deaf_calls == [(True, False), (True, False), (False, False), (False, True)]


def test_petting_quiets_presence_until_after_the_end():
    o = _orch([TouchEvent(TouchKind.PET_START, 1.0)])
    o._observe_touch(SimpleNamespace(), 1.0)
    assert o._touch_quiet_until == float("inf")
    o._touch.events = [TouchEvent(TouchKind.PET_END, 9.0, 8.0)]
    o._touch_deaf_until = 0.0
    o._observe_touch(SimpleNamespace(), 9.0)
    assert o._touch_quiet_until == 9.0 + 15.0


class _Presence:
    def __init__(self):
        self.committed = []

    def observe(self, *args, **kwargs):
        pass

    def pending(self, wall):
        return SimpleNamespace(text="SYSTEM NOTICE (presença): chegou uma pessoa", respond=True)

    def commit(self, notice, wall):
        self.committed.append(notice)


def test_presence_notice_absorbed_while_touch_quiet():
    o = _orch([])
    o._presence = _Presence()
    o._last_presence_poll = 0.0
    o.camera_worker = SimpleNamespace(get_presence=lambda: [], get_pets=lambda: [])
    o._touch_quiet_until = 100.0
    o._maybe_send_presence_notice(50.0)
    assert o.realtime.sent == [] and len(o._presence.committed) == 1
