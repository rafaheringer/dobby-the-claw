"""TouchSpeechGate: when a touch earns a spoken reaction, and its notice text."""

from bridge.runtime.touch import TouchEvent, TouchKind, TouchSpeechGate


def test_first_pet_end_speaks_then_cooldown():
    g = TouchSpeechGate(cooldown_s=300)
    assert g.notice(TouchEvent(TouchKind.PET_END, 10, 6.2), 10) == "[toque] carinho de 6 s terminou"
    assert g.notice(TouchEvent(TouchKind.PET_END, 100, 3.0), 100) is None
    assert g.notice(TouchEvent(TouchKind.PET_END, 311, 3.0), 311) is not None


def test_kinds_have_independent_cooldowns():
    g = TouchSpeechGate()
    g.notice(TouchEvent(TouchKind.PET_END, 0, 4), 0)
    assert g.notice(TouchEvent(TouchKind.TAP, 1), 1) == "[toque] levou batidinhas na cabeça"


def test_pet_start_never_speaks():
    assert TouchSpeechGate().notice(TouchEvent(TouchKind.PET_START, 0), 0) is None
