"""TouchDetector against the probe recordings (16 kHz mono, 8 s each)."""

from pathlib import Path

import numpy as np

from bridge.runtime.touch import TouchDetector, TouchKind

FIXTURES = Path(__file__).parent / "fixtures" / "touch"
RATE = 16000


def run(name, chunk=160, tail_s=3.0, deaf_from=None, stereo=False):
    audio = np.load(FIXTURES / f"{name}.npy").astype(np.float32) / 32767
    audio = np.concatenate([audio, np.zeros(int(tail_s * RATE), dtype=np.float32)])
    detector = TouchDetector()
    events = []
    for start in range(0, len(audio), chunk):
        piece = audio[start : start + chunk]
        if stereo:
            piece = np.stack([piece, piece], axis=1)
        now = (start + len(audio[start : start + chunk])) / RATE
        deaf = deaf_from is not None and now >= deaf_from
        events.extend(detector.observe(RATE, piece, now, deaf=deaf))
    return events


def test_silence_and_speech_produce_nothing():
    assert run("silencio") == []
    assert run("fala") == []


def test_petting_starts_fast_and_ends_once():
    ev = run("carinho")
    assert [e.kind for e in ev] == [TouchKind.PET_START, TouchKind.PET_END]
    assert ev[0].at <= 1.6
    assert ev[1].duration_s > 5.0


def test_light_petting_detected_without_taps():
    ev = run("carinho_leve")
    assert ev[0].kind is TouchKind.PET_START and ev[0].at <= 2.5
    assert TouchKind.TAP not in [e.kind for e in ev]


def test_tap_emitted_once_after_hold():
    ev = run("batidinha")
    assert [e.kind for e in ev] == [TouchKind.TAP]
    assert 3.4 <= ev[0].at <= 3.7


def test_deaf_suppresses_everything():
    assert run("carinho", deaf_from=0.0) == []


def test_deaf_mid_episode_ends_it_once():
    kinds = [e.kind for e in run("carinho", deaf_from=3.0)]
    assert kinds == [TouchKind.PET_START, TouchKind.PET_END]


def test_chunking_and_stereo_do_not_change_events():
    base = [(e.kind, round(e.at, 1)) for e in run("carinho")]
    assert [(e.kind, round(e.at, 1)) for e in run("carinho", chunk=437)] == base
    assert [(e.kind, round(e.at, 1)) for e in run("carinho", stereo=True)] == base


def test_muted_after_start_keeps_one_episode():
    audio = np.load(FIXTURES / "carinho.npy").astype(np.float32) / 32767
    audio = np.concatenate([audio, np.zeros(3 * RATE, dtype=np.float32)])
    detector = TouchDetector()
    events, muted_until = [], 0.0
    for start in range(0, len(audio), 160):
        now = (start + 160) / RATE
        new = detector.observe(RATE, audio[start : start + 160], now, muted=now < muted_until)
        if any(e.kind is TouchKind.PET_START for e in new):
            muted_until = now + 5.6  # reaction sound playing
        events.extend(new)
    assert [e.kind for e in events] == [TouchKind.PET_START, TouchKind.PET_END]
    assert events[1].duration_s > 5.0 and not events[1].interrupted


def test_deaf_mid_episode_marks_pet_end_interrupted():
    ev = run("carinho", deaf_from=3.0)
    assert ev[1].interrupted
