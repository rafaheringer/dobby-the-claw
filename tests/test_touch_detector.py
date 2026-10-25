"""TouchDetector against the probe recordings (16 kHz mono, 8 s each)."""

from pathlib import Path

import numpy as np

from bridge.runtime.touch import SpeechVad, TouchDetector, TouchKind

FIXTURES = Path(__file__).parent / "fixtures" / "touch"
RATE = 16000


def run(name, chunk=160, tail_s=3.0, deaf_from=None, stereo=False, vad=None):
    audio = np.load(FIXTURES / f"{name}.npy").astype(np.float32) / 32767
    audio = np.concatenate([audio, np.zeros(int(tail_s * RATE), dtype=np.float32)])
    detector = TouchDetector(vad=vad)
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
    ev = run("carinho_leve_g2")
    # Late: its first strokes look like a knock series, which holds petting back.
    assert ev[0].kind is TouchKind.PET_START and ev[0].at <= 7.5
    assert TouchKind.TAP not in [e.kind for e in ev]


def test_tap_emitted_once_per_series_after_hold():
    # Four series of 4-8 knocks; each is reported 1 s after its last knock.
    ev = run("batidinha_g2")
    assert [e.kind for e in ev] == [TouchKind.TAP] * 4
    assert 2.5 <= ev[0].at <= 2.9


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


def _kinds(events):
    return [e.kind for e in events]


def test_rapid_taps_at_full_gain_are_taps_not_petting():
    # Bursts of ~5 taps/s: each tap is one or two loud windows with silence in between.
    kinds = _kinds(run("batidinha_g2"))
    assert TouchKind.PET_START not in kinds
    assert TouchKind.TAP in kinds


def test_petting_at_full_gain_is_one_episode_without_taps():
    for name in ("carinho_g2", "carinho_leve_g2"):
        kinds = _kinds(run(name))
        assert TouchKind.TAP not in kinds, name
        assert kinds[0] is TouchKind.PET_START, name
        assert kinds.count(TouchKind.PET_START) == kinds.count(TouchKind.PET_END) >= 1, name


def test_close_speech_is_not_a_touch():
    # Talking 30-50 cm from the head reaches the petting level in sparse windows.
    assert run("fala_perto_g2", vad=SpeechVad()) == []


def test_speech_filter_keeps_touches():
    assert TouchKind.PET_START not in _kinds(run("batidinha_g2", vad=SpeechVad()))
    for name in ("carinho", "carinho_g2", "carinho_leve_g2"):
        kinds = _kinds(run(name, vad=SpeechVad()))
        assert kinds[0] is TouchKind.PET_START and TouchKind.TAP not in kinds, name


def test_speech_vad_scores_voice_high_and_touch_low():
    def score(name, rate=RATE):
        audio = np.load(FIXTURES / f"{name}.npy").astype(np.float32) / 32767
        if rate != RATE:
            audio = np.repeat(audio, rate // RATE)
        vad = SpeechVad()
        return max(vad(audio[i : i + 960], rate) for i in range(0, len(audio), 960))

    assert score("fala_perto_g2") > 0.9
    assert score("fala_perto_g2", rate=48000) > 0.9
    assert score("batidinha_g2") < 0.5


def _knocks(times, length_s=6.0):
    """Silence with a 20 ms full-scale knock at each time (s)."""
    audio = np.zeros(int(length_s * RATE), dtype=np.float32)
    for t in times:
        start = int(t * RATE)
        audio[start : start + RATE // 50] = 0.9 * np.sign(np.sin(np.arange(RATE // 50)))
    return audio


def _feed(audio):
    detector, events = TouchDetector(), []
    for start in range(0, len(audio), 160):
        events.extend(detector.observe(RATE, audio[start : start + 160], (start + 160) / RATE))
    return _kinds(events)


def test_three_regular_knocks_are_taps():
    assert _feed(_knocks([1.0, 1.25, 1.5])) == [TouchKind.TAP]


def test_one_or_two_knocks_are_not_taps():
    # A notification chime or a single bump on the table.
    assert _feed(_knocks([1.0])) == []
    assert _feed(_knocks([1.0, 1.3])) == []


def test_irregular_knocks_are_not_taps():
    # Typing next to the robot: uneven gaps.
    assert _feed(_knocks([1.0, 1.2, 1.75, 1.95, 2.5])) == []
