"""Restoring the XMOS AGC gain after petting saturated the mic array."""

import threading

from reachy.camera_worker import CameraWorker


class _Respeaker:
    def __init__(self):
        self.writes = []

    def write(self, name, values):
        self.writes.append((name, values))


def _worker(respeaker):
    w = object.__new__(CameraWorker)
    w._doa_lock = threading.Lock()
    w._audio_doa = None if respeaker is None else type("Doa", (), {"respeaker": respeaker})()
    return w


def test_restore_mic_gain_writes_agc_gain_off_thread():
    rs = _Respeaker()
    thread = _worker(rs).restore_mic_gain(2.0)
    thread.join(2)
    assert rs.writes == [("PP_AGCGAIN", [2.0])]


def test_restore_mic_gain_opens_the_device_when_doa_is_not_ready(monkeypatch):
    import reachy.camera_worker as cw

    rs = _Respeaker()
    rs.closed = False
    rs.close = lambda: setattr(rs, "closed", True)
    monkeypatch.setattr(cw, "_open_respeaker", lambda: rs)
    _worker(None).restore_mic_gain(2.0).join(2)
    assert rs.writes == [("PP_AGCGAIN", [2.0])] and rs.closed


def test_restore_mic_gain_warns_without_any_device(monkeypatch, caplog):
    import reachy.camera_worker as cw

    monkeypatch.setattr(cw, "_open_respeaker", lambda: None)
    cw.restore_mic_gain(2.0).join(2)
    assert "AGC" in caplog.text and "not restored" in caplog.text


class _GainRespeaker(_Respeaker):
    def __init__(self, gain):
        super().__init__()
        self.gain = gain

    def read(self, name):
        assert name == "PP_AGCGAIN"
        return (self.gain,)


def test_ensure_mic_gain_restores_a_collapsed_agc():
    from reachy.camera_worker import ensure_mic_gain

    rs = _GainRespeaker(0.049)  # left there by an earlier petting nobody detected
    assert ensure_mic_gain(rs, threading.Lock(), 2.0) is True
    assert rs.writes == [("PP_AGCGAIN", [2.0])]


def test_ensure_mic_gain_leaves_a_working_agc_alone():
    from reachy.camera_worker import ensure_mic_gain

    rs = _GainRespeaker(0.8)  # the AGC turning down a loud voice is normal
    assert ensure_mic_gain(rs, threading.Lock(), 2.0) is False
    assert rs.writes == []
