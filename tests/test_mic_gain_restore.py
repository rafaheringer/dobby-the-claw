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


def test_restore_mic_gain_without_mic_array_is_a_no_op():
    assert _worker(None).restore_mic_gain(2.0) is None
