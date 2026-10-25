"""Keeping the XMOS mic array at a fixed gain with its AGC off.

The AGC drops the gain within ~100 ms of a hand touching the array (and never recovers), and raises
it up to 64x in a quiet room so a TV or footsteps reach the petting level.
"""

import threading


class _Respeaker:
    def __init__(self, agc_on=True, gain=0.05):
        self.agc_on, self.gain, self.writes, self.closed = agc_on, gain, [], False

    def read(self, name):
        if name == "PP_AGCONOFF":
            return [0, int(self.agc_on), 0, 0, 0]  # status byte, then the int32 bytes
        assert name == "PP_AGCGAIN"
        return (self.gain,)

    def write(self, name, values):
        self.writes.append((name, values))

    def close(self):
        self.closed = True


def test_turns_the_agc_off_and_sets_the_gain():
    from reachy.camera_worker import ensure_fixed_mic_gain

    rs = _Respeaker(agc_on=True, gain=0.05)
    assert ensure_fixed_mic_gain(rs, threading.Lock(), 2.0) is True
    assert rs.writes == [("PP_AGCONOFF", [0]), ("PP_AGCGAIN", [2.0])]


def test_leaves_a_fixed_gain_alone():
    from reachy.camera_worker import ensure_fixed_mic_gain

    rs = _Respeaker(agc_on=False, gain=2.0)
    assert ensure_fixed_mic_gain(rs, threading.Lock(), 2.0) is False
    assert rs.writes == []


def test_apply_opens_and_closes_a_device_handle(monkeypatch):
    import reachy.camera_worker as cw

    rs = _Respeaker()
    monkeypatch.setattr(cw, "_open_respeaker", lambda: rs)
    cw.apply_fixed_mic_gain(2.0).join(2)
    assert ("PP_AGCONOFF", [0]) in rs.writes and rs.closed


def test_apply_warns_without_any_device(monkeypatch, caplog):
    import reachy.camera_worker as cw

    monkeypatch.setattr(cw, "_open_respeaker", lambda: None)
    cw.apply_fixed_mic_gain(2.0).join(2)
    assert "fixed mic gain not applied" in caplog.text.lower()
