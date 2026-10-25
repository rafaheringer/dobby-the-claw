"""ReachyRobotActions.react_touch must never break the mic loop."""

from bridge.runtime.adapters.reachy_actions import ReachyRobotActions


class _NoSdkClient:
    def get_sdk_instance(self):
        raise RuntimeError("no robot")


def test_react_touch_survives_missing_sdk():
    actions = ReachyRobotActions(_NoSdkClient())
    assert actions.react_touch("pet", "start", asleep=True) == 0.0
    assert actions.react_touch("tap", "start", asleep=False) == 0.0


def _write_wav(path, seconds):
    import wave

    import numpy as np

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(int(seconds * 16000), dtype=np.int16).tobytes())


def test_react_touch_does_not_wait_for_play_sound(tmp_path, monkeypatch):
    import threading
    import time

    import bridge.runtime.adapters.reachy_actions as ra

    for name in ("loving1", "surprised1", "relief1", "oops1"):
        _write_wav(tmp_path / f"{name}.wav", 1.5)
    monkeypatch.setattr(ra, "_QUIET_DIR", tmp_path / "quiet")
    release, played = threading.Event(), []

    class _BlockingMedia:
        def play_sound(self, path):
            release.wait(5)
            played.append(path)

    class _Client:
        def get_sdk_instance(self):
            return type("Sdk", (), {"media": _BlockingMedia()})()

    actions = ReachyRobotActions(_Client(), sound_dir=tmp_path)
    t0 = time.monotonic()
    assert actions.react_touch("pet", "start", asleep=True) == 1.5
    assert time.monotonic() - t0 < 0.1
    release.set()
    for _ in range(50):
        if played:
            break
        time.sleep(0.02)
    assert played and played[0].endswith("relief1.wav")
