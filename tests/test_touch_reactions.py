"""ReachyRobotActions.react_touch must never break the mic loop."""

from bridge.runtime.adapters.reachy_actions import ReachyRobotActions


class _NoSdkClient:
    def get_sdk_instance(self):
        raise RuntimeError("no robot")


def test_react_touch_survives_missing_sdk():
    actions = ReachyRobotActions(_NoSdkClient())
    assert actions.react_touch("pet", "start", asleep=True) == 0.0
    assert actions.react_touch("tap", "start", asleep=False) == 0.0
