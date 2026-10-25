"""The web UI's robot control: manual mode ownership, safety releases, emotions and the pose stream."""

from reachy_mini.utils import create_head_pose

from reachy.manual import ManualMove
from web_ui.robot_control import EMOTIONS, IDLE_RELEASE_S, RobotControl


class FakeMotion:
    def __init__(self):
        self.queue = []
        self.cleared = 0
        self.pose = (create_head_pose(yaw=0.3, degrees=False, mm=False), (0.5, -0.5), 0.2)

    def commanded_pose(self):
        return self.pose

    def queue_move(self, move):
        self.queue.append(move)

    def clear_move_queue(self):
        self.queue.clear()
        self.cleared += 1

    def mark_activity(self):
        pass


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def make(asleep=False, tools=None):
    motion, clock, state = FakeMotion(), Clock(), {"asleep": asleep}
    calls = tools if tools is not None else []
    control = RobotControl(
        motion, lambda name, args: calls.append((name, args)), lambda: state["asleep"], clock=clock
    )
    return control, motion, clock, state, calls


def test_entering_manual_holds_the_current_pose():
    control, motion, *_ = make()
    assert control.set_manual(True, owner="a") is None
    assert len(motion.queue) == 1 and isinstance(motion.queue[0], ManualMove)
    assert abs(motion.queue[0].target().yaw - 0.3) < 1e-6
    assert control.snapshot()["mode"] == "manual"


def test_a_target_enters_manual_and_moves_the_hold():
    control, motion, *_ = make()
    assert control.set_target({"yaw": 0.5}, owner="a") is None
    move = motion.queue[0]
    assert (
        abs(move.target().yaw - 0.5) < 1e-9 and abs(move.target().antennas[0] - 0.5) < 1e-9
    )  # other fields kept


def test_back_to_auto_lets_the_robot_move_on_its_own():
    control, motion, *_ = make()
    control.set_manual(True, owner="a")
    control.set_manual(False, owner="a")
    assert motion.queue == [] and control.snapshot()["mode"] == "auto"


def test_bad_target_is_refused_without_moving():
    control, motion, *_ = make()
    assert control.set_target({"yaw": "x"}, owner="a")
    assert motion.queue == []


def test_no_manual_control_while_asleep():
    control, motion, *_ = make(asleep=True)
    assert control.set_manual(True, owner="a")
    assert control.set_target({"yaw": 0.1}, owner="a")
    assert motion.queue == []


def test_falling_asleep_releases_manual():
    control, motion, _, state, _ = make()
    control.set_manual(True, owner="a")
    state["asleep"] = True
    control.tick()
    assert motion.queue == [] and control.snapshot()["mode"] == "auto"


def test_the_controlling_page_leaving_releases_manual():
    control, _motion, *_ = make()
    control.set_manual(True, owner="a")
    control.release("b")
    assert control.snapshot()["mode"] == "manual"
    control.release("a")
    assert control.snapshot()["mode"] == "auto"


def test_forgotten_manual_mode_times_out():
    control, _motion, clock, *_ = make()
    control.set_target({"yaw": 0.1}, owner="a")
    clock.t += IDLE_RELEASE_S - 1
    control.tick()
    assert control.snapshot()["mode"] == "manual"
    clock.t += 2
    control.tick()
    assert control.snapshot()["mode"] == "auto"


def test_emotion_plays_the_recorded_move_and_keeps_manual_afterwards():
    control, motion, _, _, calls = make()
    control.set_manual(True, owner="a")
    assert control.play_emotion("feliz", owner="a") is None
    assert calls == [("express_emotion", {"emotion": EMOTIONS["feliz"], "sound": True})]
    assert isinstance(motion.queue[-1], ManualMove)  # the hold resumes after the emotion


def test_unknown_emotion_is_refused():
    control, _, _, _, calls = make()
    assert control.play_emotion("dança-proibida", owner="a")
    assert calls == []


def test_snapshot_reports_the_pose_the_robot_is_commanded_to():
    control, *_ = make()
    snap = control.snapshot()
    assert snap["type"] == "robot" and abs(snap["pose"]["yaw"] - 0.3) < 1e-3
    assert snap["pose"]["antennas"] == [0.5, -0.5] and snap["pose"]["yawBody"] == 0.2


def test_without_a_motion_manager_nothing_is_available():
    control = RobotControl(None, None, lambda: False)
    assert control.snapshot() is None
    assert control.set_manual(True, owner="a")


def test_the_head_follows_the_body_past_its_turn_range():
    from reachy.manual import HEAD_TURN_MAX, ManualTarget

    t = ManualTarget.from_message({"yaw": 0.0, "yawBody": 2.0})
    assert abs(t.yaw - (2.0 - HEAD_TURN_MAX)) < 1e-9 and t.body_yaw == 2.0
    # Looking far to the side is fine as long as the body turns too.
    t = ManualTarget.from_message({"yaw": 2.2, "yawBody": 1.4})
    assert abs(t.yaw - 2.2) < 1e-9
