"""Gaze target choice and the head-centering controller."""

import math

import numpy as np
import pytest
from reachy_mini.vision.look_at import DEFAULT_HEAD_TO_CAMERA_TRANSFORM
from scipy.spatial.transform import Rotation as R

from reachy.camera_worker import CameraWorker, FaceSighting
from reachy.presence import FOCAL_OVER_WIDTH, Box, GazeSelector, Track


def _track(track_id: int, x: float, height: float, name: str | None = None) -> Track:
    box = Box(x - 0.05, 0.1, x + 0.05, 0.1 + height)
    return Track(id=track_id, box=box, first_seen=0.0, last_seen=0.0, name=name, confirmed_at=0.0)


def test_looks_at_the_nearest_person_and_keeps_them() -> None:
    gaze = GazeSelector()
    near, far = _track(1, 0.3, 0.6), _track(2, 0.7, 0.3)
    assert gaze.choose([near, far], now=100.0).id == 1
    for now in range(101, 160):  # no random glances away
        assert gaze.choose([near, far], now=float(now)).id == 1


def test_switches_only_when_someone_is_clearly_nearer() -> None:
    gaze = GazeSelector()
    a, b = _track(1, 0.3, 0.5), _track(2, 0.7, 0.55)
    assert gaze.choose([a, b], now=100.0).id == 2
    b.box = Box(0.65, 0.1, 0.75, 0.6)
    a.box = Box(0.25, 0.1, 0.35, 0.65)  # a bit taller, not enough to switch
    assert gaze.choose([a, b], now=101.0).id == 2
    a.box = Box(0.25, 0.1, 0.35, 0.8)
    assert gaze.choose([a, b], now=102.0).id == 1


def test_speaker_wins_over_the_nearest() -> None:
    gaze = GazeSelector()
    near, talker = _track(1, 0.3, 0.6), _track(2, 0.7, 0.3, name="Ana")
    assert gaze.choose([near, talker], now=100.0, speech_x=0.72).id == 2
    assert gaze.choose([near, talker], now=101.0, speaker_name="Ana").id == 2


class _NoCamera:
    class media:
        camera = None


def _rotation(pitch: float, yaw: float, roll: float = 0.0) -> np.ndarray:
    return R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


def _project(pitch: float, yaw: float, head: np.ndarray, focal: float, width: int, height: int) -> list[float]:
    """Image point ([-1, 1] coordinates) of the world direction (pitch, yaw) seen by a camera on `head`."""
    direction = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)])
    ray = (head @ DEFAULT_HEAD_TO_CAMERA_TRANSFORM[:3, :3]).T @ direction
    u = width / 2 + focal * ray[0] / ray[2]
    v = height / 2 + focal * ray[1] / ray[2]
    return [2.0 * u / width - 1.0, 2.0 * v / height - 1.0]


@pytest.mark.parametrize("roll", [-0.3, 0.0, 0.3])
def test_direction_ignores_dobby_own_head_roll(roll: float) -> None:
    """A tilted head tilts the image; the face's direction in the world must not change."""
    worker = CameraWorker(_NoCamera())
    width, height = 1280, 720
    head = _rotation(-0.2, 0.3, roll)
    center = _project(-0.35, 0.45, head, FOCAL_OVER_WIDTH * width, width, height)
    pitch, yaw = worker._world_direction(center, head, width, height)
    assert pitch == pytest.approx(-0.35, abs=1e-6)
    assert yaw == pytest.approx(0.45, abs=1e-6)


@pytest.mark.parametrize("delay_frames", [1, 3, 8])
def test_head_converges_on_the_face_center_despite_late_sightings(delay_frames: int) -> None:
    """Sightings come from frames taken `delay_frames` ago: the head still settles on the face, without overshoot."""
    worker = CameraWorker(_NoCamera())
    width, height = 1280, 720
    focal = FOCAL_OVER_WIDTH * width
    face_pitch, face_yaw = 0.2, -0.5
    pitch = yaw = 0.0
    history = [(0.0, 0.0)] * delay_frames
    max_overshoot = 0.0
    for _ in range(40):
        seen_head = history[-delay_frames]  # the head pose when the frame was captured
        center = _project(face_pitch, face_yaw, _rotation(*seen_head), focal, width, height)
        target_pitch, target_yaw = worker._world_direction(center, _rotation(*seen_head), width, height)
        if abs(target_yaw - yaw) >= worker._center_deadband_rad:
            yaw += worker._center_gain * (target_yaw - yaw)
        if abs(target_pitch - pitch) >= worker._center_deadband_rad:
            pitch += worker._center_gain * (target_pitch - pitch)
        history.append((pitch, yaw))
        max_overshoot = max(max_overshoot, face_yaw - yaw)
    assert yaw == pytest.approx(face_yaw, abs=worker._center_deadband_rad)
    assert pitch == pytest.approx(face_pitch, abs=worker._center_deadband_rad)
    assert max_overshoot < 0.01


def test_head_point_of_someone_close_is_at_face_height() -> None:
    """A box cut by the bottom of the image: the head estimate follows the shoulders, not the cut height."""
    box = Box(0.31, 0.58, 0.69, 1.0)  # seated person close to the robot, eyes at y ~0.8
    assert 0.72 <= box.head_point()[1] <= 0.8


def _face(x: float, y: float, area: float) -> FaceSighting:
    return FaceSighting(center=(x, y), box=(0.0, 0.0, 10.0, 10.0), area=area)


def test_follows_the_same_face_even_when_a_bigger_one_appears() -> None:
    worker = CameraWorker(_NoCamera())
    head = np.eye(3)
    first = worker._choose_tracking_target([_face(0.0, 0.0, 0.02)], head, 100.0, 1280, 720)
    assert first is not None
    target = worker._choose_tracking_target([_face(0.05, 0.0, 0.02), _face(-0.7, 0.0, 0.08)], head, 100.1, 1280, 720)
    assert target is not None and target[0][0] == pytest.approx(0.05)


def test_picks_the_face_of_the_chosen_person() -> None:
    worker = CameraWorker(_NoCamera())
    worker._gaze_target = Box(0.6, 0.3, 0.9, 1.0)  # chosen person on the right
    worker._person_update_ts = 100.0
    faces = [_face(-0.5, -0.2, 0.06), _face(0.5, -0.1, 0.02)]
    target = worker._choose_tracking_target(faces, np.eye(3), 100.0, 1280, 720)
    assert target is not None and target[0][0] == pytest.approx(0.5)


def test_falls_back_to_the_head_estimate_when_the_chosen_person_shows_no_face() -> None:
    worker = CameraWorker(_NoCamera())
    worker._gaze_target = Box(0.6, 0.3, 0.9, 1.0)
    worker._person_update_ts = 100.0
    target = worker._choose_tracking_target([_face(-0.5, -0.2, 0.06)], np.eye(3), 100.0, 1280, 720)
    assert target is not None and target[0][0] == pytest.approx(0.5, abs=0.01)
