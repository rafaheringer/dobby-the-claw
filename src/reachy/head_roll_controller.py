from __future__ import annotations

import time

import numpy as np


class HeadRollController:
    """Simple, robust tilt->roll controller with strong neutral lock."""

    def __init__(self) -> None:
        self._gain = 1.25
        self._max_roll_rad = np.deg2rad(18.0)
        self._smoothing_alpha = 0.20
        self._max_speed_rad_s = np.deg2rad(45.0)

        self._hard_deadband_rad = np.deg2rad(8.0)
        self._follow_enter_rad = np.deg2rad(12.0)
        self._follow_exit_rad = np.deg2rad(6.0)
        self._enter_required_frames = 3
        self._exit_required_frames = 6

        self._bias_alpha_neutral = 0.10
        self._bias_alpha_follow = 0.015

        self._neutral_locked = True
        self._smoothed_roll_rad = 0.0
        self._bias_rad = 0.0
        self._last_filtered_tilt_rad = 0.0

        self._enter_counter = 0
        self._exit_counter = 0

        self._last_update_ts = time.time()

    def reset(self) -> None:
        self._neutral_locked = True
        self._smoothed_roll_rad = 0.0
        self._bias_rad = 0.0
        self._last_filtered_tilt_rad = 0.0
        self._enter_counter = 0
        self._exit_counter = 0
        self._last_update_ts = time.time()

    @property
    def last_filtered_tilt_rad(self) -> float:
        return float(self._last_filtered_tilt_rad)

    @property
    def bias_rad(self) -> float:
        return float(self._bias_rad)

    @property
    def is_neutral_locked(self) -> bool:
        return bool(self._neutral_locked)

    def update(self, raw_tilt_rad: float, now: float | None = None) -> float:
        ts = time.time() if now is None else float(now)
        dt = max(1e-3, ts - self._last_update_ts)
        self._last_update_ts = ts

        raw_tilt = float(np.clip(raw_tilt_rad, -np.deg2rad(35.0), np.deg2rad(35.0)))

        if self._neutral_locked:
            self._bias_rad += self._bias_alpha_neutral * (raw_tilt - self._bias_rad)
        else:
            self._bias_rad += self._bias_alpha_follow * (raw_tilt - self._bias_rad)

        centered_tilt = raw_tilt - self._bias_rad
        filtered_tilt = centered_tilt
        self._last_filtered_tilt_rad = filtered_tilt
        abs_tilt = abs(filtered_tilt)

        if self._neutral_locked:
            if abs_tilt >= self._follow_enter_rad:
                self._enter_counter += 1
                if self._enter_counter >= self._enter_required_frames:
                    self._neutral_locked = False
                    self._enter_counter = 0
                    self._exit_counter = 0
            else:
                self._enter_counter = 0

            target_roll = 0.0
        else:
            if abs_tilt <= self._follow_exit_rad:
                self._exit_counter += 1
                if self._exit_counter >= self._exit_required_frames or abs_tilt <= self._hard_deadband_rad:
                    self._neutral_locked = True
                    self._exit_counter = 0
                    target_roll = 0.0
                else:
                    target_roll = float(np.clip(filtered_tilt * self._gain, -self._max_roll_rad, self._max_roll_rad))
            else:
                self._exit_counter = 0
                target_roll = float(np.clip(filtered_tilt * self._gain, -self._max_roll_rad, self._max_roll_rad))

        filtered_roll = ((1.0 - self._smoothing_alpha) * self._smoothed_roll_rad) + (self._smoothing_alpha * target_roll)
        max_step = self._max_speed_rad_s * dt
        delta = filtered_roll - self._smoothed_roll_rad
        delta = max(-max_step, min(max_step, delta))
        self._smoothed_roll_rad += delta

        if self._neutral_locked and abs(self._smoothed_roll_rad) < np.deg2rad(0.30):
            self._smoothed_roll_rad = 0.0

        return float(self._smoothed_roll_rad)
