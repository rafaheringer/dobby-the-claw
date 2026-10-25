import type { RobotTarget } from "./types";

export const ZERO_TARGET: RobotTarget = { x: 0, y: 0, z: 0, roll: 0, pitch: 0, yaw: 0, antennas: [0, 0], yawBody: 0 };

export const LIMITS = {
  x: [-0.02, 0.02],
  y: [-0.02, 0.02],
  z: [-0.04, 0.025],
  roll: [-0.5, 0.5],
  pitch: [-0.6, 0.6],
  yaw: [-1.1, 1.1],
  antenna: [-2.5, 2.5],
  yawBody: [-2.8, 2.8],
} as const;

const clamp = (v: number, [lo, hi]: readonly [number, number]) => Math.min(hi, Math.max(lo, v));

export function clampTarget(t: RobotTarget): RobotTarget {
  return {
    x: clamp(t.x, LIMITS.x),
    y: clamp(t.y, LIMITS.y),
    z: clamp(t.z, LIMITS.z),
    roll: clamp(t.roll, LIMITS.roll),
    pitch: clamp(t.pitch, LIMITS.pitch),
    yaw: clamp(t.yaw, LIMITS.yaw),
    antennas: [clamp(t.antennas[0], LIMITS.antenna), clamp(t.antennas[1], LIMITS.antenna)],
    yawBody: clamp(t.yawBody, LIMITS.yawBody),
  };
}

export function addTargets(a: RobotTarget, b: Partial<RobotTarget>): RobotTarget {
  return {
    x: a.x + (b.x ?? 0),
    y: a.y + (b.y ?? 0),
    z: a.z + (b.z ?? 0),
    roll: a.roll + (b.roll ?? 0),
    pitch: a.pitch + (b.pitch ?? 0),
    yaw: a.yaw + (b.yaw ?? 0),
    antennas: [a.antennas[0] + (b.antennas?.[0] ?? 0), a.antennas[1] + (b.antennas?.[1] ?? 0)],
    yawBody: a.yawBody + (b.yawBody ?? 0),
  };
}

/** Head pose as the Reachy SDK builds it: R = Rz(yaw)·Ry(pitch)·Rx(roll), row-major 4x4. */
export function poseMatrix(t: RobotTarget): number[] {
  const [cr, sr] = [Math.cos(t.roll), Math.sin(t.roll)];
  const [cp, sp] = [Math.cos(t.pitch), Math.sin(t.pitch)];
  const [cy, sy] = [Math.cos(t.yaw), Math.sin(t.yaw)];
  const m = [
    cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, t.x,
    sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, t.y,
    -sp, cp * sr, cp * cr, t.z,
    0, 0, 0, 1,
  ];
  return m.map((v) => v + 0); // no -0
}

/** A slow "alive" motion: breathing and a gentle antenna sway. */
export function idleOffset(tSec: number): Partial<RobotTarget> {
  const breath = Math.sin(2 * Math.PI * 0.25 * tSec);
  const sway = Math.sin(2 * Math.PI * 0.45 * tSec);
  return { z: 0.003 * breath, pitch: 0.03 * breath, antennas: [0.25 * sway, -0.25 * sway] };
}
