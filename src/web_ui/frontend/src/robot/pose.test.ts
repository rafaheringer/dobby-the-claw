import { describe, expect, it } from "vitest";
import { clampTarget, idleOffset, LIMITS, poseMatrix, ZERO_TARGET } from "./pose";

describe("poseMatrix", () => {
  it("is the identity at rest", () => {
    expect(poseMatrix(ZERO_TARGET)).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  });

  it("puts the translation in the last column (row-major)", () => {
    const m = poseMatrix({ ...ZERO_TARGET, x: 0.01, z: -0.02 });
    expect(m[3]).toBe(0.01);
    expect(m[7]).toBe(0);
    expect(m[11]).toBe(-0.02);
  });

  it("rotates about z for yaw", () => {
    const m = poseMatrix({ ...ZERO_TARGET, yaw: Math.PI / 2 });
    expect(m[0]).toBeCloseTo(0);
    expect(m[1]).toBeCloseTo(-1);
    expect(m[4]).toBeCloseTo(1);
    expect(m[5]).toBeCloseTo(0);
  });
});

describe("clampTarget", () => {
  it("keeps every field inside LIMITS", () => {
    const t = clampTarget({ x: 1, y: -1, z: 1, roll: 9, pitch: -9, yaw: 9, antennas: [9, -9], yawBody: -9 });
    expect(t).toEqual({
      x: LIMITS.x[1], y: LIMITS.y[0], z: LIMITS.z[1], roll: LIMITS.roll[1], pitch: LIMITS.pitch[0], yaw: LIMITS.yaw[1],
      antennas: [LIMITS.antenna[1], LIMITS.antenna[0]], yawBody: LIMITS.yawBody[0],
    });
  });
});

describe("idleOffset", () => {
  it("stays small", () => {
    for (let t = 0; t <= 20; t += 0.1) {
      const o = idleOffset(t);
      expect(Math.abs(o.z ?? 0)).toBeLessThanOrEqual(0.003 + 1e-9);
      expect(Math.abs(o.pitch ?? 0)).toBeLessThanOrEqual(0.03 + 1e-9);
      for (const a of o.antennas ?? [0, 0]) expect(Math.abs(a)).toBeLessThanOrEqual(0.25 + 1e-9);
    }
  });
});
