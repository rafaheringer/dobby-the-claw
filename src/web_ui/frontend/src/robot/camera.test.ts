import { describe, expect, it } from "vitest";
import { CAMERA_YAW_MAX, HEAD_ALONE, PAN_SPEED, steer, TILT_SPEED } from "./camera";
import { clampTarget, LIMITS, ZERO_TARGET } from "./pose";

const ahead = { yaw: 0, pitch: 0, yawBody: 0 };

describe("camera joystick", () => {
  it("stays put with the stick centered", () => {
    expect(steer({ yaw: 0.3, pitch: -0.1, yawBody: 0.1 }, 0, 0, 1)).toEqual({ yaw: 0.3, pitch: -0.1, yawBody: 0.1 });
  });

  it("turns left while held left, at a speed that grows with the push", () => {
    expect(steer(ahead, -1, 0, 0.5).yaw).toBeCloseTo(PAN_SPEED * 0.5);
    expect(steer(ahead, -0.5, 0, 0.5).yaw).toBeCloseTo(PAN_SPEED * 0.5 * 0.25);
    expect(steer(ahead, 1, 0, 0.5).yaw).toBeCloseTo(-PAN_SPEED * 0.5); // right is negative yaw, like the twin
  });

  it("tilts down while held down", () => {
    expect(steer(ahead, 0, 1, 0.5).pitch).toBeCloseTo(TILT_SPEED * 0.5);
    expect(steer(ahead, 0, -1, 100).pitch).toBe(LIMITS.pitch[0]);
  });

  it("moves only the head near the middle and drags the body along past it", () => {
    let aim = ahead;
    for (let i = 0; i < 100; i++) aim = steer(aim, -1, 0, 0.05);
    expect(aim.yaw).toBe(CAMERA_YAW_MAX);
    expect(aim.yawBody).toBeCloseTo(CAMERA_YAW_MAX - HEAD_ALONE);
    expect(clampTarget({ ...ZERO_TARGET, ...aim })).toMatchObject(aim); // the bridge accepts it as is
    expect(steer(ahead, -1, 0, 0.3).yawBody).toBe(0); // a small turn leaves the body alone
  });

  it("leaves a turned body where it is until the head needs it", () => {
    const aim = steer({ yaw: 1, pitch: 0, yawBody: 0.6 }, 1, 0, 0.1);
    expect(aim.yawBody).toBe(0.6);
  });
});
