import { LIMITS } from "./pose";
import type { RobotTarget } from "./types";

// The camera sits in the head, so pointing it is pointing the head. Up and down is the head's
// pitch. Left and right is the head's yaw (relative to the fixed base); the head turns alone while
// it is within HEAD_ALONE of the body and, past that, drags the body along, so the head never
// strains against its turn range. The SDK's inverse kinematics turns that head pose into the six
// head motors.
//
// The joystick drives it like a security camera: pushing it turns the camera at a speed that
// grows with the push, letting go stops it where it is.

/** How far left or right the camera can look (radians). */
export const CAMERA_YAW_MAX = LIMITS.yaw[1];
/** The head turns alone up to this far from the body; past it the body follows. */
export const HEAD_ALONE = 0.8;
/** Turning speed at full push (radians per second). */
export const PAN_SPEED = 1.2;
export const TILT_SPEED = 0.7;

type Aim = Pick<RobotTarget, "yaw" | "pitch" | "yawBody">;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
/** Gentle near the center for fine aiming, fast at the rim. */
const curve = (v: number) => Math.sign(v) * clamp(v, -1, 1) ** 2;

/**
 * One step of steering: the joystick at (x, y) (right / down positive, -1..1) held for `dt`
 * seconds, from the current aim. Right turns the camera right (negative yaw), down tilts it down.
 */
export function steer(from: Aim, x: number, y: number, dt: number): Aim {
  const yaw = clamp(from.yaw - curve(x) * PAN_SPEED * dt, -CAMERA_YAW_MAX, CAMERA_YAW_MAX);
  const pitch = clamp(from.pitch + curve(y) * TILT_SPEED * dt, LIMITS.pitch[0], LIMITS.pitch[1]);
  const yawBody = clamp(clamp(from.yawBody, yaw - HEAD_ALONE, yaw + HEAD_ALONE), LIMITS.yawBody[0], LIMITS.yawBody[1]);
  return { yaw: yaw + 0, pitch: pitch + 0, yawBody: yawBody + 0 }; // + 0: no -0
}
