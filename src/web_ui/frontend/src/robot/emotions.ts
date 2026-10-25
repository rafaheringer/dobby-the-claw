import type { EmotionName, RobotTarget } from "./types";

type Offset = Partial<RobotTarget>;
interface Emotion {
  duration: number;
  keys: { at: number; target: Offset }[];
}

// Short keyframed moves, added on top of the current pose. Seconds, metres, radians.
export const EMOTIONS: Record<EmotionName | "atento", Emotion> = {
  feliz: {
    duration: 1.6,
    keys: [
      { at: 0, target: {} },
      { at: 0.3, target: { pitch: -0.15, z: 0.01, antennas: [0.9, -0.9] } },
      { at: 0.6, target: { pitch: 0.05, antennas: [-0.4, 0.4] } },
      { at: 0.9, target: { pitch: -0.12, z: 0.01, antennas: [0.9, -0.9] } },
      { at: 1.2, target: { roll: 0.12, antennas: [0.3, -0.3] } },
      { at: 1.6, target: {} },
    ],
  },
  curioso: {
    duration: 1.8,
    keys: [
      { at: 0, target: {} },
      { at: 0.5, target: { roll: 0.3, pitch: -0.1, antennas: [0.6, 0.2] } },
      { at: 1.3, target: { roll: 0.3, pitch: -0.12, yaw: 0.15, antennas: [0.7, 0.3] } },
      { at: 1.8, target: {} },
    ],
  },
  sono: {
    duration: 2.4,
    keys: [
      { at: 0, target: {} },
      { at: 0.9, target: { pitch: 0.4, z: -0.02, antennas: [-1.2, 1.2] } },
      { at: 1.5, target: { pitch: 0.35, z: -0.018, roll: -0.08, antennas: [-1.3, 1.3] } },
      { at: 2.4, target: {} },
    ],
  },
  surpreso: {
    duration: 1.2,
    keys: [
      { at: 0, target: {} },
      { at: 0.15, target: { pitch: -0.25, z: 0.02, antennas: [1.6, -1.6] } },
      { at: 0.7, target: { pitch: -0.2, z: 0.018, antennas: [1.5, -1.5] } },
      { at: 1.2, target: {} },
    ],
  },
  atento: {
    duration: 1.1,
    keys: [
      { at: 0, target: {} },
      { at: 0.3, target: { roll: -0.15, pitch: -0.08, antennas: [0.7, -0.7] } },
      { at: 0.8, target: { roll: -0.12, pitch: -0.06, antennas: [0.6, -0.6] } },
      { at: 1.1, target: {} },
    ],
  },
};

const lerp = (a: number, b: number, u: number) => a + (b - a) * u;

/** The offset of `name` at `tSec` after it started, or null once it is over. */
export function sampleEmotion(name: EmotionName | "atento", tSec: number): Offset | null {
  const { duration, keys } = EMOTIONS[name];
  if (tSec < 0 || tSec > duration) return null;
  const i = Math.max(0, keys.findIndex((_k, j) => j === keys.length - 1 || keys[j + 1].at > tSec));
  const a = keys[i];
  const b = keys[Math.min(i + 1, keys.length - 1)];
  const u = b.at === a.at ? 0 : (tSec - a.at) / (b.at - a.at);
  if (u === 0) return { ...a.target };
  const out: Offset = {};
  for (const f of ["x", "y", "z", "roll", "pitch", "yaw", "yawBody"] as const) {
    if (f in a.target || f in b.target) out[f] = lerp(a.target[f] ?? 0, b.target[f] ?? 0, u);
  }
  if (a.target.antennas || b.target.antennas) {
    const [a0, a1] = a.target.antennas ?? [0, 0];
    const [b0, b1] = b.target.antennas ?? [0, 0];
    out.antennas = [lerp(a0, b0, u), lerp(a1, b1, u)];
  }
  return out;
}
