type Delta = Partial<Record<"yaw" | "pitch" | "roll" | "z", number>>;

const STEP = 0.05; // radians
const Z_STEP = 0.002; // metres

const KEYS: Record<string, Delta> = {
  arrowleft: { yaw: STEP },
  a: { yaw: STEP },
  arrowright: { yaw: -STEP },
  d: { yaw: -STEP },
  arrowup: { pitch: -STEP },
  w: { pitch: -STEP },
  arrowdown: { pitch: STEP },
  s: { pitch: STEP },
  q: { roll: -STEP },
  e: { roll: STEP },
  r: { z: Z_STEP },
  f: { z: -Z_STEP },
};

type Mods = Partial<Record<"ctrlKey" | "metaKey" | "altKey" | "shiftKey", boolean>>;

/** How a key nudges the head target on the twin page, or null (other keys and browser shortcuts). */
export function keyDelta(key: string, mods: Mods = {}): Delta | null {
  if (mods.ctrlKey || mods.metaKey || mods.altKey) return null; // Ctrl+R, Ctrl+F… stay the browser's
  return KEYS[key.toLowerCase()] ?? null;
}
