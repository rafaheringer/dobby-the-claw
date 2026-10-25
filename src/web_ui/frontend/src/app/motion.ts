import { useReducedMotion, type Transition } from "motion/react";

// One spring and one fade for the whole app, so every movement feels the same.
export const spring = { type: "spring", stiffness: 380, damping: 34 } as const satisfies Transition;
export const fade = { duration: 0.15 } as const satisfies Transition;

/** The presets to use right now: with reduced motion the spring becomes a plain fade. */
export function useMotionPreset(): { spring: Transition; fade: Transition; reduced: boolean } {
  const reduced = useReducedMotion() ?? false;
  return { spring: reduced ? fade : spring, fade, reduced };
}
