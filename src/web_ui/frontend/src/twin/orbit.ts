import { create } from "zustand";

/**
 * Where the person has turned the full-screen twin's camera, like the official app's orbit view:
 * drag to go around the robot, wheel or pinch to zoom, double-click to go back to the front.
 * Angles are offsets from the default framing; zoom scales its distance.
 */
export interface Orbit {
  az: number; // radians around the robot, positive to its left
  el: number; // radians above the default view
  zoom: number;
}

export const FRONT: Orbit = { az: 0, el: 0, zoom: 1 };
const EL_RANGE = [-0.3, 1.15] as const; // from slightly below to nearly overhead
const ZOOM_RANGE = [0.5, 2] as const;

const clamp = (v: number, [lo, hi]: readonly [number, number]) => Math.min(hi, Math.max(lo, v));

/** A drag of (dx, dy) pixels on a view `height` pixels tall: a full screen height is a full turn. */
export function dragOrbit(o: Orbit, dx: number, dy: number, height: number): Orbit {
  const k = (2 * Math.PI) / Math.max(1, height);
  return { ...o, az: o.az - dx * k, el: clamp(o.el + dy * k, EL_RANGE) };
}

/** `factor` > 1 moves away. */
export function zoomOrbit(o: Orbit, factor: number): Orbit {
  return { ...o, zoom: clamp(o.zoom * factor, ZOOM_RANGE) };
}

/** The camera's offset from the point it looks at, for a framing `distance` seen from `baseEl` above. */
export function orbitOffset(o: Orbit, distance: number, baseEl: number): [number, number, number] {
  const r = distance * o.zoom;
  const el = baseEl + o.el;
  return [r * Math.cos(el) * Math.sin(o.az), r * Math.sin(el), r * Math.cos(el) * Math.cos(o.az)];
}

export const useOrbit = create<Orbit>(() => FRONT);
export const resetOrbit = () => useOrbit.setState(FRONT);
