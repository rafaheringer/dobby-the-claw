import { describe, expect, it } from "vitest";
import { dragOrbit, FRONT, orbitOffset, zoomOrbit } from "./orbit";

describe("orbit", () => {
  it("looks from the front by default", () => {
    const [x, y, z] = orbitOffset(FRONT, 1, 0);
    expect(x).toBeCloseTo(0);
    expect(y).toBeCloseTo(0);
    expect(z).toBeCloseTo(1);
  });

  it("goes around the robot when dragged sideways", () => {
    const half = dragOrbit(FRONT, -400, 0, 800); // half a screen height: half a turn
    const [x, , z] = orbitOffset(half, 1, 0);
    expect(x).toBeCloseTo(0);
    expect(z).toBeCloseTo(-1); // behind it
  });

  it("never flips over the top or goes far under", () => {
    expect(dragOrbit(FRONT, 0, 10000, 800).el).toBeLessThan(Math.PI / 2);
    expect(dragOrbit(FRONT, 0, -10000, 800).el).toBeGreaterThan(-0.5);
  });

  it("keeps zoom in range and scales the distance", () => {
    expect(zoomOrbit(FRONT, 100).zoom).toBe(2);
    expect(zoomOrbit(FRONT, 0.01).zoom).toBe(0.5);
    expect(orbitOffset(zoomOrbit(FRONT, 1.5), 1, 0)[2]).toBeCloseTo(1.5);
  });
});
