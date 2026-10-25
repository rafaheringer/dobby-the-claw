import { describe, expect, it } from "vitest";
import { fitDistance, stageModeFor } from "./framing";

describe("fitDistance", () => {
  it("backs off until the box height fits the vertical field of view", () => {
    // 90° fov: half-height 1 needs distance 1, times the margin
    expect(fitDistance({ width: 0.1, height: 2 }, 90, 1, 1)).toBeCloseTo(1);
    expect(fitDistance({ width: 0.1, height: 2 }, 90, 1, 1.5)).toBeCloseTo(1.5);
  });

  it("uses the width when the view is narrow", () => {
    // aspect 0.5: horizontal half-fov tan = 0.5, half-width 1 → distance 2
    expect(fitDistance({ width: 2, height: 0.1 }, 90, 0.5, 1)).toBeCloseTo(2);
  });
});

describe("stageModeFor", () => {
  it("draws the head on home, the whole robot on /twin, nothing elsewhere", () => {
    expect(stageModeFor("/")).toBe("head");
    expect(stageModeFor("/twin")).toBe("full");
    expect(stageModeFor("/logs")).toBe("hidden");
    expect(stageModeFor("/sistema/nome")).toBe("hidden");
  });
});
