import { describe, expect, it } from "vitest";
import { doaRotation, micLevel, pushSample, toggledVolume } from "./audioValues";

describe("audio values", () => {
  it("scales and caps the mic level", () => {
    expect(micLevel(0.025)).toBeCloseTo(0.5);
    expect(micLevel(1)).toBe(1);
  });

  it("points the arrow where the sound comes from", () => {
    expect(doaRotation(0)).toBe(-90); // left
    expect(doaRotation(Math.PI / 2)).toBe(0); // front
    expect(doaRotation(Math.PI)).toBe(90); // right
  });

  it("mutes and restores the last audible level", () => {
    expect(toggledVolume(70, 70)).toBe(0);
    expect(toggledVolume(0, 70)).toBe(70);
    expect(toggledVolume(0, 0)).toBe(50);
  });

  it("keeps a bounded, clamped history", () => {
    expect(pushSample([0.1, 0.2], 2, 2)).toEqual([0.2, 0.95]);
    expect(pushSample([], 0, 3)).toEqual([0.05]);
  });
});
