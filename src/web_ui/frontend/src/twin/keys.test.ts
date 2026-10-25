import { describe, expect, it } from "vitest";
import { keyDelta } from "./keys";

describe("keyDelta", () => {
  it.each([
    ["ArrowLeft", { yaw: 0.05 }],
    ["a", { yaw: 0.05 }],
    ["ArrowRight", { yaw: -0.05 }],
    ["d", { yaw: -0.05 }],
    ["ArrowUp", { pitch: -0.05 }],
    ["w", { pitch: -0.05 }],
    ["ArrowDown", { pitch: 0.05 }],
    ["s", { pitch: 0.05 }],
    ["q", { roll: -0.05 }],
    ["e", { roll: 0.05 }],
    ["r", { z: 0.002 }],
    ["f", { z: -0.002 }],
  ])("%s", (key, delta) => {
    expect(keyDelta(key)).toEqual(delta);
  });

  it("ignores case for letters and unknown keys", () => {
    expect(keyDelta("A")).toEqual({ yaw: 0.05 });
    expect(keyDelta("x")).toBeNull();
    expect(keyDelta("Enter")).toBeNull();
  });
});
