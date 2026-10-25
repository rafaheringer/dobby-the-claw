import { describe, expect, it } from "vitest";
import { fade, spring } from "./motion";

describe("motion presets", () => {
  it("uses one spring for the whole app", () => {
    expect(spring).toEqual({ type: "spring", stiffness: 380, damping: 34 });
  });

  it("uses one short fade", () => {
    expect(fade).toEqual({ duration: 0.15 });
  });
});
