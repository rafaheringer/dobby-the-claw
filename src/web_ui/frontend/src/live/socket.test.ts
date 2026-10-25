import { describe, expect, it } from "vitest";
import { backoffMs } from "./socket";

describe("backoffMs", () => {
  it("doubles from 800 ms and stops at 10 s", () => {
    expect(backoffMs(0)).toBe(800);
    expect(backoffMs(2)).toBe(3200);
    expect(backoffMs(10)).toBe(10000);
  });
});
