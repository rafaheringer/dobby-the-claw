import { describe, expect, it } from "vitest";
import { keyboardInset } from "./keyboard";

describe("keyboardInset", () => {
  it("is the covered height when the on-screen keyboard is open", () => {
    expect(keyboardInset(800, 460, 0)).toBe(340);
  });

  it("ignores small gaps from browser bars and zoom, so the box stays above the tab bar", () => {
    expect(keyboardInset(800, 770, 0)).toBe(0);
    expect(keyboardInset(800, 799.5, 0)).toBe(0);
  });

  it("is never negative", () => {
    expect(keyboardInset(800, 820, 0)).toBe(0);
  });
});
