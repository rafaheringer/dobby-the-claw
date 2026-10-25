import { describe, expect, it } from "vitest";
import type { LogLine } from "../live/protocol";
import { followKey, keepFollowing } from "./follow";

const lines = (from: number, n: number): LogLine[] =>
  Array.from({ length: n }, (_, i) => ({ id: from + i, t: 0, lv: 20, cat: "Sistema", msg: "" }));

describe("followKey", () => {
  it("changes when new lines arrive even if the capped length stays the same", () => {
    expect(followKey(lines(1, 4000))).not.toBe(followKey(lines(11, 4000)));
  });

  it("is stable when nothing changed", () => {
    expect(followKey(lines(1, 10))).toBe(followKey(lines(1, 10)));
    expect(followKey([])).toBe(followKey([]));
  });
});

describe("keepFollowing", () => {
  it("keeps following when the list itself moves the scroll (new lines, rows measured)", () => {
    expect(keepFollowing(true, { nearBottom: false, byUser: false })).toBe(true);
  });

  it("stops when the person scrolls up to read", () => {
    expect(keepFollowing(true, { nearBottom: false, byUser: true })).toBe(false);
  });

  it("stays put while the person reads, even when the list moves", () => {
    expect(keepFollowing(false, { nearBottom: false, byUser: false })).toBe(false);
  });

  it("follows again once the person scrolls back to the bottom", () => {
    expect(keepFollowing(false, { nearBottom: true, byUser: true })).toBe(true);
  });
});
