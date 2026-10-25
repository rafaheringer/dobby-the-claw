import { describe, expect, it } from "vitest";
import type { LogLine } from "../live/protocol";
import { followKey } from "./follow";

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
