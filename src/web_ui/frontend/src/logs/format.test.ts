import { describe, expect, it } from "vitest";
import type { LogLine } from "../live/protocol";
import { CATEGORY_COLORS } from "./categoryColors";
import { copyText, messageColor } from "./format";

const line = (lv: number, msg: string, cat = "Conversa"): LogLine => ({ id: 1, t: 0, lv, cat, msg });

describe("log format", () => {
  it("colors by severity first, then by category", () => {
    expect(messageColor(line(40, "boom"))).toBe("#cc0000");
    expect(messageColor(line(20, "step FAILED"))).toBe("#cc0000");
    expect(messageColor(line(30, "hmm"))).toBe("#d97706");
    expect(messageColor(line(20, "pronto ✓"))).toBe("#00aa00");
    expect(messageColor(line(20, "olá", "Visão"))).toBe(CATEGORY_COLORS["Visão"]);
  });

  it("copies time-stamped lines", () => {
    expect(copyText([line(20, "a"), line(20, "b")]).split("\n")).toHaveLength(2);
    expect(copyText([line(20, "a")])).toMatch(/^\[\d\d:\d\d:\d\d\] a$/);
  });
});
