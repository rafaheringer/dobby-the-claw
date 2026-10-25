import { describe, expect, it } from "vitest";
import { EMOTIONS, sampleEmotion } from "./emotions";

describe("emotions", () => {
  it("every emotion starts at 0 and ends at its duration", () => {
    for (const e of Object.values(EMOTIONS)) {
      expect(e.keys[0].at).toBe(0);
      expect(e.keys.at(-1)!.at).toBe(e.duration);
    }
  });

  it("samples the first key at 0 and nothing after the end", () => {
    const feliz = EMOTIONS.feliz;
    expect(sampleEmotion("feliz", 0)).toEqual(feliz.keys[0].target);
    expect(sampleEmotion("feliz", feliz.duration + 0.01)).toBeNull();
  });

  it("interpolates between keys", () => {
    const [a, b] = EMOTIONS.curioso.keys;
    const mid = sampleEmotion("curioso", (a.at + b.at) / 2)!;
    const key = Object.keys(b.target).find((k) => typeof (b.target as Record<string, unknown>)[k] === "number")!;
    const va = ((a.target as Record<string, number>)[key] ?? 0) as number;
    const vb = (b.target as Record<string, number>)[key];
    expect((mid as Record<string, number>)[key]).toBeCloseTo((va + vb) / 2);
  });
});
