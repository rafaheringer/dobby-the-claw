import { describe, expect, it } from "vitest";
import { streamWidth, videoUrl } from "./video";

describe("streamWidth", () => {
  it("picks one of three fixed widths", () => {
    expect(streamWidth(700)).toBe(640);
    expect(streamWidth(701)).toBe(960);
    expect(streamWidth(1100)).toBe(960);
    expect(streamWidth(1101)).toBe(1280);
  });
});

describe("videoUrl", () => {
  it("encodes overlays as 0/1 and the width", () => {
    expect(videoUrl({ people: true, voice: false, tracking: true, width: 960 })).toBe(
      "/api/video?people=1&voice=0&tracking=1&w=960",
    );
  });
});
