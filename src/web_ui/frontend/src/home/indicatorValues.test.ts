import { describe, expect, it } from "vitest";
import type { Status } from "../live/protocol";
import { indicatorsFrom } from "./indicatorValues";

const base: Status = {
  state: "IDLE", sleeping: false, transition: "", model: "", people: [], speaker: null, camera: true,
  head_tracking: false, video_enabled: true, fps: 0, mic: { rms: 0.01 }, wake: { heard: null }, services: {}, log_level: "INFO",
};

describe("indicatorsFrom", () => {
  it("counts people and lists names, unknown as ?", () => {
    expect(indicatorsFrom(base).people).toBe("0");
    expect(indicatorsFrom({ ...base, people: [{ id: 1, name: "Ana" }, { id: 2, name: null }] }).people).toBe("2 · Ana, ?");
  });

  it("shows the speaker or a dash", () => {
    expect(indicatorsFrom(base).speaker).toBe("—");
    expect(indicatorsFrom({ ...base, speaker: "Rafael" }).speaker).toBe("Rafael");
  });
});
