import { describe, expect, it } from "vitest";
import type { RobotState, Status } from "../live/protocol";
import { stateLabel } from "./stateLabel";

const base: Status = {
  state: "IDLE", sleeping: false, transition: "", model: "", people: [], speaker: null, camera: true,
  head_tracking: false, video_enabled: true, fps: 0, mic: null, wake: null, services: {}, log_level: "INFO",
};

describe("stateLabel", () => {
  it.each<[RobotState, string, string]>([
    ["IDLE", "Em espera", "sleeping"],
    ["LISTENING", "Ouvindo", "listening"],
    ["THINKING", "Pensando", "thinking"],
    ["EXECUTING", "Executando", "executing"],
    ["DELEGATING", "Delegando", "executing"],
    ["CONFIRMING", "Confirmando", "confirming"],
    ["ERROR", "Erro", "error"],
  ])("%s → %s", (state, text, tone) => {
    expect(stateLabel({ ...base, state }, "open")).toMatchObject({ text, tone });
  });

  it("sleeping wins over the state", () => {
    expect(stateLabel({ ...base, state: "LISTENING", sleeping: true }, "open")).toMatchObject({ text: "Dormindo", tone: "sleeping" });
  });

  it("a transition wins over sleeping and pulses", () => {
    expect(stateLabel({ ...base, sleeping: true, transition: "wake" }, "open")).toMatchObject({ text: "Acordando…", pulse: true });
    expect(stateLabel({ ...base, transition: "sleep" }, "open")).toMatchObject({ text: "Indo dormir…", pulse: true });
  });

  it("offline wins over everything", () => {
    expect(stateLabel({ ...base, transition: "wake" }, "closed")).toMatchObject({ text: "reconectando…", tone: "offline" });
    expect(stateLabel(null, "connecting")).toMatchObject({ text: "reconectando…", tone: "offline" });
  });

  it("listening pulses, idle does not", () => {
    expect(stateLabel({ ...base, state: "LISTENING" }, "open").pulse).toBe(true);
    expect(stateLabel(base, "open").pulse).toBe(false);
  });
});
