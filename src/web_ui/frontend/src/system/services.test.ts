import { describe, expect, it } from "vitest";
import type { Status } from "../live/protocol";
import { serviceRows } from "./services";

const base: Status = {
  state: "LISTENING", sleeping: false, transition: "", model: "gemini · live", people: [], speaker: null, camera: true,
  head_tracking: false, video_enabled: true, fps: 8, mic: null, wake: null, services: {}, log_level: "INFO",
};
const row = (s: Status, key: string) => serviceRows(s).find((r) => r.key === key)!;

describe("serviceRows", () => {
  it("shows state and model", () => {
    expect(row(base, "state")).toMatchObject({ label: "Estado", value: "Ouvindo", tone: "good" });
    expect(row({ ...base, sleeping: true }, "state")).toMatchObject({ value: "Dormindo", tone: "off" });
    expect(row(base, "model")).toMatchObject({ label: "Modelo", value: "gemini · live" });
  });

  it("maps Home Assistant and OpenClaw to online / fora do ar / desligado", () => {
    expect(row({ ...base, services: { ha: true } }, "ha")).toMatchObject({ label: "Home Assistant", value: "online", tone: "good" });
    expect(row({ ...base, services: { oc: false } }, "oc")).toMatchObject({ label: "OpenClaw", value: "fora do ar", tone: "bad" });
    expect(row({ ...base, services: { oc: null } }, "oc")).toMatchObject({ value: "fora do ar", tone: "bad" });
    expect(row(base, "ha")).toMatchObject({ value: "desligado", tone: "off" });
  });

  it("formats the Raspberry Pi load and memory", () => {
    const pi = { cpu_load: 0.42, mem_used_gb: 1.25, mem_total_gb: 3.8 };
    expect(row({ ...base, services: { pi } }, "pi")).toMatchObject({ label: "Raspberry Pi", value: "cpu 0.4 · 1.3/3.8 GB", tone: "good" });
    expect(row({ ...base, services: { pi: null } }, "pi")).toMatchObject({ value: "sem resposta", tone: "off" });
    expect(row(base, "pi")).toMatchObject({ value: "sem resposta", tone: "off" });
  });

  it("shows camera fps, good from 5 fps", () => {
    expect(row(base, "camera")).toMatchObject({ label: "Câmera", value: "8 fps", tone: "good" });
    expect(row({ ...base, fps: 3 }, "camera")).toMatchObject({ value: "3 fps", tone: "bad" });
    expect(row({ ...base, fps: 0 }, "camera")).toMatchObject({ value: "parada", tone: "off" });
  });
});
