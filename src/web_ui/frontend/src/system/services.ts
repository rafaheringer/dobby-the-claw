import { stateLabel } from "../app/stateLabel";
import type { Status } from "../live/protocol";

export type RowTone = "good" | "bad" | "off";
export interface ServiceRow {
  key: string;
  label: string;
  value: string;
  tone: RowTone;
}

function integration(s: Status, key: "ha" | "oc", label: string): ServiceRow {
  if (!(key in s.services)) return { key, label, value: "desligado", tone: "off" };
  return s.services[key] ? { key, label, value: "online", tone: "good" } : { key, label, value: "fora do ar", tone: "bad" };
}

export function serviceRows(s: Status): ServiceRow[] {
  const state = stateLabel(s, "open");
  const pi = s.services.pi;
  return [
    { key: "state", label: "Estado", value: state.text, tone: s.sleeping ? "off" : s.state === "ERROR" ? "bad" : "good" },
    { key: "model", label: "Modelo", value: s.model, tone: "good" },
    integration(s, "ha", "Home Assistant"),
    integration(s, "oc", "OpenClaw"),
    pi
      ? { key: "pi", label: "Raspberry Pi", value: `cpu ${pi.cpu_load.toFixed(1)} · ${pi.mem_used_gb.toFixed(1)}/${pi.mem_total_gb.toFixed(1)} GB`, tone: "good" }
      : { key: "pi", label: "Raspberry Pi", value: "sem resposta", tone: "off" },
    s.fps > 0
      ? { key: "camera", label: "Câmera", value: `${s.fps} fps`, tone: s.fps >= 5 ? "good" : "bad" }
      : { key: "camera", label: "Câmera", value: "parada", tone: "off" },
  ];
}
