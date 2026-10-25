import type { RobotState, Status } from "../live/protocol";
import type { LiveState } from "../live/reducer";

export type Tone = "listening" | "thinking" | "executing" | "confirming" | "sleeping" | "error" | "offline";

const STATES: Record<RobotState, { text: string; tone: Tone; pulse: boolean }> = {
  IDLE: { text: "Em espera", tone: "sleeping", pulse: false },
  LISTENING: { text: "Ouvindo", tone: "listening", pulse: true },
  THINKING: { text: "Pensando", tone: "thinking", pulse: true },
  EXECUTING: { text: "Executando", tone: "executing", pulse: true },
  DELEGATING: { text: "Delegando", tone: "executing", pulse: true },
  CONFIRMING: { text: "Confirmando", tone: "confirming", pulse: true },
  ERROR: { text: "Erro", tone: "error", pulse: false },
};

export function stateLabel(s: Status | null, connection: LiveState["connection"]): { text: string; tone: Tone; pulse: boolean } {
  if (connection !== "open" || !s) return { text: "reconectando…", tone: "offline", pulse: false };
  if (s.transition === "sleep") return { text: "Indo dormir…", tone: "sleeping", pulse: true };
  if (s.transition === "wake") return { text: "Acordando…", tone: "thinking", pulse: true };
  if (s.sleeping) return { text: "Dormindo", tone: "sleeping", pulse: false };
  return STATES[s.state] ?? { text: s.state, tone: "sleeping", pulse: false };
}

export const TONE_COLOR: Record<Tone, string> = {
  listening: "var(--st-listening)",
  thinking: "var(--st-thinking)",
  executing: "var(--st-executing)",
  confirming: "var(--st-confirming)",
  sleeping: "var(--st-sleeping)",
  error: "var(--st-error)",
  offline: "var(--st-error)",
};
