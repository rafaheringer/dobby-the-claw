import type { Status } from "../live/protocol";

const MIC_FULL_RMS = 0.05; // the level that fills the meter

export function indicatorsFrom(s: Status): { people: string; speaker: string; mic: number | null; heard: string | null } {
  const names = s.people.map((p) => p.name || "?").join(", ");
  return {
    people: s.people.length ? `${s.people.length} · ${names}` : "0",
    speaker: s.speaker || "—",
    mic: s.mic ? Math.min(1, s.mic.rms / MIC_FULL_RMS) : null,
    heard: s.sleeping ? (s.wake?.heard ?? null) : null,
  };
}
