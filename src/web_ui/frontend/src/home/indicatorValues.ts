import type { Status } from "../live/protocol";

export function indicatorsFrom(s: Status): { people: string; speaker: string } {
  const names = s.people.map((p) => p.name || "?").join(", ");
  return {
    people: s.people.length ? `${s.people.length} · ${names}` : "0",
    speaker: s.speaker || "—",
  };
}
