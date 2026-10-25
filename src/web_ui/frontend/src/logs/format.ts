import type { LogLine } from "../live/protocol";
import { categoryColor } from "./categoryColors";

export const clock = (t: number) => new Date(t * 1000).toLocaleTimeString("pt-BR", { hour12: false });

/** As in the official app: errors red, warnings amber, success green, everything else its category's color. */
export function messageColor(l: LogLine): string {
  if (l.lv >= 40 || /\bERROR\b|FAILED|❌/.test(l.msg)) return "#cc0000";
  if (l.lv >= 30 || /\bWARNING\b/.test(l.msg)) return "#d97706";
  if (l.msg.includes("✓")) return "#00aa00";
  return categoryColor(l.cat);
}

/** What "copy" puts on the clipboard: one `[HH:MM:SS] message` per line. */
export const copyText = (lines: LogLine[]) => lines.map((l) => `[${clock(l.t)}] ${l.msg}`).join("\n");
