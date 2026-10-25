import type { LogLine } from "../live/protocol";

export type LogLevel = 10 | 20 | 30 | 40;

export interface LogFilter {
  q: string;
  level: LogLevel;
  cats: string[] | null; // null = every category
}

export const LEVELS: { value: LogLevel; param: string; label: string }[] = [
  { value: 10, param: "debug", label: "Debug" },
  { value: 20, param: "info", label: "Info" },
  { value: 30, param: "aviso", label: "Avisos" },
  { value: 40, param: "erro", label: "Erros" },
];
const DEFAULT_LEVEL: LogLevel = 20;

/** Words must all appear; "!word" hides lines containing it. */
export function matchLog(f: LogFilter): (l: LogLine) => boolean {
  const words = f.q.toLowerCase().split(/\s+/).filter(Boolean);
  const must = words.filter((w) => !w.startsWith("!"));
  const hide = words.filter((w) => w.startsWith("!") && w.length > 1).map((w) => w.slice(1));
  const cats = f.cats && new Set(f.cats);
  return (l) => {
    if (l.lv < f.level || (cats && !cats.has(l.cat))) return false;
    const text = l.msg.toLowerCase();
    return must.every((w) => text.includes(w)) && !hide.some((w) => text.includes(w));
  };
}

export function parseLogParams(sp: URLSearchParams): LogFilter {
  const level = LEVELS.find((l) => l.param === sp.get("nivel"))?.value ?? DEFAULT_LEVEL;
  const cat = sp.get("cat");
  return { q: sp.get("q") ?? "", level, cats: cat === null ? null : cat.split(",").filter(Boolean) };
}

export function toLogParams(f: LogFilter): URLSearchParams {
  const sp = new URLSearchParams();
  if (f.q) sp.set("q", f.q);
  if (f.level !== DEFAULT_LEVEL) sp.set("nivel", LEVELS.find((l) => l.value === f.level)!.param);
  if (f.cats) sp.set("cat", f.cats.join(","));
  return sp;
}
