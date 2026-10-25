import type { ServiceRow as Row } from "./services";

const TONE: Record<Row["tone"], string> = { good: "var(--st-listening)", bad: "var(--st-error)", off: "var(--st-sleeping)" };

export function ServiceRow({ row }: { row: Row }) {
  return (
    <div className="flex min-h-11 items-center justify-between gap-4 border-b border-white/6 last:border-0">
      <span className="text-dim">{row.label}</span>
      <span className="flex min-w-0 items-center gap-2 text-right font-medium">
        <span className="size-2 shrink-0 rounded-full" style={{ background: TONE[row.tone] }} />
        <span className="truncate tabular">{row.value}</span>
      </span>
    </div>
  );
}
