import { useVirtualizer } from "@tanstack/react-virtual";
import { useLayoutEffect, useRef } from "react";
import type { LogLine } from "../live/protocol";
import { categoryColor } from "./categoryColors";

const clock = (t: number) => new Date(t * 1000).toLocaleTimeString("pt-BR", { hour12: false });
const LEVEL_COLOR: Record<number, string> = { 30: "var(--st-thinking)", 40: "var(--st-error)", 50: "var(--st-error)" };
const NEAR_PX = 40;

/** Only the visible rows are in the DOM, so thousands of lines stay smooth. */
export function LogList({ lines }: { lines: LogLine[] }) {
  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  // eslint-disable-next-line react-hooks/incompatible-library -- the virtualizer is meant to be used this way
  const virtual = useVirtualizer({
    count: lines.length,
    getScrollElement: () => scroller.current,
    estimateSize: () => 22,
    overscan: 20,
    getItemKey: (i) => lines[i].id,
  });

  useLayoutEffect(() => {
    if (stick.current && lines.length) virtual.scrollToIndex(lines.length - 1, { align: "end" });
  }, [lines.length, virtual]);

  return (
    <div
      ref={scroller}
      role="log"
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_PX;
      }}
      className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 py-2 font-mono text-[12px] leading-[18px]"
    >
      {lines.length === 0 && <p className="py-10 text-center font-sans text-sm text-dim">Nenhuma linha com esses filtros.</p>}
      <div className="relative w-full" style={{ height: virtual.getTotalSize() }}>
        {virtual.getVirtualItems().map((row) => {
          const l = lines[row.index];
          return (
            <div
              key={row.key}
              data-index={row.index}
              ref={virtual.measureElement}
              className="absolute left-0 top-0 flex w-full gap-2 py-0.5"
              style={{ transform: `translateY(${row.start}px)` }}
            >
              <span className="shrink-0 text-dim/70 tabular">{clock(l.t)}</span>
              <span className="w-16 shrink-0 truncate sm:w-20" style={{ color: categoryColor(l.cat) }}>
                {l.cat}
              </span>
              <span className="min-w-0 break-words" style={{ color: LEVEL_COLOR[l.lv] }}>
                {l.msg}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
