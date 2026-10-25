import type { ReactNode } from "react";
import { cx } from "./Surface";

/** A small boxed tag; with `dot` it shows a colored, optionally pulsing light. */
export function Pill({ dot, pulse, className, children }: { dot?: string; pulse?: boolean; className?: string; children: ReactNode }) {
  return (
    <span
      className={cx(
        "inline-flex h-8 items-center gap-2 border border-ink bg-surface px-2.5 font-mono text-[11px] font-medium uppercase tracking-[0.08em] whitespace-nowrap",
        className,
      )}
    >
      {dot && (
        <span className="relative flex size-2">
          {pulse && <span className="absolute inline-flex size-full animate-ping rounded-full opacity-60" style={{ background: dot }} />}
          <span className="relative inline-flex size-2 rounded-full" style={{ background: dot }} />
        </span>
      )}
      {children}
    </span>
  );
}
