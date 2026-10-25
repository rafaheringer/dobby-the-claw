import type { ReactNode } from "react";
import { cx } from "./Glass";

/** A small rounded label; with `dot` it shows a colored, optionally pulsing light. */
export function Pill({ dot, pulse, className, children }: { dot?: string; pulse?: boolean; className?: string; children: ReactNode }) {
  return (
    <span className={cx("glass inline-flex h-8 items-center gap-2 rounded-full px-3 text-xs font-medium whitespace-nowrap", className)}>
      {dot && (
        <span className="relative flex size-2">
          {pulse && <span className="absolute inline-flex size-full animate-ping rounded-full opacity-70" style={{ background: dot }} />}
          <span className="relative inline-flex size-2 rounded-full" style={{ background: dot, boxShadow: `0 0 8px ${dot}` }} />
        </span>
      )}
      {children}
    </span>
  );
}
