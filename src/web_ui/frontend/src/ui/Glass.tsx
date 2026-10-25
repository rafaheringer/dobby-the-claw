import type { HTMLAttributes, ReactNode } from "react";

const cx = (...parts: (string | false | undefined)[]) => parts.filter(Boolean).join(" ");

/** The basic translucent surface: every card, bar and panel is one of these. */
export function Card({ className, children, ...rest }: HTMLAttributes<HTMLDivElement> & { children?: ReactNode }) {
  return (
    <div className={cx("glass rounded-3xl shadow-[0_8px_32px_rgba(0,0,0,.25)]", className)} {...rest}>
      {children}
    </div>
  );
}

/** A titled group of rows inside a card. */
export function Panel({ title, className, children }: { title?: string; className?: string; children?: ReactNode }) {
  return (
    <Card className={cx("p-4", className)}>
      {title && <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-dim">{title}</h2>}
      {children}
    </Card>
  );
}

export { cx };
