import type { HTMLAttributes, ReactNode } from "react";

const cx = (...parts: (string | false | undefined)[]) => parts.filter(Boolean).join(" ");

/** The basic white surface with a hairline border: every card, bar and panel is one of these. */
export function Card({ className, children, ...rest }: HTMLAttributes<HTMLDivElement> & { children?: ReactNode }) {
  return (
    <div className={cx("surface", className)} {...rest}>
      {children}
    </div>
  );
}

/** The label strip at the top of a card, like the legend on a device panel. */
export function CardHeader({ children, end, className }: { children: ReactNode; end?: ReactNode; className?: string }) {
  return (
    <div className={cx("flex min-h-9 items-center justify-between gap-3 border-b border-line px-3", className)}>
      <h2 className="label truncate">{children}</h2>
      {end}
    </div>
  );
}

/** A titled group of rows inside a card. */
export function Panel({ title, className, children }: { title?: string; className?: string; children?: ReactNode }) {
  return (
    <Card className={className}>
      {title && <CardHeader>{title}</CardHeader>}
      <div className="px-4 py-2">{children}</div>
    </Card>
  );
}

export { cx };
