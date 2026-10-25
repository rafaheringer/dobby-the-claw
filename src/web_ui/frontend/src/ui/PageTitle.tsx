import { ArrowLeft } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router";

/** The page heading over a hairline; `back` adds a small arrow to the parent page, on the same line. */
export function PageTitle({ children, action, back }: { children: ReactNode; action?: ReactNode; back?: { to: string; label: string } }) {
  return (
    <div className="mb-5 flex min-h-12 items-end justify-between gap-3 border-b border-line pb-3 pt-1 lg:pt-0">
      <div className="flex min-w-0 items-center gap-2">
        {back && (
          <Link to={back.to} aria-label={back.label} className="-ml-1 grid size-9 shrink-0 place-items-center text-dim hover:bg-subtle hover:text-text">
            <ArrowLeft className="size-5" />
          </Link>
        )}
        <h1 className="truncate text-2xl font-medium tracking-tight lg:text-[30px] lg:leading-9">{children}</h1>
      </div>
      {action}
    </div>
  );
}
