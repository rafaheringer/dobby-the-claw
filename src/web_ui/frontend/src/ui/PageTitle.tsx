import { ChevronLeft } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router";

/** The page heading; `back` adds a small arrow to the parent page, on the same line. */
export function PageTitle({ children, action, back }: { children: ReactNode; action?: ReactNode; back?: { to: string; label: string } }) {
  return (
    <div className="mb-4 flex min-h-10 items-center justify-between gap-3 pt-1 lg:pt-0">
      <div className="flex min-w-0 items-center gap-1">
        {back && (
          <Link to={back.to} aria-label={back.label} className="-ml-2 grid size-9 shrink-0 place-items-center rounded-full text-dim hover:bg-white/8 hover:text-text">
            <ChevronLeft className="size-6" />
          </Link>
        )}
        <h1 className="truncate text-2xl font-semibold tracking-tight lg:text-3xl">{children}</h1>
      </div>
      {action}
    </div>
  );
}
