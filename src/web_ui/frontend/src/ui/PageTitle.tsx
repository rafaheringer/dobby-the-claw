import type { ReactNode } from "react";

export function PageTitle({ children, action }: { children: ReactNode; action?: ReactNode }) {
  return (
    <div className="mb-4 flex items-center justify-between gap-3 pt-1 lg:pt-0">
      <h1 className="text-2xl font-semibold tracking-tight lg:text-3xl">{children}</h1>
      {action}
    </div>
  );
}
