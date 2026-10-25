import { create } from "zustand";
import { cx } from "../ui/Glass";

/** Where the home page wants the 3D head drawn; the stage measures this element. */
export const useTwinSlot = create<{ el: HTMLElement | null }>(() => ({ el: null }));

export function TwinSlot({ className, children }: { className?: string; children?: React.ReactNode }) {
  return (
    <div
      data-twin-slot
      ref={(el) => {
        useTwinSlot.setState({ el });
        return () => useTwinSlot.setState((s) => (s.el === el ? { el: null } : s));
      }}
      className={cx("relative", className)}
    >
      {children}
    </div>
  );
}
