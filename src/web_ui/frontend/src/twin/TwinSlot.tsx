import { AnimatePresence, motion } from "motion/react";
import { Link } from "react-router";
import { create } from "zustand";
import { cx } from "../ui/Glass";
import { HeadIllustration } from "./HeadIllustration";
import { useStageStatus } from "./stageStatus";

/** Where the home page wants the 3D head drawn; the stage measures this element. */
export const useTwinSlot = create<{ el: HTMLElement | null }>(() => ({ el: null }));

// Module-level so the ref callback is stable: a new one each render would re-register the slot
// and restart the stage's animation every status tick.
function registerSlot(el: HTMLDivElement | null) {
  if (!el) return;
  useTwinSlot.setState({ el });
  return () => useTwinSlot.setState((s) => (s.el === el ? { el: null } : s));
}

export function TwinSlot({ className }: { className?: string }) {
  const { ready, failed } = useStageStatus();
  return (
    <div
      data-twin-slot
      ref={registerSlot}
      className={cx("relative", className)}
    >
      {/* soft halo behind the head */}
      <div className="pointer-events-none absolute inset-[12%] rounded-full bg-[radial-gradient(circle,rgba(154,168,255,.28),transparent_68%)]" />
      <div className="pointer-events-none absolute inset-[8%] rounded-full border border-dashed border-accent/25" />
      <AnimatePresence>
        {(!ready || failed) && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 grid place-items-center">
            {failed ? (
              <Link to="/twin" aria-label="Abrir controles do Dobby" className="w-1/2">
                <HeadIllustration className="w-full" />
              </Link>
            ) : (
              <HeadIllustration className="w-1/2 animate-pulse opacity-60" />
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
