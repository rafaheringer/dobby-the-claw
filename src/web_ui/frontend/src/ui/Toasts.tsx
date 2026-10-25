import { AnimatePresence, motion } from "motion/react";
import { useEffect } from "react";
import { spring } from "../app/motion";
import { dismissToast, useLive } from "../live/store";

const TOAST_MS = 3200;

export function Toasts() {
  const toasts = useLive((s) => s.toasts);
  const latest = toasts.at(-1);

  useEffect(() => {
    if (!latest) return;
    const timer = setTimeout(() => dismissToast(latest.id), TOAST_MS);
    return () => clearTimeout(timer);
  }, [latest]);

  return (
    <div role="status" className="pointer-events-none fixed inset-x-0 top-[calc(env(safe-area-inset-top)+12px)] z-50 flex flex-col items-center gap-2 px-4">
      <AnimatePresence>
        {toasts.slice(-3).map((t) => (
          <motion.button
            key={t.id}
            layout
            initial={{ opacity: 0, y: -16, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -12, scale: 0.96 }}
            transition={spring}
            onClick={() => dismissToast(t.id)}
            className="glass pointer-events-auto max-w-sm rounded-2xl bg-[#1d2140]/80 px-4 py-3 text-sm shadow-lg"
          >
            {t.text}
          </motion.button>
        ))}
      </AnimatePresence>
    </div>
  );
}
