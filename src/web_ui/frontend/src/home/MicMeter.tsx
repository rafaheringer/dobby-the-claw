import { motion } from "motion/react";
import { spring } from "../app/motion";

/** A short level bar: green when the mic hears something, amber when it is nearly silent. */
export function MicMeter({ level }: { level: number | null }) {
  const value = level ?? 0;
  const color = value > 0.04 ? "var(--st-listening)" : "var(--st-thinking)";
  return (
    <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/10">
      <motion.div className="h-full rounded-full" style={{ background: color }} animate={{ width: `${value * 100}%` }} transition={spring} />
    </div>
  );
}
