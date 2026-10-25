import { AnimatePresence, motion } from "motion/react";
import { useLive } from "../live/store";
import { Pill } from "../ui/Pill";
import { fade } from "./motion";
import { stateLabel, TONE_COLOR } from "./stateLabel";

export function StatePill({ className }: { className?: string }) {
  const status = useLive((s) => s.status);
  const connection = useLive((s) => s.connection);
  const label = stateLabel(status, connection);
  return (
    <Pill dot={TONE_COLOR[label.tone]} pulse={label.pulse} className={className}>
      <AnimatePresence mode="wait" initial={false}>
        <motion.span key={label.text} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -4 }} transition={fade}>
          {label.text}
        </motion.span>
      </AnimatePresence>
    </Pill>
  );
}
