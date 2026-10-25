import { AnimatePresence, motion } from "motion/react";
import type { ReactNode } from "react";
import { spring } from "../app/motion";
import { useLive } from "../live/store";
import { Card } from "../ui/Surface";
import { indicatorsFrom } from "./indicatorValues";
import { MicMeter } from "./MicMeter";

/** A value that slides in when it changes. */
function Value({ children }: { children: string }) {
  return (
    <span className="relative block h-7 overflow-hidden">
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.span
          key={children}
          initial={{ y: 14, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: -14, opacity: 0 }}
          transition={spring}
          className="block truncate font-mono text-lg font-medium leading-7"
        >
          {children}
        </motion.span>
      </AnimatePresence>
    </span>
  );
}

function Cell({ label, value, extra }: { label: string; value: string; extra?: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-1.5 border-line px-4 py-3 [&:not(:last-child)]:border-b sm:[&:not(:last-child)]:border-b-0 sm:[&:not(:last-child)]:border-r">
      <span className="label">{label}</span>
      <Value>{value}</Value>
      {extra}
    </div>
  );
}

/** One instrument strip: each reading in its own cell, split by hairlines. */
export function Indicators() {
  const status = useLive((s) => s.status);
  if (!status) return null;
  const v = indicatorsFrom(status);
  return (
    <Card className="grid sm:grid-cols-3">
      <Cell label="Pessoas" value={v.people} />
      <Cell label="Falando" value={v.speaker} />
      <Cell
        label={v.heard !== null || status.sleeping ? "Ouviu" : "Microfone"}
        value={status.sleeping ? (v.heard ? `“${v.heard}”` : "—") : v.mic === null ? "desligado" : v.mic > 0.04 ? "ouvindo" : "silêncio"}
        extra={!status.sleeping && v.mic !== null ? <MicMeter level={v.mic} /> : undefined}
      />
    </Card>
  );
}
