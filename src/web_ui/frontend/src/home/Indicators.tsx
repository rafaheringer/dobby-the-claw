import { AnimatePresence, motion } from "motion/react";
import type { ReactNode } from "react";
import { spring } from "../app/motion";
import { useLive } from "../live/store";
import { Card } from "../ui/Glass";
import { indicatorsFrom } from "./indicatorValues";
import { MicMeter } from "./MicMeter";

/** A value that slides in when it changes. */
function Value({ children }: { children: string }) {
  return (
    <span className="relative block h-6 overflow-hidden">
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.span
          key={children}
          initial={{ y: 14, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: -14, opacity: 0 }}
          transition={spring}
          className="block truncate text-base font-semibold"
        >
          {children}
        </motion.span>
      </AnimatePresence>
    </span>
  );
}

function Tile({ label, value, extra }: { label: string; value: string; extra?: ReactNode }) {
  return (
    <Card className="min-w-0 flex-1 px-4 py-3">
      <div className="text-xs text-dim">{label}</div>
      <Value>{value}</Value>
      {extra}
    </Card>
  );
}

export function Indicators() {
  const status = useLive((s) => s.status);
  if (!status) return null;
  const v = indicatorsFrom(status);
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      <Tile label="Pessoas" value={v.people} />
      <Tile label="Falando" value={v.speaker} />
      <Tile
        label={v.heard !== null || status.sleeping ? "Ouviu" : "Microfone"}
        value={status.sleeping ? (v.heard ? `“${v.heard}”` : "—") : v.mic === null ? "desligado" : v.mic > 0.04 ? "ouvindo" : "silêncio"}
        extra={!status.sleeping && v.mic !== null ? <MicMeter level={v.mic} /> : undefined}
      />
    </div>
  );
}
