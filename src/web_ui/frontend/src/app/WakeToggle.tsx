import { Moon, Sun } from "lucide-react";
import { send, useLive } from "../live/store";
import { Button } from "../ui/Button";

/** One button that puts Dobby to sleep or wakes him, depending on how he is now. */
export function WakeToggle({ className, compact = false }: { className?: string; compact?: boolean }) {
  const status = useLive((s) => s.status);
  const open = useLive((s) => s.connection === "open");
  const asleep = !!status?.sleeping;
  const label = asleep ? "Acordar" : "Dormir";
  return (
    <Button
      className={className}
      icon={compact}
      size={compact ? "sm" : "md"}
      variant={asleep ? "primary" : "outline"}
      busy={!!status?.transition}
      disabled={!open || !status}
      aria-label={label}
      onClick={() => send({ type: "cmd", name: asleep ? "wake" : "sleep" })}
    >
      {!status?.transition && (asleep ? <Sun className="size-4" /> : <Moon className="size-4" />)}
      {!compact && label}
    </Button>
  );
}
