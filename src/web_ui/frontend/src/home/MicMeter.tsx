const BARS = 16;

/** A segmented level meter: lit segments in ink, the loudest in orange; grey when nearly silent. */
export function MicMeter({ level }: { level: number | null }) {
  const value = level ?? 0;
  const lit = Math.round(value * BARS);
  const hearing = value > 0.04;
  return (
    <div className="flex h-2.5 gap-[3px]" aria-hidden>
      {Array.from({ length: BARS }, (_, i) => (
        <span
          key={i}
          className="flex-1 transition-colors duration-150"
          style={{ background: i >= lit ? "var(--line)" : !hearing ? "#bdbdbd" : i === lit - 1 ? "var(--accent)" : "var(--ink)" }}
        />
      ))}
    </div>
  );
}
