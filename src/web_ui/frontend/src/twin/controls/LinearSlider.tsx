import { Slider } from "radix-ui";
import { cx } from "../../ui/Glass";

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  vertical?: boolean;
  onChange(v: number): void;
}

/** A glass slider (Radix) for one axis; vertical for height, horizontal for roll. */
export function LinearSlider({ label, value, min, max, vertical = false, onChange }: Props) {
  return (
    <div className={cx("flex items-center gap-2", vertical ? "flex-col" : "w-full")}>
      {!vertical && <span className="w-16 shrink-0 text-xs text-dim">{label}</span>}
      <Slider.Root
        value={[value]}
        min={min}
        max={max}
        step={(max - min) / 200}
        orientation={vertical ? "vertical" : "horizontal"}
        onValueChange={([v]) => onChange(v)}
        aria-label={label}
        className={cx("relative flex touch-none select-none items-center", vertical ? "h-32 w-8 flex-col" : "h-8 w-full")}
      >
        <Slider.Track className={cx("relative grow overflow-hidden rounded-full bg-white/12", vertical ? "w-1.5" : "h-1.5")}>
          <Slider.Range className={cx("absolute rounded-full bg-accent/70", vertical ? "w-full" : "h-full")} />
        </Slider.Track>
        <Slider.Thumb className="block size-6 rounded-full bg-white shadow-[0_0_12px_rgba(154,168,255,.6)] outline-none focus-visible:ring-2 focus-visible:ring-accent" />
      </Slider.Root>
      {vertical && <span className="text-xs text-dim">{label}</span>}
    </div>
  );
}
