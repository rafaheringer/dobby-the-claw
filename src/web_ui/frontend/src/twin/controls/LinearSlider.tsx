import { Slider } from "radix-ui";
import { cx } from "../../ui/Surface";

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  vertical?: boolean;
  onChange(v: number): void;
}

/** A slider (Radix) for one axis; vertical for height, horizontal for roll. */
export function LinearSlider({ label, value, min, max, vertical = false, onChange }: Props) {
  return (
    <div className={cx("flex items-center gap-2", vertical ? "flex-col" : "w-full")}>
      {!vertical && <span className="w-16 shrink-0 label">{label}</span>}
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
        <Slider.Track className={cx("relative grow overflow-hidden bg-line", vertical ? "w-1" : "h-1")}>
          <Slider.Range className={cx("absolute bg-ink", vertical ? "w-full" : "h-full")} />
        </Slider.Track>
        <Slider.Thumb className="block size-5 rounded-full border-2 border-ink bg-accent outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2" />
      </Slider.Root>
      {vertical && <span className="label">{label}</span>}
    </div>
  );
}
