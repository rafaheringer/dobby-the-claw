import { motion } from "motion/react";
import { useRef, useState } from "react";
import { spring } from "../../app/motion";
import { cx } from "../../ui/Surface";

const clamp1 = (v: number) => Math.max(-1, Math.min(1, v));
// Pad and knob sizes; the knob travels until its edge meets the pad's.
const SIZES = {
  md: { pad: "size-32", knob: "size-10 -ml-5 -mt-5", travel: 44 },
  sm: { pad: "size-24", knob: "size-8 -ml-4 -mt-4", travel: 32 },
};

/**
 * A 2D pad: x and y in [-1, 1] (right / down positive). The knob shows `x`, `y`: kept where it is
 * released for a position (the head it points), or put back at the center through `onRelease`
 * for a speed (the camera stick).
 */
export function Joystick({
  x,
  y,
  label,
  size = "md",
  showLabel = true,
  className,
  onChange,
  onRelease,
}: {
  x: number;
  y: number;
  label: string;
  size?: keyof typeof SIZES;
  showLabel?: boolean;
  className?: string;
  onChange(x: number, y: number): void;
  onRelease?(): void;
}) {
  const dims = SIZES[size];
  const pad = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const held = useRef(false); // the state lags a render: a fast drag's first moves would be lost

  const update = (e: React.PointerEvent) => {
    const r = pad.current!.getBoundingClientRect();
    const nx = clamp1(((e.clientX - r.left) / r.width) * 2 - 1);
    const ny = clamp1(((e.clientY - r.top) / r.height) * 2 - 1);
    const len = Math.hypot(nx, ny);
    onChange(len > 1 ? nx / len : nx, len > 1 ? ny / len : ny); // keep the knob inside the circle
  };

  return (
    <div className="flex flex-col items-center gap-1.5">
      <div
        ref={pad}
        role="slider"
        aria-label={label}
        aria-valuetext={`${Math.round(x * 100)}, ${Math.round(y * 100)}`}
        tabIndex={0}
        onPointerDown={(e) => {
          held.current = true;
          setDragging(true);
          e.currentTarget.setPointerCapture(e.pointerId);
          update(e);
        }}
        onPointerMove={(e) => held.current && update(e)}
        onPointerUp={() => {
          held.current = false;
          setDragging(false);
          onRelease?.();
        }}
        onPointerCancel={() => {
          held.current = false;
          setDragging(false);
          onRelease?.();
        }}
        className={cx(
          "relative touch-none rounded-full border border-ink outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2",
          dims.pad,
          className ?? "bg-subtle",
        )}
      >
        <span className="absolute left-1/2 top-2 bottom-2 w-px -translate-x-1/2 bg-line" />
        <span className="absolute top-1/2 left-2 right-2 h-px -translate-y-1/2 bg-line" />
        <motion.span
          className={cx("absolute left-1/2 top-1/2 rounded-full border-2 border-ink bg-accent", dims.knob)}
          animate={{ x: x * dims.travel, y: y * dims.travel }}
          transition={dragging ? { duration: 0 } : spring}
        />
      </div>
      {showLabel && <span className="label">{label}</span>}
    </div>
  );
}
