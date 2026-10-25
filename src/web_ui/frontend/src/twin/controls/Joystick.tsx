import { motion } from "motion/react";
import { useRef, useState } from "react";
import { spring } from "../../app/motion";

const clamp1 = (v: number) => Math.max(-1, Math.min(1, v));

/**
 * A 2D pad: x and y in [-1, 1] (right / down positive). The knob stays where it is released,
 * like the head it points.
 */
export function Joystick({ x, y, label, onChange }: { x: number; y: number; label: string; onChange(x: number, y: number): void }) {
  const pad = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);

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
          setDragging(true);
          e.currentTarget.setPointerCapture(e.pointerId);
          update(e);
        }}
        onPointerMove={(e) => dragging && update(e)}
        onPointerUp={() => setDragging(false)}
        onPointerCancel={() => setDragging(false)}
        className="relative size-32 touch-none rounded-full border border-glass-border bg-[radial-gradient(circle,rgba(255,255,255,.08),rgba(255,255,255,.02)_70%)] outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        <span className="absolute left-1/2 top-2 bottom-2 w-px -translate-x-1/2 bg-white/10" />
        <span className="absolute top-1/2 left-2 right-2 h-px -translate-y-1/2 bg-white/10" />
        <motion.span
          className="absolute left-1/2 top-1/2 -ml-5 -mt-5 size-10 rounded-full bg-accent shadow-[0_0_18px_rgba(154,168,255,.7)]"
          animate={{ x: x * 44, y: y * 44 }}
          transition={dragging ? { duration: 0 } : spring}
        />
      </div>
      <span className="text-xs text-dim">{label}</span>
    </div>
  );
}
