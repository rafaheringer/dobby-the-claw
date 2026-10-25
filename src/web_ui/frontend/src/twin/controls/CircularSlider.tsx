import { useRef } from "react";

const SWEEP = 270; // degrees of arc, open at the bottom
const START = 135; // where the arc begins, clockwise from +x (screen)
const R = 30;

const polar = (deg: number) => {
  const a = (deg * Math.PI) / 180;
  return [40 + R * Math.cos(a), 40 + R * Math.sin(a)];
};

function arc(fromDeg: number, toDeg: number): string {
  const [x1, y1] = polar(fromDeg);
  const [x2, y2] = polar(toDeg);
  return `M ${x1} ${y1} A ${R} ${R} 0 ${toDeg - fromDeg > 180 ? 1 : 0} 1 ${x2} ${y2}`;
}

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange(v: number): void;
}

/** A knob for a rotation (antennas, body): drag around the ring. */
export function CircularSlider({ label, value, min, max, onChange }: Props) {
  const ring = useRef<SVGSVGElement>(null);
  const dragging = useRef(false);
  const u = (value - min) / (max - min);
  const at = START + u * SWEEP;
  const mid = START + 0.5 * SWEEP;
  const [kx, ky] = polar(at);

  const update = (e: React.PointerEvent) => {
    const r = ring.current!.getBoundingClientRect();
    let deg = (Math.atan2(e.clientY - (r.top + r.height / 2), e.clientX - (r.left + r.width / 2)) * 180) / Math.PI;
    deg = (deg - START + 360) % 360; // 0..360 measured from the arc start
    if (deg > SWEEP) deg = deg - SWEEP < (360 - SWEEP) / 2 ? SWEEP : 0; // in the gap: snap to the nearer end
    onChange(min + (deg / SWEEP) * (max - min));
  };

  return (
    <div className="flex flex-col items-center gap-1">
      <svg
        ref={ring}
        viewBox="0 0 80 80"
        role="slider"
        aria-label={label}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={Number(value.toFixed(2))}
        tabIndex={0}
        onKeyDown={(e) => {
          const step = (max - min) / 40;
          if (e.key === "ArrowRight" || e.key === "ArrowUp") onChange(Math.min(max, value + step));
          if (e.key === "ArrowLeft" || e.key === "ArrowDown") onChange(Math.max(min, value - step));
        }}
        onPointerDown={(e) => {
          dragging.current = true;
          e.currentTarget.setPointerCapture(e.pointerId);
          update(e);
        }}
        onPointerMove={(e) => dragging.current && update(e)}
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
        className="size-20 touch-none rounded-full outline-none focus-visible:ring-2 focus-visible:ring-accent"
      >
        <path d={arc(START, START + SWEEP)} fill="none" stroke="var(--line)" strokeWidth="4" strokeLinecap="butt" />
        <path d={at >= mid ? arc(mid, at) : arc(at, mid)} fill="none" stroke="var(--ink)" strokeWidth="4" strokeLinecap="butt" />
        <circle cx={kx} cy={ky} r="6" fill="var(--accent)" stroke="var(--ink)" strokeWidth="2" />
        <text x="40" y="44" textAnchor="middle" fontSize="11" fill="var(--text)" className="font-mono tabular">
          {Math.round((value * 180) / Math.PI)}°
        </text>
      </svg>
      <span className="label">{label}</span>
    </div>
  );
}
