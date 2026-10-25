import { useEffect, useRef } from "react";
import { useAudio } from "../live/store";
import { micLevel, pushSample } from "./audioValues";

const HISTORY = 60;
const STEP_MS = 50; // 20 samples a second, like the official app's visualizer
const INK = [17, 17, 17] as const;

/**
 * A scrolling waveform of the mic level: a line with a soft fill under it and a dot at the newest
 * sample. Reads the 10 Hz audio stream directly, so the page around it never re-renders.
 */
export function Waveform({ active }: { active: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const el = canvas.current;
    const ctx = el?.getContext("2d");
    if (!el || !ctx) return;
    let history: number[] = Array(HISTORY).fill(0.05);
    let last = 0;
    let frame = 0;
    const [r, g, b] = INK;

    const draw = (now: number) => {
      frame = requestAnimationFrame(draw);
      const dpr = window.devicePixelRatio || 1;
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (!w || !h) return;
      if (el.width !== Math.round(w * dpr)) el.width = Math.round(w * dpr);
      if (el.height !== Math.round(h * dpr)) el.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (!active) {
        history = Array(HISTORY).fill(0.05);
        return;
      }
      if (now - last >= STEP_MS) {
        const rms = useAudio.getState().sample?.rms;
        history = pushSample(history, rms == null ? 0 : micLevel(rms), HISTORY);
        last = now;
      }
      const stepX = w / (history.length - 1);
      const y = (v: number) => h - v * h * 0.8 - h * 0.1;
      const line = new Path2D();
      history.forEach((v, i) => (i === 0 ? line.moveTo(0, y(v)) : line.lineTo(i * stepX, y(v))));
      const fill = new Path2D(line);
      fill.lineTo(w, h);
      fill.lineTo(0, h);
      fill.closePath();
      const gradient = ctx.createLinearGradient(0, h * 0.4, 0, h);
      gradient.addColorStop(0, `rgba(${r},${g},${b},0.12)`);
      gradient.addColorStop(1, `rgba(${r},${g},${b},0.04)`);
      ctx.fillStyle = gradient;
      ctx.fill(fill);
      ctx.strokeStyle = `rgba(${r},${g},${b},0.45)`;
      ctx.lineWidth = 1.5;
      ctx.lineJoin = ctx.lineCap = "round";
      ctx.stroke(line);
      ctx.fillStyle = "#ff5a1f";
      ctx.beginPath();
      ctx.arc(w - 2, y(history[history.length - 1]), 2, 0, Math.PI * 2);
      ctx.fill();
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [active]);

  return <canvas ref={canvas} aria-hidden className="block h-8 w-full" />;
}
