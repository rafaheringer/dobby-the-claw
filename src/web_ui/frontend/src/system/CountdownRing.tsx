import { motion } from "motion/react";

const R = 52;
const C = 2 * Math.PI * R;

/** A ring that empties over `seconds`; `thinking` turns it into a spinner. */
export function CountdownRing({ seconds, thinking }: { seconds: number; thinking: boolean }) {
  return (
    <svg viewBox="0 0 120 120" className="size-full -rotate-90">
      <circle cx="60" cy="60" r={R} fill="none" stroke="rgba(255,255,255,.1)" strokeWidth="6" />
      {thinking ? (
        <motion.circle
          cx="60" cy="60" r={R} fill="none" stroke="var(--st-thinking)" strokeWidth="6" strokeLinecap="round"
          strokeDasharray={`${C * 0.25} ${C}`}
          animate={{ rotate: 360 }}
          style={{ originX: "60px", originY: "60px" }}
          transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
        />
      ) : (
        <motion.circle
          cx="60" cy="60" r={R} fill="none" stroke="var(--accent)" strokeWidth="6" strokeLinecap="round"
          strokeDasharray={C}
          initial={{ strokeDashoffset: 0 }}
          animate={{ strokeDashoffset: C }}
          transition={{ duration: seconds, ease: "linear" }}
        />
      )}
    </svg>
  );
}
