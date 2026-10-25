import { motion } from "motion/react";
import { useMotionPreset } from "../app/motion";
import type { ChatEntry } from "../live/protocol";
import { cx } from "../ui/Surface";

const time = (t: number) => new Date(t * 1000).toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit" });

export function Bubble({ entry }: { entry: ChatEntry }) {
  const { spring } = useMotionPreset();
  const mine = entry.who === "you";
  return (
    <motion.li
      layout="position"
      initial={{ opacity: 0, y: 12, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={spring}
      className={cx("flex max-w-[85%] flex-col gap-1", mine ? "items-end self-end" : "items-start self-start")}
    >
      <div
        className={cx(
          "whitespace-pre-wrap break-words px-4 py-2.5 text-[15px] leading-snug",
          mine ? "bg-ink text-white" : "surface shadow-[inset_3px_0_0_var(--accent)]",
        )}
      >
        {entry.text}
      </div>
      <span className="label px-0.5 text-[10px]">
        {mine ? "Você" : "Dobby"} · {time(entry.t)}
      </span>
    </motion.li>
  );
}
