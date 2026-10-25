import { motion, type HTMLMotionProps } from "motion/react";
import type { ReactNode } from "react";
import { spring } from "../app/motion";
import { cx } from "./Glass";

type Props = Omit<HTMLMotionProps<"button">, "children"> & { variant?: "glass" | "primary"; busy?: boolean; icon?: boolean; children?: ReactNode };

export function Button({ variant = "glass", busy = false, icon = false, disabled, className, children, ...rest }: Props) {
  return (
    <motion.button
      whileTap={{ scale: 0.96 }}
      transition={spring}
      disabled={disabled || busy}
      className={cx(
        "inline-flex h-10 shrink-0 items-center justify-center gap-2 rounded-full text-sm font-medium transition-colors",
        icon ? "w-10" : "px-4",
        "disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-accent",
        variant === "primary" ? "bg-accent text-accent-ink hover:brightness-110" : "glass hover:bg-glass-strong",
        className,
      )}
      {...rest}
    >
      {busy && <span className="size-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />}
      {children}
    </motion.button>
  );
}
