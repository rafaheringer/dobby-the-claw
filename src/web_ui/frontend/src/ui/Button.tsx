import { motion, type HTMLMotionProps } from "motion/react";
import type { ReactNode } from "react";
import { spring } from "../app/motion";
import { cx } from "./Surface";

// "sm" matches the height of a Pill, for buttons that sit next to one in a bar.
type Props = Omit<HTMLMotionProps<"button">, "children"> & {
  variant?: "outline" | "primary";
  size?: "md" | "sm";
  busy?: boolean;
  icon?: boolean;
  children?: ReactNode;
};

export function Button({ variant = "outline", size = "md", busy = false, icon = false, disabled, className, children, ...rest }: Props) {
  return (
    <motion.button
      whileTap={{ scale: 0.97 }}
      transition={spring}
      disabled={disabled || busy}
      className={cx(
        "inline-flex shrink-0 items-center justify-center gap-2 border font-mono text-xs font-medium uppercase tracking-[0.08em] transition-colors",
        size === "sm" ? "h-8" : "h-10",
        icon ? (size === "sm" ? "w-8" : "w-10") : size === "sm" ? "px-3" : "px-4",
        "disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent",
        variant === "primary" ? "border-ink bg-ink text-white hover:bg-[#333]" : "border-ink bg-surface text-ink hover:bg-subtle",
        className,
      )}
      {...rest}
    >
      {busy && <span className="size-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />}
      {children}
    </motion.button>
  );
}
