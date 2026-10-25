import { motion } from "motion/react";
import { Link, useLocation } from "react-router";
import { useMotionPreset } from "./motion";
import { NAV } from "./nav";
import { sectionIndex } from "./routes";

/** Bottom navigation for phones: a white bar with a hairline on top; the active tab gets an orange mark. */
export function TabBar() {
  const { pathname } = useLocation();
  const active = sectionIndex(pathname);
  const { spring } = useMotionPreset();
  return (
    <nav className="pb-safe fixed inset-x-0 bottom-0 z-30 border-t border-ink bg-surface lg:hidden">
      <div className="mx-auto flex max-w-md">
        {NAV.map(({ path, label, Icon }, i) => (
          <Link
            key={path}
            to={path}
            aria-current={i === active ? "page" : undefined}
            className={`relative flex h-14 flex-1 flex-col items-center justify-center gap-1 font-mono text-[10px] font-medium uppercase tracking-[0.08em] ${i === active ? "text-text" : "text-dim"}`}
          >
            {i === active && <motion.span layoutId="tab-indicator" transition={spring} className="absolute inset-x-3 top-0 h-[3px] bg-accent" />}
            <Icon className="size-5" strokeWidth={1.75} />
            {label}
          </Link>
        ))}
      </div>
    </nav>
  );
}
