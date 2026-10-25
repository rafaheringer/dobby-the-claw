import { motion } from "motion/react";
import { Link, useLocation } from "react-router";
import { useMotionPreset } from "./motion";
import { NAV } from "./nav";
import { sectionIndex } from "./routes";

/** Floating bottom navigation for phones. */
export function TabBar() {
  const { pathname } = useLocation();
  const active = sectionIndex(pathname);
  const { spring } = useMotionPreset();
  return (
    <nav className="fixed inset-x-3 bottom-[calc(env(safe-area-inset-bottom)+10px)] z-30 lg:hidden">
      <div className="glass mx-auto flex max-w-md justify-around rounded-[28px] bg-[#1a1e38]/60 p-1.5 shadow-[0_10px_40px_rgba(0,0,0,.4)]">
        {NAV.map(({ path, label, Icon }, i) => (
          <Link
            key={path}
            to={path}
            aria-current={i === active ? "page" : undefined}
            className="relative flex h-14 flex-1 flex-col items-center justify-center gap-0.5 rounded-3xl text-[11px] font-medium"
          >
            {i === active && <motion.span layoutId="tab-indicator" transition={spring} className="absolute inset-0 rounded-3xl bg-white/12" />}
            <Icon className={`relative size-5 ${i === active ? "text-accent" : "text-dim"}`} />
            <span className={`relative ${i === active ? "text-text" : "text-dim"}`}>{label}</span>
          </Link>
        ))}
      </div>
    </nav>
  );
}
