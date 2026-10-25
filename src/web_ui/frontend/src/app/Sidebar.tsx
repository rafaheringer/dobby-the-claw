import { motion } from "motion/react";
import { Link, useLocation } from "react-router";
import { Card } from "../ui/Glass";
import { useMotionPreset } from "./motion";
import { NAV } from "./nav";
import { sectionIndex } from "./routes";
import { StatePill } from "./StatePill";
import { WakeToggle } from "./WakeToggle";

/** Desktop navigation. */
export function Sidebar() {
  const { pathname } = useLocation();
  const active = sectionIndex(pathname);
  const { spring } = useMotionPreset();
  return (
    <Card className="sticky top-4 hidden h-[calc(100dvh-2rem)] w-60 shrink-0 flex-col p-4 lg:flex">
      <div className="mb-6 flex items-center gap-3 px-2 pt-1">
        <img src="/icon.svg" alt="" className="size-9 rounded-xl" />
        <span className="text-lg font-semibold">Dobby</span>
      </div>
      <nav className="flex flex-col gap-1">
        {NAV.map(({ path, label, Icon }, i) => (
          <Link
            key={path}
            to={path}
            aria-current={i === active ? "page" : undefined}
            className={`relative flex h-11 items-center gap-3 rounded-2xl px-3 text-sm font-medium ${i === active ? "text-text" : "text-dim hover:text-text"}`}
          >
            {i === active && <motion.span layoutId="side-indicator" transition={spring} className="absolute inset-0 rounded-2xl bg-white/10" />}
            <Icon className={`relative size-[18px] ${i === active ? "text-accent" : ""}`} />
            <span className="relative">{label}</span>
          </Link>
        ))}
      </nav>
      <div className="mt-auto flex flex-col gap-3">
        <StatePill className="self-start" />
        <WakeToggle className="w-full" />
      </div>
    </Card>
  );
}
