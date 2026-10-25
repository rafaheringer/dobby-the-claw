import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { useLocation, useOutlet } from "react-router";
import Home from "../routes/Home";
import { Toasts } from "../ui/Toasts";
import { useMotionPreset } from "./motion";
import { sectionIndex } from "./routes";
import { Sidebar } from "./Sidebar";
import { StatePill } from "./StatePill";
import { TabBar } from "./TabBar";
import { WakeToggle } from "./WakeToggle";

/** Which way the page slides: forward when moving to a tab further right. */
function useDirection(index: number): number {
  const [seen, setSeen] = useState({ index, dir: 0 });
  if (index >= 0 && index !== seen.index) {
    const next = { index, dir: index > seen.index ? 1 : -1 };
    setSeen(next); // React re-renders at once with the new value (derived state)
    return next.dir;
  }
  return seen.dir;
}

export default function Layout() {
  const { pathname } = useLocation();
  const outlet = useOutlet();
  const { spring, fade, reduced } = useMotionPreset();
  const twin = pathname === "/twin";
  // The twin opens over the home page, which stays mounted (blurred) underneath.
  const pageKey = twin ? "/" : pathname;
  const page = twin ? <Home /> : outlet;
  const dir = useDirection(sectionIndex(pageKey));
  const shift = reduced ? 0 : 24 * dir;

  return (
    <div className="mx-auto flex min-h-dvh max-w-[1500px] gap-4 lg:p-4">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="pt-safe sticky top-0 z-20 flex items-center justify-between gap-3 px-4 pb-2 pt-3 lg:hidden">
          <span className="text-lg font-semibold">Dobby</span>
          <div className="flex items-center gap-2">
            <StatePill />
            <WakeToggle compact />
          </div>
        </header>
        <motion.main
          animate={twin ? { opacity: 0, filter: "blur(12px)", scale: 0.97 } : { opacity: 1, filter: "blur(0px)", scale: 1 }}
          transition={spring}
          className="relative flex-1 px-4 pb-[calc(env(safe-area-inset-bottom)+96px)] lg:px-2 lg:pb-4"
          aria-hidden={twin || undefined}
        >
          <AnimatePresence mode="popLayout" initial={false} custom={shift}>
            <motion.div
              key={pageKey}
              custom={shift}
              variants={{
                enter: (x: number) => ({ opacity: 0, x }),
                center: { opacity: 1, x: 0 },
                exit: (x: number) => ({ opacity: 0, x: -x }),
              }}
              initial="enter"
              animate="center"
              exit="exit"
              transition={reduced ? fade : { ...spring, opacity: fade }}
            >
              {page}
            </motion.div>
          </AnimatePresence>
        </motion.main>
      </div>
      {!twin && <TabBar />}
      <AnimatePresence>{twin && outlet}</AnimatePresence>
      <Toasts />
    </div>
  );
}
