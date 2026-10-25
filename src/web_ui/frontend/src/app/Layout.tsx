import { AnimatePresence, motion } from "motion/react";
import { lazy, Suspense, useState } from "react";
import { useLocation, useOutlet } from "react-router";
import Home from "../routes/Home";
import Twin from "../routes/Twin";
// three.js is most of the bundle: load the 3D stage in its own chunk, after the page is up.
const TwinStage = lazy(() => import("../twin/TwinStage").then((m) => ({ default: m.TwinStage })));
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
  // Same element for "/" and "/twin", so home is not remounted (camera restarted) when the twin opens.
  const page = pageKey === "/" ? <Home /> : outlet;
  const dir = useDirection(sectionIndex(pageKey));
  const shift = reduced ? 0 : 24 * dir;

  return (
    <>
      {/* While the twin is open the whole app behind it blurs away. A filter would make this the
          containing block of fixed children (composer), so it is only set while the twin is open. */}
      <motion.div
        animate={
          twin
            ? { opacity: 0, filter: "blur(12px)", scale: 0.97 }
            : { opacity: 1, filter: "blur(0px)", scale: 1, transitionEnd: { filter: "none", transform: "none" } }
        }
        transition={spring}
        // inert: the invisible app must not take clicks or focus (e.g. the real Sleep button)
        inert={twin || undefined}
        className="flex min-h-dvh"
      >
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex items-center justify-between gap-3 border-b border-line bg-surface px-4 pb-3 pt-[calc(env(safe-area-inset-top)+12px)] lg:hidden">
          <span className="font-mono text-base font-semibold tracking-[0.08em]">DOBBY</span>
          <div className="flex items-center gap-2">
            <StatePill />
            <WakeToggle compact />
          </div>
        </header>
        <main className="relative max-w-[1280px] flex-1 px-4 pb-[calc(env(safe-area-inset-bottom)+80px)] pt-4 lg:px-8 lg:pb-8 lg:pt-7">
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
        </main>
      </div>
      </motion.div>
      <Suspense fallback={null}>
        <TwinStage />
      </Suspense>
      {!twin && <TabBar />}
      {/* rendered here (keyed) rather than via the outlet so its exit animation can play */}
      <AnimatePresence>{twin && <Twin key="twin" />}</AnimatePresence>
      <Toasts />
    </>
  );
}
