import { ChevronLeft, Crosshair } from "lucide-react";
import { motion, type Variants } from "motion/react";
import { useEffect } from "react";
import { useNavigate } from "react-router";
import { useMotionPreset } from "../app/motion";
import { HEAD_TURN_MAX, LIMITS, robotLink, type RobotTarget } from "../robot";
import { closeTwin } from "../twin/close";
import { CircularSlider } from "../twin/controls/CircularSlider";
import { EmotionMenu } from "../twin/controls/EmotionMenu";
import { Joystick } from "../twin/controls/Joystick";
import { LinearSlider } from "../twin/controls/LinearSlider";
import { keyDelta } from "../twin/keys";
import { Button } from "../ui/Button";
import { cx } from "../ui/Surface";
import { Pill } from "../ui/Pill";
import { HeadIllustration } from "../twin/HeadIllustration";
import { useStageStatus, useTwinPanel } from "../twin/stageStatus";

const SWIPE_CLOSE_PX = 120;

function useTwinKeys() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null;
      if (el?.closest("input, textarea, select, [role=slider], [role=menu]")) return;
      const delta = keyDelta(e.key, e);
      if (!delta) return;
      e.preventDefault();
      const t = robotLink.target();
      robotLink.setTarget(Object.fromEntries(Object.entries(delta).map(([k, d]) => [k, t[k as keyof typeof delta] + (d ?? 0)])) as Partial<RobotTarget>);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

function ModeSwitch() {
  const { mode } = robotLink.useLinkState();
  return (
    <div className="flex h-8 border border-ink bg-surface font-mono text-[11px] font-medium uppercase tracking-[0.08em]">
      {(["auto", "manual"] as const).map((m) => (
        <button
          key={m}
          onClick={() => robotLink.setManual(m === "manual")}
          aria-pressed={mode === m}
          className={cx("relative px-3", mode === m ? "text-white" : "text-dim")}
        >
          {mode === m && <motion.span layoutId="twin-mode" className="absolute inset-0 bg-ink" />}
          <span className="relative">{m === "auto" ? "Auto" : "Manual"}</span>
        </button>
      ))}
    </div>
  );
}

// Module-level so the ref callback is stable. Tells the 3D framing how much of a phone screen the
// controls take (on desktop the panel is a side column and this value is not used).
function measurePanel(el: HTMLElement | null) {
  if (!el) return;
  const observer = new ResizeObserver(() => useTwinPanel.setState({ phonePx: el.offsetHeight }));
  observer.observe(el);
  return () => {
    observer.disconnect();
    useTwinPanel.setState({ phonePx: 0 });
  };
}

/** The full digital twin: the 3D stage fills the screen (TwinStage) and these controls sit over it. */
export default function Twin() {
  const navigate = useNavigate();
  const { spring, fade, reduced } = useMotionPreset();
  const failed = useStageStatus((s) => s.failed);
  const target = robotLink.useTarget();
  const { simulated } = robotLink.useLinkState();
  const close = () => closeTwin(navigate, window.history.state?.idx);
  useTwinKeys();
  // The home page stays mounted underneath: without this the whole screen scrolls it, behind the twin.
  useEffect(() => {
    const els = [document.documentElement, document.body]; // both: iOS Safari ignores html alone
    const before = els.map((el) => el.style.overflow);
    els.forEach((el) => (el.style.overflow = "hidden"));
    return () => els.forEach((el, i) => (el.style.overflow = before[i]));
  }, []);
  // Leaving the controls hands the robot back to Dobby; a frozen head nobody is driving looks broken.
  useEffect(
    () => () => {
      if (robotLink.state().mode === "manual") robotLink.setManual(false);
    },
    [],
  );

  const set = (t: Partial<RobotTarget>) => robotLink.setTarget(t);
  const panel: Variants = {
    hidden: { opacity: 0, y: reduced ? 0 : 24 },
    shown: { opacity: 1, y: 0, transition: { ...spring, delayChildren: reduced ? 0 : 0.22, staggerChildren: reduced ? 0 : 0.04 } },
    exit: { opacity: 0, y: reduced ? 0 : 24, transition: fade },
  };
  const item: Variants = { hidden: { opacity: 0, y: reduced ? 0 : 12 }, shown: { opacity: 1, y: 0, transition: spring } };

  return (
    <div className="pointer-events-none fixed inset-0 z-[45] flex flex-col">
      <motion.header
        initial={{ opacity: 0, y: -12 }}
        animate={{ opacity: 1, y: 0, transition: { ...spring, delay: reduced ? 0 : 0.15 } }}
        exit={{ opacity: 0, transition: fade }}
        drag={reduced ? false : "y"}
        dragConstraints={{ top: 0, bottom: 0 }}
        dragElastic={{ top: 0, bottom: 0.6 }}
        onDragEnd={(_, info) => info.offset.y > SWIPE_CLOSE_PX && close()}
        className="pointer-events-auto flex touch-none items-center justify-between gap-2 px-4 pb-2 pt-[calc(env(safe-area-inset-top)+12px)] lg:mr-[384px] lg:pt-4"
      >
        <Button size="sm" onClick={close} aria-label="Voltar">
          <ChevronLeft className="size-4" /> Voltar
        </Button>
        <div className="flex items-center gap-2">
          {simulated ? <Pill dot="var(--st-thinking)">simulado</Pill> : <Pill dot="var(--accent)" pulse>ao vivo</Pill>}
          <ModeSwitch />
        </div>
      </motion.header>

      {failed && (
        <div className="pointer-events-auto mx-4 mt-6 flex flex-col items-center gap-3 text-center lg:mr-[400px]">
          <HeadIllustration className="w-28 opacity-80" />
          <p className="surface px-4 py-3 text-sm text-dim">O Dobby em 3D não pode ser mostrado neste aparelho. Os controles continuam funcionando.</p>
        </div>
      )}
      <motion.section
        variants={panel}
        initial="hidden"
        animate="shown"
        exit="exit"
        aria-label="Controles"
        ref={measurePanel}
        // Every control fits at once: the panel takes its content's height (never scrolls) and the
        // 3D view gets the rest of the screen above it.
        className={cx(
          "pointer-events-auto mt-auto overflow-hidden border-t border-ink bg-surface px-4 pb-[calc(env(safe-area-inset-bottom)+14px)] pt-4",
          "lg:absolute lg:bottom-0 lg:right-0 lg:top-0 lg:mt-0 lg:w-[368px] lg:border-l lg:border-t-0 lg:px-5 lg:py-7",
        )}
      >
        <div className="mx-auto flex max-w-md flex-col gap-3 lg:gap-6">
          <motion.div variants={item} className="flex items-end justify-between">
            <Joystick
              label="Cabeça"
              // relative to the body: the head's own turn range
              x={(target.yawBody - target.yaw) / HEAD_TURN_MAX}
              y={target.pitch / LIMITS.pitch[1]}
              onChange={(x, y) => set({ yaw: target.yawBody - x * HEAD_TURN_MAX, pitch: y * LIMITS.pitch[1] })}
            />
            <LinearSlider vertical label="Altura" value={target.z} min={LIMITS.z[0]} max={LIMITS.z[1]} onChange={(z) => set({ z })} />
            <CircularSlider label="Corpo" value={target.yawBody} min={LIMITS.yawBody[0]} max={LIMITS.yawBody[1]} onChange={(yawBody) => set({ yawBody })} />
          </motion.div>
          <motion.div variants={item}>
            <LinearSlider label="Inclinar" value={-target.roll} min={LIMITS.roll[0]} max={LIMITS.roll[1]} onChange={(v) => set({ roll: -v })} />
          </motion.div>
          <motion.div variants={item} className="flex items-center justify-between gap-3">
            <CircularSlider label="Antena esq." value={target.antennas[1]} min={LIMITS.antenna[0]} max={LIMITS.antenna[1]} onChange={(v) => set({ antennas: [target.antennas[0], v] })} />
            <CircularSlider label="Antena dir." value={target.antennas[0]} min={LIMITS.antenna[0]} max={LIMITS.antenna[1]} onChange={(v) => set({ antennas: [v, target.antennas[1]] })} />
            <div className="flex w-32 flex-col gap-2">
              <Button size="sm" className="w-full" onClick={() => robotLink.center()}>
                <Crosshair className="size-4" /> Centralizar
              </Button>
              <EmotionMenu />
            </div>
          </motion.div>
        </div>
      </motion.section>
    </div>
  );
}
