import { Canvas } from "@react-three/fiber";
import { animate, motion, useMotionValue } from "motion/react";
import { Component, Suspense, useEffect, useRef, useState, type ReactNode } from "react";
import { useLocation, useNavigate } from "react-router";
import { useMotionPreset } from "../app/motion";
import { usePageVisible } from "../lib/usePageVisible";
import { CAMERA_FOV, stageModeFor } from "./framing";
import { dragOrbit, resetOrbit, useOrbit, zoomOrbit } from "./orbit";
import { Scene, snapCamera } from "./Scene";
import { useStageStatus } from "./stageStatus";
import { useTwinSlot } from "./TwinSlot";

class StageBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(error: unknown) {
    console.warn("3D robot unavailable:", error);
    useStageStatus.setState({ failed: true });
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

type Rect = { top: number; left: number; width: number; height: number };
const SETTLE_MS = 700; // how long a mode change animates before the stage just follows the slot
const DESKTOP_MIN_PX = 1024;
// On phones the head's slot and the full screen differ too much for the stage to grow between them
// (the robot would fly in from the top): it fades out, reframes, and fades back in instead.
const SWAP_MS = 220;
const viewport = (): Rect => ({ top: 0, left: 0, width: window.innerWidth, height: window.innerHeight });

/** True while `el` is at least partly on screen. */
function useOnScreen(el: HTMLElement | null): boolean {
  const [on, setOn] = useState(true);
  useEffect(() => {
    if (!el) return;
    const observer = new IntersectionObserver(([entry]) => setOn(entry.isIntersecting));
    observer.observe(el);
    return () => observer.disconnect();
  }, [el]);
  return on;
}

/** Drag to turn around the robot, pinch or wheel to zoom, double-click for the front view. */
function useOrbitGestures(enabled: boolean) {
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const pinch = useRef(0);
  const end = (e: React.PointerEvent) => {
    pointers.current.delete(e.pointerId);
    pinch.current = 0;
  };
  if (!enabled) return {};
  return {
    onPointerDown(e: React.PointerEvent) {
      e.currentTarget.setPointerCapture(e.pointerId);
      pointers.current.set(e.pointerId, { x: e.clientX, y: e.clientY });
    },
    onPointerMove(e: React.PointerEvent) {
      const all = pointers.current;
      const prev = all.get(e.pointerId);
      if (!prev) return;
      all.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (all.size === 1) {
        useOrbit.setState((o) => dragOrbit(o, e.clientX - prev.x, e.clientY - prev.y, e.currentTarget.clientHeight));
      } else if (all.size === 2) {
        const [a, b] = [...all.values()];
        const gap = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch.current && gap) useOrbit.setState((o) => zoomOrbit(o, pinch.current / gap));
        pinch.current = gap;
      }
    },
    onPointerUp: end,
    onPointerCancel: end,
    onWheel(e: React.WheelEvent) {
      useOrbit.setState((o) => zoomOrbit(o, Math.exp(e.deltaY * 0.001)));
    },
    onDoubleClick: resetOrbit,
  };
}

/**
 * One WebGL canvas for the whole app. On "/" it sits over the home page's slot and shows the head;
 * on "/twin" it grows to full screen and shows the whole robot; elsewhere it fades out and stops.
 */
export function TwinStage() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const mode = stageModeFor(pathname);
  const slot = useTwinSlot((s) => s.el);
  const { failed } = useStageStatus();
  const pageVisible = usePageVisible();
  const slotOnScreen = useOnScreen(slot);
  const { spring, reduced } = useMotionPreset();

  const shown = useMotionValue(1);
  const top = useMotionValue(0);
  const left = useMotionValue(0);
  const width = useMotionValue(0);
  const height = useMotionValue(0);
  const settleUntil = useRef(0);
  const placed = useRef(false);
  const gestures = useOrbitGestures(mode === "full");
  // Each visit to the full twin starts from the front.
  useEffect(() => {
    if (mode !== "full") resetOrbit();
  }, [mode]);

  // Move to the new frame with the spring when the mode changes...
  useEffect(() => {
    const target = mode === "full" ? viewport() : slot ? slot.getBoundingClientRect() : null;
    if (!target) return;
    if (!placed.current || reduced) {
      top.jump(target.top);
      left.jump(target.left);
      width.jump(target.width);
      height.jump(target.height);
      placed.current = true;
      return;
    }
    if (window.innerWidth < DESKTOP_MIN_PX) {
      shown.jump(0);
      settleUntil.current = 0;
      top.jump(target.top);
      left.jump(target.left);
      width.jump(target.width);
      height.jump(target.height);
      snapCamera(SWAP_MS);
      const run = animate(shown, 1, { duration: 0.25, delay: SWAP_MS / 1000 });
      return () => run.stop();
    }
    settleUntil.current = performance.now() + SETTLE_MS;
    const runs = [animate(top, target.top, spring), animate(left, target.left, spring), animate(width, target.width, spring), animate(height, target.height, spring)];
    return () => runs.forEach((r) => r.stop());
  }, [mode, slot, spring, reduced, top, left, width, height, shown]);

  // ...and otherwise stick to it as the page scrolls or the window resizes.
  useEffect(() => {
    if (mode === "hidden") return;
    let frame = 0;
    const follow = () => {
      frame = requestAnimationFrame(follow);
      if (performance.now() < settleUntil.current) return; // the spring is still moving it
      const r = mode === "full" ? viewport() : slot?.getBoundingClientRect();
      if (!r) return;
      top.jump(r.top);
      left.jump(r.left);
      width.jump(r.width);
      height.jump(r.height);
    };
    frame = requestAnimationFrame(follow);
    return () => cancelAnimationFrame(frame);
  }, [mode, slot, top, left, width, height]);

  if (failed) return null;
  const drawing = mode !== "hidden" && pageVisible && (mode === "full" || slotOnScreen);

  return (
    <motion.div
      aria-hidden={mode !== "full"}
      className="fixed"
      style={{
        top,
        left,
        width,
        height,
        zIndex: mode === "full" ? 40 : 10,
        pointerEvents: mode === "hidden" ? "none" : "auto",
        touchAction: mode === "full" ? "none" : undefined,
      }}
      {...gestures}
      animate={{ opacity: mode === "hidden" ? 0 : 1 }}
      transition={{ duration: 0.2 }}
      onClick={mode === "head" ? () => navigate("/twin") : undefined}
    >
      <motion.div className="size-full" style={{ opacity: shown }}>
      <StageBoundary>
        <Canvas
          dpr={[1, 2]}
          gl={{ alpha: true, antialias: true }}
          camera={{ fov: CAMERA_FOV, near: 0.01, far: 20, position: [0, 0.3, 1] }}
          frameloop={drawing ? "always" : "never"}
          className={mode === "head" ? "cursor-pointer" : mode === "full" ? "cursor-grab active:cursor-grabbing" : undefined}
          aria-label={mode === "head" ? "Abrir o Dobby em 3D" : undefined}
        >
          <Suspense fallback={null}>
            <Scene full={mode === "full"} snap={reduced} />
          </Suspense>
        </Canvas>
      </StageBoundary>
      </motion.div>
    </motion.div>
  );
}
