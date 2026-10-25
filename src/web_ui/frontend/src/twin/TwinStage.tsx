import { Canvas } from "@react-three/fiber";
import { animate, motion, useMotionValue } from "motion/react";
import { Component, Suspense, useEffect, useRef, useState, type ReactNode } from "react";
import { useLocation, useNavigate } from "react-router";
import { useMotionPreset } from "../app/motion";
import { usePageVisible } from "../lib/usePageVisible";
import { CAMERA_FOV, stageModeFor } from "./framing";
import { Scene } from "./Scene";
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

  const top = useMotionValue(0);
  const left = useMotionValue(0);
  const width = useMotionValue(0);
  const height = useMotionValue(0);
  const settling = useRef(false);
  const placed = useRef(false);

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
    settling.current = true;
    const runs = [animate(top, target.top, spring), animate(left, target.left, spring), animate(width, target.width, spring), animate(height, target.height, spring)];
    Promise.all(runs).then(() => (settling.current = false));
    return () => runs.forEach((r) => r.stop());
  }, [mode, slot, spring, reduced, top, left, width, height]);

  // ...and otherwise stick to it as the page scrolls or the window resizes.
  useEffect(() => {
    if (mode === "hidden") return;
    let frame = 0;
    const follow = () => {
      frame = requestAnimationFrame(follow);
      if (settling.current) return;
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
      style={{ top, left, width, height, zIndex: mode === "full" ? 40 : 10, pointerEvents: mode === "head" ? "auto" : "none" }}
      animate={{ opacity: mode === "hidden" ? 0 : 1 }}
      transition={{ duration: 0.2 }}
      onClick={mode === "head" ? () => navigate("/twin") : undefined}
    >
      <StageBoundary>
        <Canvas
          dpr={[1, 2]}
          gl={{ alpha: true, antialias: true }}
          camera={{ fov: CAMERA_FOV, near: 0.01, far: 20, position: [0, 0.3, 1] }}
          frameloop={drawing ? "always" : "never"}
          className={mode === "head" ? "cursor-pointer" : undefined}
          aria-label={mode === "head" ? "Abrir o Dobby em 3D" : undefined}
        >
          <Suspense fallback={null}>
            <Scene full={mode === "full"} snap={reduced} />
          </Suspense>
        </Canvas>
      </StageBoundary>
    </motion.div>
  );
}
