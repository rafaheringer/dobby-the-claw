import { CameraOff, Ellipsis } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router";
import { fade } from "../app/motion";
import { useLive } from "../live/store";
import { loadPref, savePref } from "../lib/storage";
import { usePageVisible } from "../lib/usePageVisible";
import { streamWidth, videoUrl } from "../lib/video";
import { robotLink } from "../robot";
import { steer } from "../robot/camera";
import { Joystick } from "../twin/controls/Joystick";
import { Card, CardHeader, cx } from "../ui/Surface";
import { Menu } from "../ui/Menu";

const OVERLAYS_KEY = "dobby.overlays";
const RETRY_MS = 3000;

function useStreamWidth(ref: React.RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(960);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let timer: ReturnType<typeof setTimeout>;
    const measure = () => setWidth(streamWidth(el.clientWidth * (window.devicePixelRatio || 1)));
    measure();
    const observer = new ResizeObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(measure, 400);
    });
    observer.observe(el);
    return () => {
      observer.disconnect();
      clearTimeout(timer);
    };
  }, [ref]);
  return width;
}

const CAMERA_HOLD_MS = 30_000; // after this long without touching the joystick, Dobby takes the head back
const DEAD_ZONE = 0.06;

/**
 * Steers the camera like a security camera: while the stick is pushed the camera turns that way,
 * faster the further it is pushed; let go and the stick springs back and the camera stops. The
 * first push puts the robot in manual mode (it stops moving on its own); 30 s after the last touch,
 * or on leaving the page, Dobby gets the head back.
 */
function CameraStick() {
  const [stick, setStick] = useState({ x: 0, y: 0 });
  const push = useRef(stick);
  const hold = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const handBack = () => {
    clearTimeout(hold.current);
    if (hold.current !== undefined && robotLink.state().mode === "manual") robotLink.setManual(false);
    hold.current = undefined;
  };
  useEffect(() => handBack, []);

  const pushed = Math.hypot(stick.x, stick.y) > DEAD_ZONE;
  useEffect(() => {
    if (!pushed) return;
    let frame = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      const t = robotLink.target();
      robotLink.setTarget(steer(t, push.current.x, push.current.y, dt));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [pushed]);

  const move = (x: number, y: number) => {
    push.current = { x, y };
    setStick(push.current);
    clearTimeout(hold.current);
    hold.current = setTimeout(handBack, CAMERA_HOLD_MS);
  };
  return (
    <div className="absolute bottom-3 right-3">
      <Joystick
        label="Mover a câmera"
        size="sm"
        showLabel={false}
        className="bg-white/60 backdrop-blur-sm"
        x={stick.x}
        y={stick.y}
        onChange={move}
        onRelease={() => move(0, 0)}
      />
    </div>
  );
}

/** The live camera with detection overlays; the stream only runs while someone can see it. */
export function CameraCard({ className }: { className?: string }) {
  const status = useLive((s) => s.status);
  const open = useLive((s) => s.connection === "open");
  const onHome = useLocation().pathname === "/";
  const visible = usePageVisible();
  const frame = useRef<HTMLDivElement>(null);
  const width = useStreamWidth(frame);
  const [overlays, setOverlays] = useState(() => loadPref(OVERLAYS_KEY, { people: true, voice: true, tracking: true }));
  const [failedAt, setFailedAt] = useState(0);
  const [live, setLive] = useState(false);

  const camera = !!status?.camera;
  const enabled = camera && status?.video_enabled !== false;
  const wanted = open && onHome && visible && enabled && failedAt === 0;
  const src = wanted ? videoUrl({ ...overlays, width }) : undefined;

  useEffect(() => {
    if (!failedAt) return;
    const timer = setTimeout(() => setFailedAt(0), RETRY_MS);
    return () => clearTimeout(timer);
  }, [failedAt]);

  const toggle = (key: keyof typeof overlays) => {
    const next = { ...overlays, [key]: !overlays[key] };
    setOverlays(next);
    savePref(OVERLAYS_KEY, next);
  };

  const message = !status ? "" : !camera ? "Sem câmera" : !enabled ? "Câmera desligada" : "Aguardando imagem…";
  const showImage = !!src && live;
  const steerable = showImage && !status?.sleeping;

  return (
    <Card className={cx("relative overflow-hidden", className)}>
      <CardHeader
        className="pr-1"
        end={
          <div className="flex items-center gap-2">
            {showImage && (
              <span className="label flex items-center gap-1.5 text-text">
                <span className="size-1.5 rounded-full bg-accent" />
                Ao vivo
              </span>
            )}
            <Menu
              trigger={
                <button aria-label="Opções da câmera" className="grid size-8 place-items-center text-dim hover:bg-subtle hover:text-text">
                  <Ellipsis className="size-4" />
                </button>
              }
              items={[
                { label: "Pessoas", checked: overlays.people, onSelect: () => toggle("people") },
                { label: "Voz", checked: overlays.voice, onSelect: () => toggle("voice") },
                { label: "Rastreio", checked: overlays.tracking, onSelect: () => toggle("tracking") },
              ]}
            />
          </div>
        }
      >
        Câmera
      </CardHeader>
      <div ref={frame} className="relative aspect-video w-full bg-[#ededed]">
        {src && (
          <img
            key={src}
            src={src}
            alt="Câmera do Dobby"
            onLoad={() => setLive(true)}
            onError={() => {
              setLive(false);
              setFailedAt(Date.now());
            }}
            className={cx("absolute inset-0 size-full object-cover transition-opacity duration-300", showImage ? "opacity-100" : "opacity-0")}
          />
        )}
        <AnimatePresence>
          {!showImage && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={fade} className="absolute inset-0 grid place-items-center font-mono text-xs uppercase tracking-[0.08em] text-dim">
              <span className="flex items-center gap-2">
                {!enabled && status && <CameraOff className="size-4" />}
                {message}
              </span>
            </motion.div>
          )}
        </AnimatePresence>
        {steerable && <CameraStick />}
      </div>
    </Card>
  );
}
