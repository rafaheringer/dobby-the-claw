import { CameraOff, Ellipsis } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router";
import { fade } from "../app/motion";
import { useLive } from "../live/store";
import { loadPref, savePref } from "../lib/storage";
import { usePageVisible } from "../lib/usePageVisible";
import { streamWidth, videoUrl } from "../lib/video";
import { Card, cx } from "../ui/Glass";
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

  return (
    <Card className={cx("relative overflow-hidden", className)}>
      <div ref={frame} className="relative aspect-video w-full bg-black/30">
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
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={fade} className="absolute inset-0 grid place-items-center text-sm text-dim">
              <span className="flex items-center gap-2">
                {!enabled && status && <CameraOff className="size-4" />}
                {message}
              </span>
            </motion.div>
          )}
        </AnimatePresence>
        <div className="absolute right-2 top-2">
          <Menu
            trigger={
              <button aria-label="Opções da câmera" className="glass grid size-9 place-items-center rounded-full bg-black/30">
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
      </div>
    </Card>
  );
}
