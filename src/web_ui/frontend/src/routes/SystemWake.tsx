import { Check, ChevronLeft, Mic, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect } from "react";
import { Link } from "react-router";
import { useMotionPreset } from "../app/motion";
import { pushToast, send, useLive } from "../live/store";
import { CountdownRing } from "../system/CountdownRing";
import { Button } from "../ui/Button";
import { Card, cx } from "../ui/Glass";
import { PageTitle } from "../ui/PageTitle";

const GUARD_EXTRA_S = 15;

function tryText(t: { text: string; hit: boolean; silent: boolean }): string {
  if (t.silent) return "Não ouvi nada. Chegue mais perto ou fale mais alto.";
  return t.hit ? `Ouvi “${t.text}”. Eu acordaria.` : `Ouvi “${t.text || "…"}”. Não acordaria.`;
}

/** If the server never answers (dropped socket), don't stay stuck recording. */
function useStuckGuard(active: boolean, seconds: number) {
  useEffect(() => {
    if (!active) return;
    const timer = setTimeout(() => {
      useLive.setState((s) => ({ wakeTest: { ...s.wakeTest, recording: false, thinking: false } }));
      pushToast("O Dobby não respondeu ao teste. Tente de novo.");
    }, (seconds + GUARD_EXTRA_S) * 1000);
    return () => clearTimeout(timer);
  }, [active, seconds]);
}

export default function SystemWake() {
  const status = useLive((s) => s.status);
  const open = useLive((s) => s.connection === "open");
  const test = useLive((s) => s.wakeTest);
  const { spring } = useMotionPreset();
  const busy = test.recording || test.thinking;
  useStuckGuard(busy, test.seconds);

  const start = () => {
    useLive.setState((s) => ({ wakeTest: { ...s.wakeTest, recording: true, thinking: false, seconds: 3 } })); // instant feedback
    send({ type: "wake_test" });
  };
  const awake = status && !status.sleeping && !status.transition;

  return (
    <div className="max-w-xl">
      <Link to="/sistema" className="mb-1 inline-flex items-center gap-1 pt-2 text-sm text-dim hover:text-text">
        <ChevronLeft className="size-4" /> Sistema
      </Link>
      <PageTitle>Teste do nome</PageTitle>

      <AnimatePresence>
        {awake && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} transition={spring}>
            <Card className="mb-4 flex items-center justify-between gap-3 p-3 pl-4 text-sm">
              <span className="text-dim">Ele está acordado e pode responder.</span>
              <Button disabled={!open} onClick={() => send({ type: "cmd", name: "sleep" })}>
                Colocar para dormir
              </Button>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      <Card className="flex flex-col items-center gap-5 px-6 py-8">
        <div className="relative size-40">
          {busy && (
            <div className="absolute inset-0">
              <CountdownRing key={test.thinking ? "t" : "r"} seconds={test.seconds || 3} thinking={test.thinking} />
            </div>
          )}
          <motion.button
            whileTap={{ scale: 0.94 }}
            transition={spring}
            disabled={busy || !open}
            onClick={start}
            aria-label="Testar"
            className="absolute inset-4 grid place-items-center rounded-full bg-accent text-accent-ink shadow-[0_0_40px_rgba(154,168,255,.35)] disabled:opacity-60"
          >
            <Mic className="size-10" />
          </motion.button>
        </div>
        <p className="h-6 text-center font-medium">
          {test.recording ? "Diga “Dobby”" : test.thinking ? "Entendendo…" : "Toque e diga “Dobby”"}
        </p>
        {status?.wake && (
          <p className="text-center text-sm text-dim">
            Ouvido enquanto dorme: {status.wake.heard ? `“${status.wake.heard}”` : "nada nos últimos segundos"}
          </p>
        )}
      </Card>

      <ul className="mt-4 flex flex-col gap-2">
        <AnimatePresence initial={false}>
          {test.tries.map((t) => (
            <motion.li key={t.t} layout initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={spring}>
              <Card className="flex items-center gap-3 px-4 py-3 text-sm">
                <span className={cx("grid size-7 shrink-0 place-items-center rounded-full", t.hit ? "bg-st-listening/20 text-st-listening" : "bg-st-error/20 text-st-error")}>
                  {t.hit ? <Check className="size-4" /> : <X className="size-4" />}
                </span>
                {tryText(t)}
              </Card>
            </motion.li>
          ))}
        </AnimatePresence>
      </ul>
    </div>
  );
}
