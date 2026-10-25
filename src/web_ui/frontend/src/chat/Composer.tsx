import { ArrowUp } from "lucide-react";
import { useEffect, useState } from "react";
import { WakeToggle } from "../app/WakeToggle";
import { send, useLive } from "../live/store";
import { Button } from "../ui/Button";
import { keyboardInset } from "./keyboard";

function useKeyboardInset(): number {
  const [inset, setInset] = useState(0);
  useEffect(() => {
    const vv = window.visualViewport;
    if (!vv) return;
    const update = () => setInset(keyboardInset(window.innerHeight, vv.height, vv.offsetTop));
    update();
    vv.addEventListener("resize", update);
    vv.addEventListener("scroll", update);
    return () => {
      vv.removeEventListener("resize", update);
      vv.removeEventListener("scroll", update);
    };
  }, []);
  return inset;
}

/** The message box, pinned above the tab bar (or the keyboard) on phones. */
export function Composer() {
  const [text, setText] = useState("");
  const asleep = useLive((s) => !!s.status?.sleeping);
  const open = useLive((s) => s.connection === "open");
  const keyboard = useKeyboardInset();

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const value = text.trim();
    if (!value) return;
    send({ type: "chat", text: value });
    setText("");
  };

  return (
    <div
      className="fixed inset-x-3 bottom-(--composer-bottom) z-30 lg:sticky lg:inset-x-auto lg:bottom-4 lg:mt-auto lg:pt-4"
      style={{ "--composer-bottom": keyboard > 0 ? `${keyboard + 8}px` : "calc(env(safe-area-inset-bottom) + 68px)" } as React.CSSProperties}
    >
      {asleep ? (
        <div className="flex max-w-3xl items-center justify-between gap-3 border border-ink bg-surface py-1.5 pl-4 pr-1.5">
          <span className="label">Dobby está dormindo</span>
          <WakeToggle />
        </div>
      ) : (
        <form onSubmit={submit} className="flex max-w-3xl items-center gap-2 border border-ink bg-surface p-1.5 pl-4 focus-within:shadow-[4px_4px_0_var(--accent)]">
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Mensagem"
            maxLength={2000}
            enterKeyHint="send"
            autoComplete="off"
            className="min-w-0 flex-1 bg-transparent text-[16px] outline-none placeholder:text-dim"
          />
          <Button type="submit" variant="primary" icon aria-label="Enviar" disabled={!open || !text.trim()}>
            <ArrowUp className="size-5" />
          </Button>
        </form>
      )}
    </div>
  );
}
