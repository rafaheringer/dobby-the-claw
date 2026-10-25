import { create } from "zustand";
import type { ClientMessage } from "./protocol";
import { initialLive, onConnection, reduce, withToast, type LiveState } from "./reducer";
import { connectLive } from "./socket";

export const useLive = create<LiveState>(() => initialLive);

let link: ReturnType<typeof connectLive> | null = null;

export function pushToast(text: string): void {
  useLive.setState((s) => withToast(s, text));
}

export function dismissToast(id: number): void {
  useLive.setState((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }));
}

export function send(m: ClientMessage): void {
  if (!link?.send(m)) pushToast("Sem conexão com o Dobby");
}

export function startLive(): void {
  if (link) return;
  const url = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;
  link = connectLive(url, {
    onMessage: (m) => useLive.setState((s) => reduce(s, m)),
    onConnection: (c) => useLive.setState((s) => onConnection(s, c)),
  });
}
