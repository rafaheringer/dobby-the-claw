import { create } from "zustand";
import type { AudioSample, ClientMessage, RobotSnapshot } from "./protocol";
import { initialLive, onConnection, reduce, withToast, type LiveState } from "./reducer";
import { connectLive } from "./socket";

export const useLive = create<LiveState>(() => initialLive);

/** The latest mic level and sound direction (null until the bridge sends one). */
export const useAudio = create<{ sample: AudioSample | null }>(() => ({ sample: null }));

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

/** Like send(), without the toast: for the twin's control stream, which falls back on its own. */
export function trySend(m: ClientMessage): boolean {
  return !!link?.send(m);
}

const robotListeners = new Set<(r: RobotSnapshot | null) => void>();

/** The robot's pose stream (null: this bridge cannot report it). Returns the unsubscribe. */
export function onRobot(fn: (r: RobotSnapshot | null) => void): () => void {
  robotListeners.add(fn);
  return () => robotListeners.delete(fn);
}

export function startLive(): void {
  if (link) return;
  const url = `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/ws`;
  link = connectLive(url, {
    onMessage: (m) => {
      if (m.type === "robot" || m.type === "hello") robotListeners.forEach((fn) => fn(m.type === "robot" ? m : m.robot));
      if (m.type === "audio") useAudio.setState({ sample: m });
      else if (m.type !== "robot") useLive.setState((s) => reduce(s, m));
    },
    onConnection: (c) => {
      if (c !== "open") useAudio.setState({ sample: null });
      useLive.setState((s) => onConnection(s, c));
    },
  });
}
