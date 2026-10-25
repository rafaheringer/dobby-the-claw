import type { ChatEntry, LogLine, ServerMessage, Status } from "./protocol";

export const MAX_LOGS = 4000;
export const MAX_CHAT = 500;
export const MAX_TRIES = 6;

export interface WakeTry {
  text: string;
  hit: boolean;
  silent: boolean;
  t: number;
}

export interface LiveState {
  connection: "connecting" | "open" | "closed";
  status: Status | null;
  categories: string[];
  logs: LogLine[];
  chat: ChatEntry[];
  wakeTest: { recording: boolean; thinking: boolean; seconds: number; tries: WakeTry[] };
  toasts: { id: number; text: string }[];
}

export const initialLive: LiveState = {
  connection: "connecting",
  status: null,
  categories: [],
  logs: [],
  chat: [],
  wakeTest: { recording: false, thinking: false, seconds: 0, tries: [] },
  toasts: [],
};

let toastSeq = 0;

export function withToast(s: LiveState, text: string): LiveState {
  return { ...s, toasts: [...s.toasts, { id: ++toastSeq, text }] };
}

const tail = <T,>(items: T[], max: number): T[] => (items.length > max ? items.slice(-max) : items);

/** Apply one server message; pure, never mutates `s`. */
export function reduce(s: LiveState, m: ServerMessage, now: number = Date.now() / 1000): LiveState {
  switch (m.type) {
    case "hello":
      return { ...s, categories: m.categories, logs: tail(m.logs, MAX_LOGS), chat: tail(m.chat, MAX_CHAT), status: m.status };
    case "status": {
      const { type: _type, ...status } = m;
      return { ...s, status };
    }
    case "logs":
      return { ...s, logs: tail([...s.logs, ...m.lines], MAX_LOGS) };
    case "chat":
      return { ...s, chat: tail([...s.chat, m.entry], MAX_CHAT) };
    case "wake_state":
      return {
        ...s,
        wakeTest: { ...s.wakeTest, recording: m.recording, thinking: !!m.thinking, seconds: m.seconds ?? s.wakeTest.seconds },
      };
    case "wake_result": {
      const entry = { text: m.text, hit: m.hit, silent: m.silent, t: now };
      return { ...s, wakeTest: { ...s.wakeTest, recording: false, thinking: false, tries: [entry, ...s.wakeTest.tries].slice(0, MAX_TRIES) } };
    }
    case "error": {
      // The server answers a refused wake test (detector off, test already running) with an error.
      const wakeTest = { ...s.wakeTest, recording: false, thinking: false };
      return withToast({ ...s, wakeTest }, m.text);
    }
  }
}

/** Socket state changed; a dropped socket also ends any wake test in progress. */
export function onConnection(s: LiveState, connection: LiveState["connection"]): LiveState {
  const wakeTest = connection === "open" ? s.wakeTest : { ...s.wakeTest, recording: false, thinking: false };
  return { ...s, connection, wakeTest };
}
