import type { ClientMessage, ServerMessage } from "./protocol";

export function backoffMs(retry: number): number {
  return Math.min(10000, 800 * 2 ** retry);
}

export interface LiveHandlers {
  onMessage(m: ServerMessage): void;
  onConnection(c: "connecting" | "open" | "closed"): void;
}

/** One WebSocket to the bridge that reconnects by itself until close() is called. */
export function connectLive(url: string, handlers: LiveHandlers): { send(m: ClientMessage): boolean; close(): void } {
  let ws: WebSocket | null = null;
  let retry = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let closed = false;

  const open = () => {
    handlers.onConnection("connecting");
    ws = new WebSocket(url);
    ws.onopen = () => {
      retry = 0;
      handlers.onConnection("open");
    };
    ws.onmessage = (event) => {
      try {
        handlers.onMessage(JSON.parse(event.data as string) as ServerMessage);
      } catch {
        // a malformed frame is dropped; the next status tick resyncs the page
      }
    };
    ws.onclose = () => {
      handlers.onConnection("closed");
      if (!closed) timer = setTimeout(open, backoffMs(retry++));
    };
  };
  open();

  return {
    send(m) {
      if (ws?.readyState !== WebSocket.OPEN) return false;
      ws.send(JSON.stringify(m));
      return true;
    },
    close() {
      closed = true;
      clearTimeout(timer);
      ws?.close();
    },
  };
}
