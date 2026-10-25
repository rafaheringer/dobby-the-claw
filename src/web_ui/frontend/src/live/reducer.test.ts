import { describe, expect, it } from "vitest";
import type { ChatEntry, LogLine, Status } from "./protocol";
import { initialLive, MAX_CHAT, MAX_LOGS, MAX_TRIES, onConnection, reduce } from "./reducer";

const status: Status = {
  state: "IDLE", sleeping: false, transition: "", model: "gemini · x", people: [], speaker: null,
  camera: true, head_tracking: false, video_enabled: true, fps: 8, mic: { rms: 0.01 },
  wake: { heard: null }, services: {}, log_level: "INFO",
};
const line = (id: number): LogLine => ({ id, t: id, lv: 20, cat: "Sistema", msg: `m${id}` });
const chat = (n: number): ChatEntry => ({ t: n, who: n % 2 ? "you" : "dobby", text: `c${n}` });

describe("reduce", () => {
  it("hello replaces logs, chat, status and categories", () => {
    const before = reduce(initialLive, { type: "logs", lines: [line(1)] });
    const s = reduce(before, { type: "hello", categories: ["Visão"], logs: [line(7)], chat: [chat(1)], status, robot: null });
    expect(s.logs).toEqual([line(7)]);
    expect(s.chat).toEqual([chat(1)]);
    expect(s.status).toEqual(status);
    expect(s.categories).toEqual(["Visão"]);
  });

  it("logs append and cap at MAX_LOGS", () => {
    const lines = Array.from({ length: 4500 }, (_, i) => line(i + 1));
    const s = reduce(initialLive, { type: "logs", lines });
    expect(s.logs).toHaveLength(MAX_LOGS);
    expect(s.logs.at(-1)!.id).toBe(4500);
  });

  it("chat appends and caps at MAX_CHAT", () => {
    let s = initialLive;
    for (let i = 0; i < MAX_CHAT + 20; i++) s = reduce(s, { type: "chat", entry: chat(i) });
    expect(s.chat).toHaveLength(MAX_CHAT);
    expect(s.chat.at(-1)!.text).toBe(`c${MAX_CHAT + 19}`);
  });

  it("status message replaces status without the type field", () => {
    const s = reduce(initialLive, { type: "status", ...status, state: "LISTENING" });
    expect(s.status).toEqual({ ...status, state: "LISTENING" });
    expect(s.status).not.toHaveProperty("type");
  });

  it("wake_state recording true sets recording and seconds; thinking true clears recording", () => {
    const rec = reduce(initialLive, { type: "wake_state", recording: true, seconds: 3 });
    expect(rec.wakeTest).toMatchObject({ recording: true, thinking: false, seconds: 3 });
    const thinking = reduce(rec, { type: "wake_state", recording: false, thinking: true });
    expect(thinking.wakeTest).toMatchObject({ recording: false, thinking: true });
  });

  it("wake_result prepends a try, keeps MAX_TRIES and clears recording/thinking", () => {
    let s = reduce(initialLive, { type: "wake_state", recording: false, thinking: true });
    for (let i = 0; i < 7; i++) s = reduce(s, { type: "wake_result", text: `t${i}`, hit: i % 2 === 0, silent: false }, i);
    expect(s.wakeTest.tries).toHaveLength(MAX_TRIES);
    expect(s.wakeTest.tries[0]).toEqual({ text: "t6", hit: true, silent: false, t: 6 });
    expect(s.wakeTest).toMatchObject({ recording: false, thinking: false });
  });

  it("error pushes a toast with the server text and an increasing id", () => {
    const a = reduce(initialLive, { type: "error", text: "mensagem longa demais" });
    const b = reduce(a, { type: "error", text: "outra" });
    expect(b.toasts.map((t) => t.text)).toEqual(["mensagem longa demais", "outra"]);
    expect(b.toasts[1].id).toBeGreaterThan(b.toasts[0].id);
  });

  it("an error ends a wake test in progress (the server refused it)", () => {
    const rec = reduce(initialLive, { type: "wake_state", recording: true, seconds: 3 });
    const s = reduce(rec, { type: "error", text: "O detector de nome está desligado (OFFLINE_WAKEWORD_ENABLED=0)." });
    expect(s.wakeTest).toMatchObject({ recording: false, thinking: false });
  });

  it("onConnection closed clears wake recording and thinking", () => {
    const rec = reduce(initialLive, { type: "wake_state", recording: true, seconds: 3 });
    const s = onConnection(rec, "closed");
    expect(s.connection).toBe("closed");
    expect(s.wakeTest).toMatchObject({ recording: false, thinking: false });
  });

  it("status with camera false and mic/wake null is stored as is", () => {
    const off = { ...status, camera: false, mic: null, wake: null };
    expect(reduce(initialLive, { type: "status", ...off }).status).toEqual(off);
  });
});
