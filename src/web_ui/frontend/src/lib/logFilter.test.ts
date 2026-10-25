import { describe, expect, it } from "vitest";
import type { LogLine } from "../live/protocol";
import { matchLog, parseLogParams, toLogParams, type LogFilter } from "./logFilter";

const l = (msg: string, lv = 20, cat = "Visão"): LogLine => ({ id: 1, t: 0, lv, cat, msg });
const all: LogFilter = { q: "", level: 20, cats: null };

describe("matchLog", () => {
  it("hides lines below the level", () => {
    expect(matchLog(all)(l("x", 10))).toBe(false);
    expect(matchLog(all)(l("x", 30))).toBe(true);
  });

  it("hides categories that are not enabled; null shows all", () => {
    expect(matchLog({ ...all, cats: ["Áudio"] })(l("x"))).toBe(false);
    expect(matchLog({ ...all, cats: ["Visão"] })(l("x"))).toBe(true);
    expect(matchLog(all)(l("x", 20, "Outros"))).toBe(true);
  });

  it("keeps words and hides !words, ignoring case", () => {
    const m = matchLog({ ...all, q: "camera !erro" });
    expect(m(l("Camera ready"))).toBe(true);
    expect(m(l("camera ERRO de frame"))).toBe(false);
    expect(m(l("mic ok"))).toBe(false);
  });

  it("ignores a lone !", () => {
    expect(matchLog({ ...all, q: "!" })(l("anything"))).toBe(true);
  });
});

describe("log URL params", () => {
  it("round trips q, nivel and cat", () => {
    const sp = new URLSearchParams("q=a&nivel=erro&cat=Visão,Áudio");
    const f = parseLogParams(sp);
    expect(f).toEqual({ q: "a", level: 40, cats: ["Visão", "Áudio"] });
    const back = toLogParams(f);
    expect(back.get("q")).toBe("a");
    expect(back.get("nivel")).toBe("erro");
    expect(back.get("cat")).toBe("Visão,Áudio");
  });

  it("defaults produce an empty query string", () => {
    expect(toLogParams(all).toString()).toBe("");
  });

  it("unknown nivel falls back to info", () => {
    expect(parseLogParams(new URLSearchParams("nivel=xyz")).level).toBe(20);
  });
});
