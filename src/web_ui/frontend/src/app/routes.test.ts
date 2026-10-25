import { describe, expect, it } from "vitest";
import { sectionIndex, SECTIONS } from "./routes";

describe("sections", () => {
  it("lists the four tabs in order", () => {
    expect(SECTIONS.map((s) => s.path)).toEqual(["/", "/conversa", "/logs", "/sistema"]);
  });

  it("maps a path to its tab", () => {
    expect(sectionIndex("/")).toBe(0);
    expect(sectionIndex("/conversa")).toBe(1);
    expect(sectionIndex("/logs")).toBe(2);
    expect(sectionIndex("/sistema")).toBe(3);
    expect(sectionIndex("/sistema/nome")).toBe(3);
    expect(sectionIndex("/twin")).toBe(-1);
    expect(sectionIndex("/nada")).toBe(-1);
  });
});
