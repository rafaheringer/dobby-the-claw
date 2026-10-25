import { describe, expect, it, vi } from "vitest";
import { closeTwin } from "./close";

describe("closeTwin", () => {
  it("goes back when the app has history", () => {
    const navigate = vi.fn();
    closeTwin(navigate, 3);
    expect(navigate).toHaveBeenCalledWith(-1);
  });

  it("goes home when /twin was opened directly", () => {
    for (const idx of [0, undefined]) {
      const navigate = vi.fn();
      closeTwin(navigate, idx);
      expect(navigate).toHaveBeenCalledWith("/", { replace: true });
    }
  });
});
