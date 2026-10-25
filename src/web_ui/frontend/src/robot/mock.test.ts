import { describe, expect, it } from "vitest";
import { EMOTIONS } from "./emotions";
import { createMockRobotLink } from "./mock";
import { poseMatrix, ZERO_TARGET } from "./pose";

function clock() {
  let t = 0;
  return { now: () => t, advance: (s: number) => (t += s * 1000) };
}

describe("MockRobotLink", () => {
  it("moves by itself in auto mode", () => {
    const c = clock();
    const link = createMockRobotLink({ now: c.now });
    const a = link.tick().headPose;
    c.advance(1.3);
    expect(link.tick().headPose).not.toEqual(a);
    expect(link.state().mode).toBe("auto");
    expect(link.state().simulated).toBe(true);
  });

  it("setTarget switches to manual and clamps", () => {
    const link = createMockRobotLink({ now: clock().now });
    link.setTarget({ yaw: 5 });
    expect(link.state().mode).toBe("manual");
    expect(link.target().yaw).toBe(1.1);
  });

  it("center returns to rest and stays manual; setManual(false) goes back to auto", () => {
    const link = createMockRobotLink({ now: clock().now });
    link.setTarget({ pitch: 0.3 });
    link.center();
    expect(link.target()).toEqual(ZERO_TARGET);
    expect(link.state().mode).toBe("manual");
    expect(link.tick().headPose).toEqual(poseMatrix(ZERO_TARGET));
    link.setManual(false);
    expect(link.state().mode).toBe("auto");
  });

  it("plays 'atento' once per new speaker", () => {
    const c = clock();
    const link = createMockRobotLink({ now: c.now });
    link.onSpeaker("Ana");
    expect(link.state().emotion).toBe("atento");
    c.advance(EMOTIONS.atento.duration + 0.1);
    link.tick();
    expect(link.state().emotion).toBeNull();
    link.onSpeaker("Ana");
    expect(link.state().emotion).toBeNull();
    link.onSpeaker(null);
    link.onSpeaker("Ana");
    expect(link.state().emotion).toBe("atento");
  });

  it("an emotion overrides the pose for its duration, then the base pose returns", () => {
    const c = clock();
    const link = createMockRobotLink({ now: c.now });
    link.center();
    const rest = link.tick();
    link.playEmotion("feliz");
    c.advance(EMOTIONS.feliz.duration / 2);
    expect(link.tick()).not.toEqual(rest);
    c.advance(EMOTIONS.feliz.duration);
    expect(link.tick()).toEqual(rest);
  });
});
