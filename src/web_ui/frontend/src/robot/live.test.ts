import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ClientMessage, RobotSnapshot } from "../live/protocol";
import { createLiveRobotLink, STALE_MS } from "./live";
import { createMockRobotLink } from "./mock";
import { poseMatrix, ZERO_TARGET } from "./pose";
import type { RobotTarget } from "./types";

const snap = (pose: Partial<RobotTarget>, mode: "auto" | "manual" = "auto", target: RobotTarget | null = null): RobotSnapshot => ({
  type: "robot",
  mode,
  pose: { ...ZERO_TARGET, ...pose },
  target,
});

let clock = 0;
let sent: ClientMessage[] = [];

function make() {
  const fallback = createMockRobotLink({ now: () => clock });
  const link = createLiveRobotLink({ fallback, send: (m) => (sent.push(m), true), now: () => clock });
  return { link, fallback };
}

beforeEach(() => {
  clock = 1000;
  sent = [];
  vi.useFakeTimers();
});
afterEach(() => vi.useRealTimers());

describe("live robot link", () => {
  it("is the simulation until the bridge reports a pose", () => {
    const { link } = make();
    expect(link.state().simulated).toBe(true);
    link.setTarget({ yaw: 0.3 });
    expect(sent).toEqual([]); // nothing goes to a robot that is not there
  });

  it("draws the robot's real pose once it streams", () => {
    const { link } = make();
    link.onSnapshot(snap({ yaw: 0.4, antennas: [0.5, -0.5] }));
    expect(link.state().simulated).toBe(false);
    for (let i = 0; i < 60; i++) {
      clock += 16;
      link.tick();
    }
    const pose = link.tick();
    expect(pose.headPose[0]).toBeCloseTo(poseMatrix({ ...ZERO_TARGET, yaw: 0.4 })[0], 3);
    expect(pose.antennas[0]).toBeCloseTo(0.5, 3);
  });

  it("falls back to the simulation when the stream stops", () => {
    const { link } = make();
    link.onSnapshot(snap({}));
    clock += STALE_MS + 1;
    vi.advanceTimersByTime(STALE_MS + 1);
    expect(link.state().simulated).toBe(true);
  });

  it("goes simulated at once when the bridge says it cannot report the robot", () => {
    const { link } = make();
    link.onSnapshot(snap({}));
    link.onSnapshot(null);
    expect(link.state().simulated).toBe(true);
  });

  it("sends the controls' target to the robot, clamped, and shows manual mode", () => {
    const { link } = make();
    link.onSnapshot(snap({}));
    link.setTarget({ yaw: 5 });
    expect(sent.at(-1)).toEqual({ type: "robot_target", target: { ...ZERO_TARGET, yaw: 1.1 } });
    expect(link.state().mode).toBe("manual");
  });

  it("sends at most ~30 targets a second while dragging, ending on the last one", () => {
    const { link } = make();
    link.onSnapshot(snap({}));
    for (let i = 1; i <= 10; i++) link.setTarget({ yaw: i / 100 });
    expect(sent.filter((m) => m.type === "robot_target")).toHaveLength(1);
    vi.advanceTimersByTime(40);
    const targets = sent.filter((m) => m.type === "robot_target");
    expect(targets).toHaveLength(2);
    expect(targets.at(-1)).toMatchObject({ target: { yaw: 0.1 } });
  });

  it("mirrors the robot in the controls while in auto mode", () => {
    const { link } = make();
    link.onSnapshot(snap({ pitch: 0.2 }));
    expect(link.target().pitch).toBeCloseTo(0.2);
  });

  it("keeps the page's own edits while the person is dragging", () => {
    const { link } = make();
    link.onSnapshot(snap({}));
    link.setTarget({ yaw: 0.5 });
    link.onSnapshot(snap({ yaw: 0.1 }, "manual", { ...ZERO_TARGET, yaw: 0.2 }));
    expect(link.target().yaw).toBeCloseTo(0.5);
  });

  it("switches modes on the bridge", () => {
    const { link } = make();
    link.onSnapshot(snap({}));
    link.setManual(true);
    link.setManual(false);
    expect(sent).toEqual([
      { type: "robot_mode", manual: true },
      { type: "robot_mode", manual: false },
    ]);
  });

  it("asks the robot for an emotion", () => {
    const { link } = make();
    link.onSnapshot(snap({}));
    link.playEmotion("feliz");
    expect(sent).toEqual([{ type: "robot_emotion", name: "feliz" }]);
  });
});
