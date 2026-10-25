import { useLive } from "../live/store";
import { createMockRobotLink } from "./mock";
import type { RobotLink } from "./types";

// The one place that picks the robot link: swap the mock for the real one here.
const mock = createMockRobotLink();
useLive.subscribe((s, prev) => {
  if (s.status?.speaker !== prev.status?.speaker) mock.onSpeaker(s.status?.speaker ?? null);
});

export const robotLink: RobotLink = mock;
export type { EmotionName, LinkState, RobotLink, RobotPose, RobotTarget } from "./types";
export { LIMITS, ZERO_TARGET } from "./pose";
