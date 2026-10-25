import { onRobot, trySend, useLive } from "../live/store";
import { createLiveRobotLink } from "./live";
import { createMockRobotLink } from "./mock";
import type { RobotLink } from "./types";

// The real robot through the bridge; the in-page simulation stands in while the bridge cannot
// stream it (offline, or running without the Reachy SDK).
const mock = createMockRobotLink();
useLive.subscribe((s, prev) => {
  if (s.status?.speaker !== prev.status?.speaker) mock.onSpeaker(s.status?.speaker ?? null);
});
const live = createLiveRobotLink({ fallback: mock, send: trySend });
onRobot((r) => live.onSnapshot(r));

export const robotLink: RobotLink = live;
export type { EmotionName, LinkState, RobotLink, RobotPose, RobotTarget } from "./types";
export { LIMITS, ZERO_TARGET } from "./pose";
