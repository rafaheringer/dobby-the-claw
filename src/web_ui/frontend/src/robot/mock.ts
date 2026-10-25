import { create } from "zustand";
import { sampleEmotion } from "./emotions";
import { addTargets, clampTarget, idleOffset, poseMatrix, ZERO_TARGET } from "./pose";
import type { EmotionName, LinkState, RobotLink, RobotPose, RobotTarget } from "./types";

/**
 * A robot that lives only in the page: idle motion in auto mode, the controls' target in manual
 * mode, and short emotions on top. It sends nothing to the real robot.
 */
export function createMockRobotLink(opts: { now?: () => number } = {}): RobotLink & { onSpeaker(name: string | null): void } {
  const now = opts.now ?? (() => performance.now());
  const store = create<{ link: LinkState; target: RobotTarget }>(() => ({
    link: { mode: "auto", simulated: true, emotion: null },
    target: ZERO_TARGET,
  }));
  let emotionStart = 0;
  let lastSpeaker: string | null = null;

  const setLink = (patch: Partial<LinkState>) => store.setState((s) => ({ link: { ...s.link, ...patch } }));

  const play = (name: EmotionName | "atento") => {
    emotionStart = now();
    setLink({ emotion: name });
  };

  return {
    tick(): RobotPose {
      const { link, target } = store.getState();
      const t = now() / 1000;
      let pose = link.mode === "auto" ? addTargets(ZERO_TARGET, idleOffset(t)) : target;
      if (link.emotion) {
        const offset = sampleEmotion(link.emotion, (now() - emotionStart) / 1000);
        if (offset) pose = addTargets(pose, offset);
        else setLink({ emotion: null });
      }
      pose = clampTarget(pose);
      return { headPose: poseMatrix(pose), antennas: pose.antennas, yawBody: pose.yawBody };
    },
    state: () => store.getState().link,
    target: () => store.getState().target,
    useLinkState: () => store((s) => s.link),
    useTarget: () => store((s) => s.target),
    setManual(on) {
      setLink({ mode: on ? "manual" : "auto" });
    },
    setTarget(t) {
      store.setState((s) => ({ target: clampTarget({ ...s.target, ...t }), link: { ...s.link, mode: "manual" } }));
    },
    center() {
      store.setState((s) => ({ target: ZERO_TARGET, link: { ...s.link, mode: "manual" } }));
    },
    playEmotion: play,
    onSpeaker(name) {
      if (name && name !== lastSpeaker) play("atento");
      lastSpeaker = name;
    },
  };
}
