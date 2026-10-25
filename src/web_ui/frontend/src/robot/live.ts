import { create } from "zustand";
import type { ClientMessage, RobotSnapshot } from "../live/protocol";
import { clampTarget, poseMatrix, ZERO_TARGET } from "./pose";
import type { EmotionName, LinkState, RobotLink, RobotPose, RobotTarget } from "./types";

export const STALE_MS = 2000; // no pose for this long: the robot is not reachable, show the simulation
const SEND_EVERY_MS = 33; // at most ~30 targets a second while dragging
const EDIT_HOLD_MS = 1000; // the page's own edits win over the stream for this long
const EASE_MS = 80; // smooths the 15 Hz stream into the 60 fps drawing

const blend = (a: number, b: number, u: number) => a + (b - a) * u;

function blendTarget(a: RobotTarget, b: RobotTarget, u: number): RobotTarget {
  return {
    x: blend(a.x, b.x, u),
    y: blend(a.y, b.y, u),
    z: blend(a.z, b.z, u),
    roll: blend(a.roll, b.roll, u),
    pitch: blend(a.pitch, b.pitch, u),
    yaw: blend(a.yaw, b.yaw, u),
    antennas: [blend(a.antennas[0], b.antennas[0], u), blend(a.antennas[1], b.antennas[1], u)],
    yawBody: blend(a.yawBody, b.yawBody, u),
  };
}

interface Deps {
  /** Used whenever the bridge does not stream the robot (bridge down, no Reachy SDK). */
  fallback: RobotLink;
  send(m: ClientMessage): boolean;
  now?: () => number;
}

/**
 * The real robot behind the twin: draws the pose the bridge streams and sends the controls to it.
 * While no pose arrives it hands everything to `fallback`, the in-page simulation.
 */
export function createLiveRobotLink({ fallback, send, now = () => performance.now() }: Deps): RobotLink & {
  onSnapshot(r: RobotSnapshot | null): void;
} {
  const store = create<{ link: LinkState; target: RobotTarget }>(() => ({
    link: { mode: "auto", simulated: true, emotion: null },
    target: ZERO_TARGET,
  }));
  let latest: RobotTarget | null = null;
  let shown: RobotTarget | null = null;
  let receivedAt = -Infinity;
  let lastTick = now();
  let lastEdit = -Infinity;
  let lastSent = -Infinity;
  let trailing: ReturnType<typeof setTimeout> | undefined;

  const live = () => !store.getState().link.simulated;
  const setLink = (patch: Partial<LinkState>) => store.setState((s) => ({ link: { ...s.link, ...patch } }));

  const goSimulated = () => {
    latest = shown = null;
    clearTimeout(trailing);
    trailing = undefined;
    setLink({ simulated: true, mode: fallback.state().mode });
  };

  setInterval(() => {
    if (live() && now() - receivedAt > STALE_MS) goSimulated();
  }, 500);

  const flush = () => {
    trailing = undefined;
    lastSent = now();
    send({ type: "robot_target", target: store.getState().target });
  };

  return {
    onSnapshot(r) {
      if (!r) {
        goSimulated();
        return;
      }
      receivedAt = now();
      latest = r.pose;
      shown ??= r.pose;
      const editing = now() - lastEdit < EDIT_HOLD_MS;
      store.setState((s) => ({
        link: { ...s.link, simulated: false, mode: editing ? s.link.mode : r.mode },
        target: editing ? s.target : (r.target ?? r.pose),
      }));
    },
    tick(): RobotPose {
      const t = now();
      const dt = t - lastTick;
      lastTick = t;
      if (!live() || !latest || !shown) return fallback.tick();
      shown = blendTarget(shown, latest, 1 - Math.exp(-dt / EASE_MS));
      return { headPose: poseMatrix(shown), antennas: shown.antennas, yawBody: shown.yawBody };
    },
    state: () => (live() ? store.getState().link : fallback.state()),
    target: () => (live() ? store.getState().target : fallback.target()),
    // Hooks: stable for React (always subscribe to both), the fallback's values while simulated.
    useLinkState: () => {
      const mine = store((s) => s.link);
      const sim = fallback.useLinkState();
      return mine.simulated ? { ...sim, simulated: true } : mine;
    },
    useTarget: () => {
      const simulated = store((s) => s.link.simulated);
      const mine = store((s) => s.target);
      const sim = fallback.useTarget();
      return simulated ? sim : mine;
    },
    setManual(on) {
      if (!live()) return fallback.setManual(on);
      lastEdit = now();
      setLink({ mode: on ? "manual" : "auto" });
      send({ type: "robot_mode", manual: on });
    },
    setTarget(t) {
      if (!live()) return fallback.setTarget(t);
      lastEdit = now();
      store.setState((s) => ({ target: clampTarget({ ...s.target, ...t }), link: { ...s.link, mode: "manual" } }));
      const wait = SEND_EVERY_MS - (now() - lastSent);
      if (wait <= 0) flush();
      else trailing ??= setTimeout(flush, wait);
    },
    center() {
      if (!live()) return fallback.center();
      this.setTarget(ZERO_TARGET);
    },
    playEmotion(name: EmotionName) {
      if (!live()) return fallback.playEmotion(name);
      send({ type: "robot_emotion", name });
    },
  };
}
