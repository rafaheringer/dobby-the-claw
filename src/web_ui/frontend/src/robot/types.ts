// The seam between the UI and the robot. Today MockRobotLink fakes it; a real link (daemon state
// stream + commands through the bridge) only has to implement RobotLink.

export interface RobotPose {
  headPose: number[]; // 4x4 row-major, metres (robot frame: x forward, y left, z up)
  antennas: [number, number]; // radians [right, left], as the Reachy SDK orders them
  yawBody: number; // radians
}

export interface RobotTarget {
  x: number;
  y: number;
  z: number;
  roll: number;
  pitch: number;
  yaw: number;
  antennas: [number, number];
  yawBody: number;
}

export type EmotionName = "feliz" | "curioso" | "sono" | "surpreso";

export interface LinkState {
  mode: "auto" | "manual";
  simulated: boolean;
  emotion: EmotionName | "atento" | null;
}

export interface RobotLink {
  /** Advance and return the pose to draw now; the 3D loop calls it once per frame. */
  tick(): RobotPose;
  state(): LinkState;
  target(): RobotTarget;
  /** React hooks over state() and target(); they change on commands, not every frame. */
  useLinkState(): LinkState;
  useTarget(): RobotTarget;
  setManual(on: boolean): void;
  setTarget(t: Partial<RobotTarget>): void;
  center(): void;
  playEmotion(name: EmotionName): void;
}
