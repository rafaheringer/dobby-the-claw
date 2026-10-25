// The /ws protocol, mirroring src/web_ui/server.py (Hub.status, Hub.hello, Hub.handle) and
// src/web_ui/robot_control.py (the "robot" pose stream and the robot_* commands).

import type { EmotionName, RobotTarget } from "../robot/types";

export type RobotState = "IDLE" | "LISTENING" | "THINKING" | "EXECUTING" | "DELEGATING" | "CONFIRMING" | "ERROR";

export interface PiStats {
  cpu_load: number;
  mem_used_gb: number;
  mem_total_gb: number;
}

export interface Status {
  state: RobotState;
  sleeping: boolean;
  transition: "" | "sleep" | "wake";
  model: string;
  people: { id: number | string; name: string | null }[];
  speaker: string | null;
  camera: boolean;
  head_tracking: boolean;
  video_enabled: boolean;
  fps: number;
  mic: { rms: number } | null;
  wake: { heard: string | null } | null;
  services: { ha?: boolean | null; oc?: boolean | null; pi?: PiStats | null };
  log_level: string;
}

export interface LogLine {
  id: number;
  t: number;
  lv: number;
  cat: string;
  msg: string;
}

export interface ChatEntry {
  t: number;
  who: "dobby" | "you";
  text: string;
}

/** The pose the robot is commanded to (15 per second), and whether a page is driving it. */
export interface RobotSnapshot {
  type: "robot";
  mode: "auto" | "manual";
  pose: RobotTarget;
  target: RobotTarget | null;
}

export type ServerMessage =
  | { type: "hello"; categories: string[]; logs: LogLine[]; chat: ChatEntry[]; status: Status; robot: RobotSnapshot | null }
  | RobotSnapshot
  | ({ type: "status" } & Status)
  | { type: "logs"; lines: LogLine[] }
  | { type: "chat"; entry: ChatEntry }
  | { type: "wake_state"; recording: boolean; thinking?: boolean; seconds?: number }
  | { type: "wake_result"; text: string; hit: boolean; silent: boolean }
  | { type: "error"; text: string };

export type ClientMessage =
  | { type: "chat"; text: string }
  | { type: "cmd"; name: "wake" | "sleep" }
  | { type: "head_tracking"; enabled: boolean }
  | { type: "video_enabled"; enabled: boolean }
  | { type: "log_level"; level: "DEBUG" | "INFO" }
  | { type: "wake_test" }
  | { type: "robot_mode"; manual: boolean }
  | { type: "robot_target"; target: Partial<RobotTarget> }
  | { type: "robot_emotion"; name: EmotionName };
