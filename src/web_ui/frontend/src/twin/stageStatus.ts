import { create } from "zustand";

/** Whether the 3D robot is drawn (ready) or could not be (failed: no WebGL or the model broke). */
export const useStageStatus = create<{ ready: boolean; failed: boolean }>(() => ({ ready: false, failed: !hasWebGL() }));

export function hasWebGL(): boolean {
  if (typeof document === "undefined") return false;
  try {
    const canvas = document.createElement("canvas");
    return !!(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch {
    return false;
  }
}

/** Height of the twin's controls panel on phones (0: not measured yet), read by the 3D framing. */
export const useTwinPanel = create<{ phonePx: number }>(() => ({ phonePx: 0 }));
