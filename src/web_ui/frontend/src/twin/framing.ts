export type StageMode = "head" | "full" | "hidden";

export function stageModeFor(pathname: string): StageMode {
  if (pathname === "/") return "head";
  if (pathname === "/twin") return "full";
  return "hidden";
}

export const CAMERA_FOV = 30;

/** Camera distance at which a box of this size fits a view with this fov (degrees) and aspect. */
export function fitDistance(size: { width: number; height: number }, fovDeg: number, aspect: number, margin = 1.25): number {
  const tanV = Math.tan((fovDeg * Math.PI) / 360);
  const tanH = tanV * aspect;
  return Math.max(size.height / 2 / tanV, size.width / 2 / tanH) * margin;
}
