export const MIC_FULL_RMS = 0.05; // the level that fills the meter and tops the waveform

/** Mic RMS to a 0-1 level. */
export const micLevel = (rms: number) => Math.min(1, rms / MIC_FULL_RMS);

/** The sound-direction arrow's CSS rotation: 0 rad (left) points left, pi/2 (front) up, pi (right) right. */
export const doaRotation = (angle: number) => (angle / Math.PI) * 180 - 90;

/** Mute toggles between silence and the last level that was not silent (half when there is none). */
export const toggledVolume = (volume: number, lastAudible: number) => (volume > 0 ? 0 : lastAudible > 0 ? lastAudible : 50);

/** Appends one waveform sample, keeping the last `max`. */
export function pushSample(history: number[], level: number, max: number): number[] {
  const next = [...history, Math.max(0.05, Math.min(0.95, level))];
  return next.length > max ? next.slice(next.length - max) : next;
}
