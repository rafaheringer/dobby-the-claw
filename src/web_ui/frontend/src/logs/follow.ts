import type { LogLine } from "../live/protocol";

/**
 * Changes whenever new lines arrive. The length alone is not enough: once the list is capped,
 * each burst drops as many old lines as it adds.
 */
export function followKey(lines: LogLine[]): string {
  return `${lines.length}:${lines.at(-1)?.id ?? 0}`;
}

/**
 * Whether the list should keep sticking to the newest line after a scroll event. Only the person
 * can stop it (by scrolling away from the bottom); scrolls the list makes itself, while rows are
 * still being measured, never do. Scrolling back to the bottom resumes it.
 */
export function keepFollowing(following: boolean, scroll: { nearBottom: boolean; byUser: boolean }): boolean {
  if (scroll.nearBottom) return true;
  return scroll.byUser ? false : following;
}
