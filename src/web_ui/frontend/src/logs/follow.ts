import type { LogLine } from "../live/protocol";

/**
 * Changes whenever new lines arrive. The length alone is not enough: once the list is capped,
 * each burst drops as many old lines as it adds.
 */
export function followKey(lines: LogLine[]): string {
  return `${lines.length}:${lines.at(-1)?.id ?? 0}`;
}
