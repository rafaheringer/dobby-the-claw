// Few fixed sizes: a width that drifts with the layout would restart the stream again and again.
export function streamWidth(px: number): 640 | 960 | 1280 {
  return px <= 700 ? 640 : px <= 1100 ? 960 : 1280;
}

export function videoUrl(o: { people: boolean; voice: boolean; tracking: boolean; width: number }): string {
  const flag = (on: boolean) => (on ? "1" : "0");
  const q = new URLSearchParams({ people: flag(o.people), voice: flag(o.voice), tracking: flag(o.tracking), w: String(o.width) });
  return `/api/video?${q}`;
}
