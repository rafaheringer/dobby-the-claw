export const SECTIONS = [
  { path: "/", label: "Início" },
  { path: "/conversa", label: "Conversa" },
  { path: "/logs", label: "Logs" },
  { path: "/sistema", label: "Sistema" },
] as const;

/** Which tab a path belongs to (sub-pages count as their tab), or -1. */
export function sectionIndex(pathname: string): number {
  if (pathname === "/") return 0;
  return SECTIONS.findIndex((s, i) => i > 0 && (pathname === s.path || pathname.startsWith(`${s.path}/`)));
}
