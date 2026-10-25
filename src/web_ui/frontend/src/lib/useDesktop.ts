import { useSyncExternalStore } from "react";

const query = "(min-width: 1024px)"; // Tailwind's lg, where the sidebar layout starts

const subscribe = (onChange: () => void) => {
  const list = window.matchMedia(query);
  list.addEventListener("change", onChange);
  return () => list.removeEventListener("change", onChange);
};

/** True on the desktop layout (sidebar), false on phones (tab bar). */
export function useDesktop(): boolean {
  return useSyncExternalStore(subscribe, () => window.matchMedia(query).matches);
}
