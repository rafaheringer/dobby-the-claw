import { useSyncExternalStore } from "react";

const subscribe = (cb: () => void) => {
  document.addEventListener("visibilitychange", cb);
  return () => document.removeEventListener("visibilitychange", cb);
};

/** False while the tab is hidden or the phone is locked. */
export function usePageVisible(): boolean {
  return useSyncExternalStore(subscribe, () => document.visibilityState === "visible");
}
