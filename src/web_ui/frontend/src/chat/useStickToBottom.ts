import { useLayoutEffect, useRef } from "react";

const NEAR_PX = 60;

const nearBottom = () => window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - NEAR_PX;

/** Keep the page scrolled to the end when `count` grows, unless the reader scrolled up. */
export function useStickToBottom(count: number): void {
  const wasNear = useRef(true);
  const first = useRef(true);

  useLayoutEffect(() => {
    const onScroll = () => (wasNear.current = nearBottom());
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useLayoutEffect(() => {
    if (first.current || wasNear.current) {
      window.scrollTo({ top: document.documentElement.scrollHeight, behavior: first.current ? "instant" : "smooth" });
    }
    first.current = false;
  }, [count]);
}
