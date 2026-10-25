import { create } from "zustand";

/**
 * The route being shown, set by the layout. A page that is sliding out still sees its own route
 * through the router hooks; this tells it the page has already changed.
 */
export const useCurrentPath = create<string>(() => window.location.pathname);
