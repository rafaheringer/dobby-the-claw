interface Navigate {
  (delta: number): unknown;
  (to: string, opts?: { replace?: boolean }): unknown;
}

/**
 * Leave the twin: back through history when the app got here itself, otherwise (a deep link or a
 * reload on /twin) to the home page, so "back" never leaves the site.
 */
export function closeTwin(navigate: Navigate, historyIdx: number | undefined): void {
  if (historyIdx && historyIdx > 0) navigate(-1);
  else navigate("/", { replace: true });
}
