// Per-device preferences; storage can be unavailable (private mode), so every access is guarded.

export function loadPref<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : { ...fallback, ...JSON.parse(raw) };
  } catch {
    return fallback;
  }
}

export function savePref(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // not saved: the preference only lasts for this visit
  }
}
