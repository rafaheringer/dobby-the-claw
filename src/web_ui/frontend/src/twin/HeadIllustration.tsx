/** A flat drawing of Dobby's head, shown while the 3D model loads or when it cannot be drawn. */
export function HeadIllustration({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 200 170" className={className} aria-hidden>
      <defs>
        <linearGradient id="hi-shell" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="1" stopColor="#c9cde0" />
        </linearGradient>
        <radialGradient id="hi-eye" cx=".4" cy=".35" r=".7">
          <stop offset="0" stopColor="#5a6aa8" />
          <stop offset=".45" stopColor="#1a1f36" />
          <stop offset="1" stopColor="#05060b" />
        </radialGradient>
      </defs>
      <path d="M70 62 C 62 40, 74 30, 64 8" stroke="#d6d9e8" strokeWidth="5" fill="none" strokeLinecap="round" />
      <circle cx="64" cy="8" r="5" fill="var(--accent)" />
      <path d="M130 62 C 138 40, 126 30, 136 8" stroke="#d6d9e8" strokeWidth="5" fill="none" strokeLinecap="round" />
      <circle cx="136" cy="8" r="5" fill="var(--accent)" />
      <rect x="22" y="52" width="156" height="104" rx="52" fill="url(#hi-shell)" />
      <rect x="38" y="72" width="124" height="64" rx="32" fill="#10131f" />
      <circle cx="74" cy="104" r="20" fill="url(#hi-eye)" />
      <circle cx="126" cy="104" r="20" fill="url(#hi-eye)" />
      <circle cx="68" cy="97" r="5" fill="#fff" opacity=".7" />
      <circle cx="120" cy="97" r="5" fill="#fff" opacity=".7" />
    </svg>
  );
}
