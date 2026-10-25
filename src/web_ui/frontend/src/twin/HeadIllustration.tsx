/** A flat line drawing of Dobby's head, shown while the 3D model loads or when it cannot be drawn. */
export function HeadIllustration({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 200 170" className={className} aria-hidden>
      <path d="M74 58 C70 36 58 22 50 10" stroke="var(--ink)" strokeWidth="4" fill="none" strokeLinecap="round" />
      <path d="M126 58 C130 36 142 22 150 10" stroke="var(--ink)" strokeWidth="4" fill="none" strokeLinecap="round" />
      <circle cx="50" cy="10" r="6" fill="var(--accent)" />
      <circle cx="150" cy="10" r="6" fill="var(--accent)" />
      <rect x="30" y="52" width="140" height="104" rx="48" fill="#fff" stroke="var(--ink)" strokeWidth="3" />
      <rect x="50" y="78" width="100" height="50" rx="25" fill="var(--ink)" />
      <circle cx="78" cy="103" r="13" fill="#333" stroke="#fff" strokeWidth="2" />
      <circle cx="122" cy="103" r="13" fill="#333" stroke="#fff" strokeWidth="2" />
      <circle cx="78" cy="103" r="4" fill="#fff" />
      <circle cx="122" cy="103" r="4" fill="#fff" />
    </svg>
  );
}
