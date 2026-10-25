import { Link, useLocation } from "react-router";
import { NAV } from "./nav";
import { sectionIndex } from "./routes";
import { StatePill } from "./StatePill";
import { WakeToggle } from "./WakeToggle";

/** Desktop navigation: a full-height column with numbered sections, like the keys on a device. */
export function Sidebar() {
  const { pathname } = useLocation();
  const active = sectionIndex(pathname);
  return (
    <aside className="sticky top-0 hidden h-dvh w-60 shrink-0 flex-col gap-8 border-r border-line px-5 py-7 lg:flex">
      <div className="flex flex-col gap-1">
        <span className="font-mono text-xl font-semibold tracking-[0.08em]">DOBBY</span>
        <span className="label">Reachy Mini · 01</span>
      </div>
      <nav className="flex flex-col gap-0.5 font-mono text-[13px] tracking-[0.06em]">
        {NAV.map(({ path, label }, i) => (
          <Link
            key={path}
            to={path}
            aria-current={i === active ? "page" : undefined}
            className={`flex h-11 items-center gap-3.5 px-3 uppercase transition-colors ${i === active ? "bg-ink text-white" : "hover:bg-subtle"}`}
          >
            <span className={i === active ? "text-accent" : "text-dim"}>{String(i + 1).padStart(2, "0")}</span>
            {label}
          </Link>
        ))}
      </nav>
      <div className="mt-auto flex flex-col gap-3">
        <span className="label">Estado</span>
        <StatePill className="self-start" />
        <WakeToggle className="w-full" />
      </div>
    </aside>
  );
}
