import { Link } from "react-router";
import { CameraCard } from "../home/CameraCard";
import { Indicators } from "../home/Indicators";
import { useShallow } from "zustand/react/shallow";
import { useLive } from "../live/store";
import { TwinSlot } from "../twin/TwinSlot";
import { Card } from "../ui/Glass";

function RecentChat() {
  const chat = useLive(useShallow((s) => s.chat.slice(-4)));
  return (
    <Card className="flex min-h-0 flex-col p-4">
      <Link to="/conversa" className="mb-2 text-xs font-semibold uppercase tracking-wider text-dim hover:text-text">
        Conversa
      </Link>
      {chat.length === 0 && <p className="text-sm text-dim">Nada ainda.</p>}
      <ul className="flex flex-col gap-2 text-sm">
        {chat.map((m) => (
          <li key={`${m.t}-${m.who}`} className="line-clamp-2">
            <b className={m.who === "dobby" ? "text-accent" : "text-text"}>{m.who === "dobby" ? "Dobby" : "Você"}</b>{" "}
            <span className="text-dim">{m.text}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

export default function Home() {
  return (
    <div className="flex flex-col gap-3 pt-1 lg:grid lg:grid-cols-[1.5fr_1fr] lg:grid-rows-[auto_1fr] lg:gap-4 lg:pt-3">
      <TwinSlot className="mx-auto aspect-square w-full max-w-[280px] lg:order-2 lg:max-w-none lg:aspect-[4/3]" />
      <div className="flex flex-col gap-3 lg:order-1 lg:row-span-2 lg:gap-4">
        <CameraCard />
        <Indicators />
      </div>
      <div className="hidden lg:order-3 lg:flex lg:flex-col">
        <RecentChat />
      </div>
    </div>
  );
}
