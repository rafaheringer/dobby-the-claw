import { Link } from "react-router";
import { useShallow } from "zustand/react/shallow";
import { CameraCard } from "../home/CameraCard";
import { Indicators } from "../home/Indicators";
import { useLive } from "../live/store";
import { TwinSlot } from "../twin/TwinSlot";
import { Card } from "../ui/Glass";

function RecentChat() {
  const chat = useLive(useShallow((s) => s.chat.slice(-4)));
  return (
    <Card className="flex min-h-0 flex-1 flex-col p-4">
      <Link to="/conversa" className="mb-2 text-xs font-semibold uppercase tracking-wider text-dim hover:text-text">
        Conversa
      </Link>
      {chat.length === 0 && <p className="text-sm text-dim">Nada ainda.</p>}
      <ul className="flex flex-col gap-2 text-sm">
        {chat.map((m, i) => (
          <li key={`${m.t}-${m.who}-${i}`} className="line-clamp-2">
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
    <div className="flex flex-col gap-3 pt-1 lg:grid lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)] lg:gap-4 lg:pt-0">
      <div className="lg:order-2 lg:flex lg:flex-col lg:gap-4">
        <Card className="overflow-hidden">
          <TwinSlot className="mx-auto aspect-square w-full max-w-[300px] lg:aspect-[16/10] lg:max-w-none" />
        </Card>
        <div className="hidden lg:flex lg:flex-1 lg:flex-col">
          <RecentChat />
        </div>
      </div>
      <div className="flex flex-col gap-3 lg:order-1 lg:gap-4">
        <CameraCard />
        <Indicators />
      </div>
    </div>
  );
}
