import { Link } from "react-router";
import { useShallow } from "zustand/react/shallow";
import { AudioCard } from "../home/AudioCard";
import { CameraCard } from "../home/CameraCard";
import { Indicators } from "../home/Indicators";
import { useLive } from "../live/store";
import { TwinSlot } from "../twin/TwinSlot";
import { Card, CardHeader } from "../ui/Surface";
import { PageTitle } from "../ui/PageTitle";
import { StatePill } from "../app/StatePill";

function RecentChat() {
  const chat = useLive(useShallow((s) => s.chat.slice(-4)));
  return (
    <Card className="flex min-h-0 flex-1 flex-col">
      <CardHeader
        end={
          <Link to="/conversa" className="label hover:text-text">
            Ver tudo →
          </Link>
        }
      >
        Última conversa
      </CardHeader>
      {chat.length === 0 && <p className="p-3 text-sm text-dim">Nada ainda.</p>}
      <ul className="flex flex-col gap-3 p-3 text-sm leading-relaxed">
        {chat.map((m, i) => (
          <li key={`${m.t}-${m.who}-${i}`} className="flex gap-3">
            <span className={`label w-12 shrink-0 pt-0.5 ${m.who === "dobby" ? "text-[#c2410c]" : ""}`}>{m.who === "dobby" ? "Dobby" : "Você"}</span>
            <span className="line-clamp-2 min-w-0">{m.text}</span>
          </li>
        ))}
      </ul>
    </Card>
  );
}

/** Dobby's head, free on the page (no card); clicking it opens the full twin and its controls. */
function TwinHead() {
  return <TwinSlot className="mx-auto aspect-square w-full max-w-[280px] lg:aspect-[16/10] lg:max-w-none" />;
}

export default function Home() {
  return (
    <div>
      <PageTitle
        action={
          // on phones the top bar already shows it
          <span className="hidden lg:block">
            <StatePill />
          </span>
        }
      >Início</PageTitle>
      <div className="flex flex-col gap-4 lg:grid lg:grid-cols-[minmax(0,1fr)_340px] lg:gap-5">
        <div className="lg:order-2 lg:flex lg:flex-col lg:gap-5">
          <TwinHead />
          <div className="hidden lg:flex lg:flex-1 lg:flex-col">
            <RecentChat />
          </div>
        </div>
        <div className="flex flex-col gap-4 lg:order-1 lg:gap-5">
          <CameraCard />
          <Indicators />
          <AudioCard />
        </div>
      </div>
    </div>
  );
}
