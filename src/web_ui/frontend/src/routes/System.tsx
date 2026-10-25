import { ChevronRight, Mic } from "lucide-react";
import { Link } from "react-router";
import { send, useLive } from "../live/store";
import { ServiceRow } from "../system/ServiceRow";
import { serviceRows } from "../system/services";
import { Panel } from "../ui/Glass";
import { PageTitle } from "../ui/PageTitle";
import { Switch } from "../ui/Switch";

export default function System() {
  const status = useLive((s) => s.status);
  const open = useLive((s) => s.connection === "open");
  const off = !open || !status;
  return (
    <div className="max-w-4xl">
      <PageTitle>Sistema</PageTitle>
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Serviços" className="lg:row-span-2">
          {status ? serviceRows(status).map((row) => <ServiceRow key={row.key} row={row} />) : <p className="text-sm text-dim">Sem conexão.</p>}
        </Panel>
        <Panel title="Controles">
          <Switch
            label="Cabeça segue"
            checked={!!status?.head_tracking}
            disabled={off || !status?.camera}
            onChange={(on) => send({ type: "head_tracking", enabled: on })}
          />
          <Switch label="Câmera ligada" checked={!!status?.video_enabled} disabled={off} onChange={(on) => send({ type: "video_enabled", enabled: on })} />
          <Switch
            label="Log em Debug"
            checked={status?.log_level === "DEBUG"}
            disabled={off}
            onChange={(on) => send({ type: "log_level", level: on ? "DEBUG" : "INFO" })}
          />
        </Panel>
        <Link to="/sistema/nome" className="glass flex min-h-14 items-center gap-3 rounded-3xl px-4 transition-colors hover:bg-glass-strong">
          <span className="grid size-9 place-items-center rounded-full bg-accent/20 text-accent">
            <Mic className="size-4" />
          </span>
          <span className="flex-1 font-medium">Teste do nome</span>
          <ChevronRight className="size-5 text-dim" />
        </Link>
      </div>
    </div>
  );
}
