import { useCallback, useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { useLive } from "../live/store";
import { matchLog, parseLogParams, toLogParams, type LogFilter } from "../lib/logFilter";
import { LogList } from "../logs/LogList";
import { LogToolbar } from "../logs/LogToolbar";
import { Card } from "../ui/Surface";
import { PageTitle } from "../ui/PageTitle";

export default function Logs() {
  const [params, setParams] = useSearchParams();
  const filter = useMemo(() => parseLogParams(params), [params]);
  const setFilter = useCallback((f: LogFilter) => setParams(toLogParams(f), { replace: true }), [setParams]);
  const logs = useLive((s) => s.logs);
  const categories = useLive((s) => s.categories);
  const [frozen, setFrozen] = useState<typeof logs | null>(null);
  const [clearedAt, setClearedAt] = useState(0);

  const source = frozen ?? logs;
  const shown = useMemo(() => {
    const match = matchLog(filter);
    return source.filter((l) => l.id > clearedAt && match(l));
  }, [source, filter, clearedAt]);

  return (
    <div className="flex h-[calc(100dvh-env(safe-area-inset-top)-10rem)] flex-col lg:h-[calc(100dvh-3.75rem)]">
      <PageTitle>Logs</PageTitle>
      <Card className="flex min-h-0 flex-1 flex-col overflow-hidden">
        <LogToolbar
          filter={filter}
          categories={categories}
          paused={frozen !== null}
          onFilter={setFilter}
          onPause={() => setFrozen(frozen ? null : logs)}
          onClear={() => setClearedAt(logs.at(-1)?.id ?? 0)}
        />
        <LogList lines={shown} />
      </Card>
    </div>
  );
}
