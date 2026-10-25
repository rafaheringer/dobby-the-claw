import { Check, Copy, Pause, Play, Search, Trash2, X } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { LEVELS, type LogFilter, type LogLevel } from "../lib/logFilter";
import { cx } from "../ui/Surface";
import { categoryColor } from "./categoryColors";

interface Props {
  filter: LogFilter;
  categories: string[];
  paused: boolean;
  onFilter(f: LogFilter): void;
  onPause(): void;
  onClear(): void;
  onCopy(): Promise<void>;
}

/** The official app's category chip: tinted with the category's color when on, plain when off. */
function Chip({ label, color, active, onClick }: { label: string; color: string; active: boolean; onClick(): void }) {
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      className="h-[22px] shrink-0 select-none rounded-[4px] border px-1.5 text-[10px] font-semibold transition-all duration-150 sm:h-[18px] sm:text-[9px]"
      style={active ? { color, background: `${color}18`, borderColor: `${color}40` } : { color: "#999", borderColor: "rgb(0 0 0 / 0.08)" }}
    >
      {label}
    </button>
  );
}

function IconButton({ label, active, onClick, children }: { label: string; active?: boolean; onClick(): void; children: ReactNode }) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      title={label}
      aria-pressed={active}
      className={cx(
        "grid size-7 shrink-0 place-items-center rounded-[4px] text-[#aaa] transition-colors hover:bg-black/[0.07] hover:text-[#555] sm:size-6",
        active ? "bg-ink text-white hover:bg-ink hover:text-white" : "bg-black/[0.03]",
      )}
    >
      {children}
    </button>
  );
}

/** One header line like the official app's full-size log console: label, filters, search, actions. */
export function LogToolbar({ filter, categories, paused, onFilter, onPause, onClear, onCopy }: Props) {
  const [q, setQ] = useState(filter.q);
  const [copied, setCopied] = useState(false);
  // Typing updates the URL only after a short pause, so history and the list don't churn per key.
  useEffect(() => {
    if (q === filter.q) return;
    const timer = setTimeout(() => onFilter({ ...filter, q }), 250);
    return () => clearTimeout(timer);
  }, [q, filter, onFilter]);
  useEffect(() => {
    if (!copied) return;
    const timer = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(timer);
  }, [copied]);

  const enabled = (cat: string) => filter.cats === null || filter.cats.includes(cat);
  const toggleCat = (cat: string) => {
    const current = filter.cats ?? categories;
    const next = enabled(cat) ? current.filter((c) => c !== cat) : [...current, cat];
    onFilter({ ...filter, cats: next.length === categories.length ? null : next });
  };

  return (
    <div className="flex flex-wrap items-center gap-1.5 border-b border-black/[0.08] px-3 py-2 sm:px-3">
      <span className="mr-0.5 text-[10px] font-semibold uppercase tracking-[0.5px] text-[#aaa]">Logs</span>
      <div className="order-last -mx-3 flex w-[calc(100%+1.5rem)] gap-1.5 overflow-x-auto px-3 [scrollbar-width:none] sm:order-none sm:mx-0 sm:w-auto sm:flex-wrap sm:px-0">
        {categories.map((cat) => (
          <Chip key={cat} label={cat} color={categoryColor(cat)} active={enabled(cat)} onClick={() => toggleCat(cat)} />
        ))}
      </div>
      <label className="flex h-7 min-w-[140px] flex-1 items-center gap-1 rounded-[5px] border border-black/[0.08] bg-black/[0.03] px-1.5 sm:h-6 sm:max-w-[220px]">
        <Search className="size-3 shrink-0 text-[#aaa]" />
        <input
          type="search"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Buscar..."
          title="Use !termo para esconder linhas"
          className="min-w-0 flex-1 bg-transparent font-mono text-[16px] text-[#333] outline-none placeholder:text-[#aaa] sm:text-[10px] [&::-webkit-search-cancel-button]:hidden"
        />
        {q && (
          <button onClick={() => setQ("")} aria-label="Limpar busca" className="text-[#aaa] hover:text-[#555]">
            <X className="size-3" />
          </button>
        )}
      </label>
      <span className="hidden flex-1 sm:block" />
      <select
        value={filter.level}
        onChange={(e) => onFilter({ ...filter, level: Number(e.target.value) as LogLevel })}
        aria-label="Nível"
        className="h-7 shrink-0 rounded-[4px] border border-black/[0.08] bg-black/[0.03] px-1 text-[10px] font-semibold text-[#555] outline-none sm:h-6 sm:text-[9px]"
      >
        {LEVELS.map((l) => (
          <option key={l.value} value={l.value}>
            {l.label}
          </option>
        ))}
      </select>
      <IconButton label={paused ? "Continuar" : "Pausar"} active={paused} onClick={onPause}>
        {paused ? <Play className="size-3" /> : <Pause className="size-3" />}
      </IconButton>
      <IconButton label="Limpar" onClick={onClear}>
        <Trash2 className="size-3" />
      </IconButton>
      <IconButton label={copied ? "Copiado" : "Copiar logs"} onClick={() => void onCopy().then(() => setCopied(true))}>
        {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
      </IconButton>
    </div>
  );
}
