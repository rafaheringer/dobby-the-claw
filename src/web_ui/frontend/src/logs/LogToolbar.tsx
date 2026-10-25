import { Pause, Play, Search, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { LEVELS, type LogFilter, type LogLevel } from "../lib/logFilter";
import { Button } from "../ui/Button";
import { cx } from "../ui/Glass";
import { categoryColor } from "./categoryColors";

interface Props {
  filter: LogFilter;
  categories: string[];
  paused: boolean;
  onFilter(f: LogFilter): void;
  onPause(): void;
  onClear(): void;
}

export function LogToolbar({ filter, categories, paused, onFilter, onPause, onClear }: Props) {
  const [q, setQ] = useState(filter.q);
  // Typing updates the URL only after a short pause, so history and the list don't churn per key.
  useEffect(() => {
    if (q === filter.q) return;
    const timer = setTimeout(() => onFilter({ ...filter, q }), 250);
    return () => clearTimeout(timer);
  }, [q, filter, onFilter]);

  const enabled = (cat: string) => filter.cats === null || filter.cats.includes(cat);
  const toggleCat = (cat: string) => {
    const current = filter.cats ?? categories;
    const next = enabled(cat) ? current.filter((c) => c !== cat) : [...current, cat];
    onFilter({ ...filter, cats: next.length === categories.length ? null : next });
  };

  return (
    <div className="flex flex-col gap-3 border-b border-glass-border p-3">
      <div className="flex items-center gap-2">
        <label className="glass flex h-10 min-w-0 flex-1 items-center gap-2 rounded-full px-3">
          <Search className="size-4 shrink-0 text-dim" />
          <input
            type="search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Buscar"
            title="Use !termo para esconder linhas"
            className="min-w-0 flex-1 bg-transparent text-[16px] outline-none placeholder:text-dim sm:text-sm"
          />
        </label>
        <select
          value={filter.level}
          onChange={(e) => onFilter({ ...filter, level: Number(e.target.value) as LogLevel })}
          aria-label="Nível"
          className="glass h-10 shrink-0 rounded-full bg-transparent px-3 text-sm outline-none [&>option]:bg-[#1d2140]"
        >
          {LEVELS.map((l) => (
            <option key={l.value} value={l.value}>
              {l.label}
            </option>
          ))}
        </select>
        <Button icon aria-label={paused ? "Continuar" : "Pausar"} onClick={onPause} variant={paused ? "primary" : "glass"}>
          {paused ? <Play className="size-4" /> : <Pause className="size-4" />}
        </Button>
        <Button icon aria-label="Limpar" onClick={onClear}>
          <Trash2 className="size-4" />
        </Button>
      </div>
      <div className="-mx-3 flex gap-2 overflow-x-auto px-3 [scrollbar-width:none]">
        {categories.map((cat) => (
          <button
            key={cat}
            onClick={() => toggleCat(cat)}
            aria-pressed={enabled(cat)}
            className={cx(
              "flex h-8 shrink-0 items-center gap-1.5 rounded-full border px-3 text-xs font-medium transition-opacity",
              enabled(cat) ? "border-white/15 bg-white/8" : "border-transparent opacity-45",
            )}
          >
            <span className="size-2 rounded-full" style={{ background: categoryColor(cat) }} />
            {cat}
          </button>
        ))}
      </div>
    </div>
  );
}
