export const CATEGORY_COLORS: Record<string, string> = {
  Conversa: "#a78bfa",
  "Visão": "#38bdf8",
  "Áudio": "#34d399",
  Movimento: "#fb923c",
  Ferramentas: "#f472b6",
  Sistema: "#94a3b8",
  Outros: "#9ca3af",
};

export const categoryColor = (cat: string) => CATEGORY_COLORS[cat] ?? "#9ca3af";
