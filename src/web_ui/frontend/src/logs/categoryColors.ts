export const CATEGORY_COLORS: Record<string, string> = {
  Conversa: "#7c3aed",
  "Visão": "#0369a1",
  "Áudio": "#047857",
  Movimento: "#c2410c",
  Ferramentas: "#be185d",
  Sistema: "#475569",
  Outros: "#6b7280",
};

export const categoryColor = (cat: string) => CATEGORY_COLORS[cat] ?? "#6b7280";
