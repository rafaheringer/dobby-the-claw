import { House, MessageCircle, ScrollText, SlidersHorizontal, type LucideIcon } from "lucide-react";
import { SECTIONS } from "./routes";

const ICONS: LucideIcon[] = [House, MessageCircle, ScrollText, SlidersHorizontal];

export const NAV = SECTIONS.map((s, i) => ({ ...s, Icon: ICONS[i] }));
