import { Sparkles } from "lucide-react";
import { robotLink, type EmotionName } from "../../robot";
import { Button } from "../../ui/Button";
import { Menu } from "../../ui/Menu";

const EMOTIONS: { name: EmotionName; label: string }[] = [
  { name: "feliz", label: "Feliz" },
  { name: "curioso", label: "Curioso" },
  { name: "sono", label: "Sono" },
  { name: "surpreso", label: "Surpreso" },
];

export function EmotionMenu() {
  return (
    <Menu
      align="end"
      trigger={
        <Button size="sm" className="w-full">
          <Sparkles className="size-4" /> Emoções
        </Button>
      }
      items={EMOTIONS.map((e) => ({ label: e.label, onSelect: () => robotLink.playEmotion(e.name) }))}
    />
  );
}
