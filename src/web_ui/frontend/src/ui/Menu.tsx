import { Check } from "lucide-react";
import { DropdownMenu } from "radix-ui";
import type { ReactNode } from "react";

export interface MenuItem {
  label: string;
  checked?: boolean; // present = a checkbox item
  onSelect(): void;
}

/** A small glass dropdown opened by `trigger`. */
export function Menu({ trigger, items, align = "end" }: { trigger: ReactNode; items: MenuItem[]; align?: "start" | "end" }) {
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>{trigger}</DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align={align}
          sideOffset={8}
          className="glass z-50 min-w-44 rounded-2xl bg-[#1d2140]/85 p-1.5 text-sm shadow-xl data-[state=open]:animate-[menu-in_.15s_ease-out]"
        >
          {items.map((item) =>
            item.checked === undefined ? (
              <DropdownMenu.Item key={item.label} onSelect={item.onSelect} className="cursor-pointer rounded-xl px-3 py-2 outline-none data-highlighted:bg-white/10">
                {item.label}
              </DropdownMenu.Item>
            ) : (
              <DropdownMenu.CheckboxItem
                key={item.label}
                checked={item.checked}
                onSelect={(e) => {
                  e.preventDefault(); // keep the menu open while toggling several options
                  item.onSelect();
                }}
                className="flex cursor-pointer items-center justify-between gap-6 rounded-xl px-3 py-2 outline-none data-highlighted:bg-white/10"
              >
                {item.label}
                <DropdownMenu.ItemIndicator>
                  <Check className="size-4 text-accent" />
                </DropdownMenu.ItemIndicator>
              </DropdownMenu.CheckboxItem>
            ),
          )}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
