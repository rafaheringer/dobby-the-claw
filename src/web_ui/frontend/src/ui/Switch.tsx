import { Switch as RadixSwitch } from "radix-ui";
import { useId } from "react";

/** A labeled on/off row. */
export function Switch({ label, checked, disabled, onChange }: { label: string; checked: boolean; disabled?: boolean; onChange(on: boolean): void }) {
  const id = useId();
  return (
    <div className="flex min-h-12 items-center justify-between gap-4 border-b border-line last:border-0">
      <label htmlFor={id} className={disabled ? "text-dim" : undefined}>
        {label}
      </label>
      <RadixSwitch.Root
        id={id}
        checked={checked}
        disabled={disabled}
        onCheckedChange={onChange}
        className="relative h-6 w-11 shrink-0 rounded-full border border-ink bg-surface transition-colors data-[state=checked]:bg-accent disabled:opacity-40"
      >
        <RadixSwitch.Thumb className="block size-4 translate-x-[3px] rounded-full bg-ink transition-transform duration-200 data-[state=checked]:translate-x-[23px]" />
      </RadixSwitch.Root>
    </div>
  );
}
