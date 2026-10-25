import { Mic, MicOff, Volume2, VolumeX } from "lucide-react";
import { Slider } from "radix-ui";
import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Volume, Volumes } from "../live/protocol";
import { send, useAudio, useLive } from "../live/store";
import { Card, CardHeader, cx } from "../ui/Surface";
import { doaRotation, toggledVolume } from "./audioValues";
import { Waveform } from "./Waveform";

const SEND_AFTER_MS = 500; // like the official app: the daemon gets the value once the slider rests
const TALK_HOLD_MS = 600; // "falando" lingers over the pauses between words

/** One device's volume: follows the daemon, but what the person is dragging wins until it is sent. */
function useVolume(device: keyof Volumes) {
  const remote = useLive((s) => s.volumes[device]);
  // The person's value, until it is sent and the daemon has answered (a new `remote`).
  const [edit, setEdit] = useState<{ value: number; seen: Volume | null; pending: boolean } | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const lastAudible = useRef(50);
  useEffect(() => () => clearTimeout(timer.current), []);

  const value = edit && (edit.pending || edit.seen === remote) ? edit.value : (remote?.volume ?? 0);
  useEffect(() => {
    if (value > 0) lastAudible.current = value;
  }, [value]);
  const commit = (volume: number) => {
    clearTimeout(timer.current);
    setEdit((e) => e && { ...e, pending: false });
    send({ type: "volume", device, volume });
  };
  return {
    remote,
    value,
    change(volume: number) {
      setEdit({ value: volume, seen: remote, pending: true });
      clearTimeout(timer.current);
      timer.current = setTimeout(() => commit(volume), SEND_AFTER_MS);
    },
    toggleMute() {
      const volume = toggledVolume(value, lastAudible.current);
      setEdit({ value: volume, seen: remote, pending: false });
      commit(volume);
    },
  };
}

/** Whether the robot hears speech now, held briefly so it does not flicker between words. */
function useTalking(): { angle: number | null; talking: boolean } {
  const angle = useAudio((s) => s.sample?.doa?.angle ?? null);
  const [talking, setTalking] = useState(false);
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined;
    const unsubscribe = useAudio.subscribe((s) => {
      if (s.sample?.doa?.speech) {
        clearTimeout(timer);
        timer = undefined;
        setTalking(true);
      } else
        timer ??= setTimeout(() => {
          timer = undefined;
          setTalking(false);
        }, TALK_HOLD_MS);
    });
    return () => {
      unsubscribe();
      clearTimeout(timer);
    };
  }, []);
  return { angle, talking };
}

/** Where the sound comes from, and whether it is speech (the official app's DoA pill). */
function DirectionPill() {
  const { angle, talking } = useTalking();
  if (angle === null) return null;
  return (
    <span
      title="Direção do som"
      className={cx(
        "flex h-6 items-center gap-1.5 border px-2 font-mono text-[10px] font-medium uppercase tracking-[0.06em] transition-colors duration-300",
        talking ? "border-ok/40 bg-ok/10 text-ok" : "border-transparent bg-subtle text-dim",
      )}
    >
      <svg viewBox="0 0 10 10" className="size-2.5 transition-transform duration-200 ease-out" style={{ transform: `rotate(${doaRotation(angle)}deg)` }}>
        <path d="M5 1.5 L5 8.5 M5 1.5 L2.5 4 M5 1.5 L7.5 4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <span className="hidden w-12 sm:inline">{talking ? "Falando" : "Ouvindo"}</span>
    </span>
  );
}

function VolumeSlider({ label, value, disabled, onChange }: { label: string; value: number; disabled: boolean; onChange(v: number): void }) {
  return (
    <Slider.Root
      value={[value]}
      min={0}
      max={100}
      step={1}
      disabled={disabled}
      onValueChange={([v]) => onChange(v)}
      aria-label={label}
      className="relative flex h-8 w-full touch-none select-none items-center data-[disabled]:opacity-40"
    >
      <Slider.Track className="relative h-[3px] grow overflow-hidden bg-line">
        <Slider.Range className="absolute h-full bg-ink" />
      </Slider.Track>
      <Slider.Thumb className="block size-4 rounded-full border-2 border-ink bg-accent outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2" />
    </Slider.Root>
  );
}

function Channel({ device, label, on, off, extra, children }: { device: keyof Volumes; label: string; on: ReactNode; off: ReactNode; extra?: ReactNode; children?: ReactNode }) {
  const v = useVolume(device);
  const missing = !v.remote;
  return (
    <div className="flex min-w-0 flex-col border-line [&:not(:last-child)]:border-r">
      <div className="flex min-w-0 flex-col gap-1.5 px-3 pb-2 pt-3 sm:px-4">
        <div className="flex h-6 items-center justify-between gap-2">
          <span className="label truncate">{label}</span>
          {extra}
        </div>
        <div className="flex min-w-0 flex-col">
          <span className="truncate font-mono text-[11px] font-medium text-text">{missing ? "Indisponível" : v.remote!.device || "—"}</span>
          <span className="truncate font-mono text-[10px] text-dim">{missing ? "sem conexão com o daemon" : v.remote!.platform}</span>
        </div>
        <div className="flex min-w-0 items-center gap-2">
          <button
            onClick={v.toggleMute}
            disabled={missing}
            aria-label={v.value > 0 ? `Silenciar ${label.toLowerCase()}` : `Ligar ${label.toLowerCase()}`}
            className="-ml-2 grid size-8 shrink-0 place-items-center text-dim hover:text-accent disabled:opacity-40"
          >
            {v.value > 0 ? on : off}
          </button>
          <div className="min-w-0 flex-1">
            <VolumeSlider label={`Volume: ${label}`} value={v.value} disabled={missing} onChange={v.change} />
          </div>
          <span className="w-7 shrink-0 text-right font-mono text-xs tabular">{missing ? "—" : v.value}</span>
        </div>
      </div>
      {children}
    </div>
  );
}

/** Speaker and microphone like the official Reachy Mini app: device, mute, volume, live waveform. */
export function AudioCard() {
  const status = useLive((s) => s.status);
  const mic = useLive((s) => s.volumes.mic);
  const hearing = !!status?.mic && !status.sleeping;
  const heard = status?.sleeping ? status.wake?.heard : null;
  return (
    <Card>
      <CardHeader>Áudio</CardHeader>
      <div className="grid grid-cols-2">
        <Channel device="speaker" label="Alto-falante" on={<Volume2 className="size-4" />} off={<VolumeX className="size-4" />} />
        <Channel device="mic" label="Microfone" on={<Mic className="size-4" />} off={<MicOff className="size-4" />} extra={hearing ? <DirectionPill /> : undefined}>
          {status?.sleeping ? (
            <p className="flex h-8 items-center gap-2 truncate border-t border-line px-4 font-mono text-[11px] text-dim">
              <span className="label">Ouviu</span>
              <span className="truncate text-text">{heard ? `“${heard}”` : "—"}</span>
            </p>
          ) : (
            status?.mic && <Waveform active={hearing && (mic?.volume ?? 1) > 0} />
          )}
        </Channel>
      </div>
    </Card>
  );
}
