import { Bubble } from "../chat/Bubble";
import { Composer } from "../chat/Composer";
import { useStickToBottom } from "../chat/useStickToBottom";
import { useLive } from "../live/store";
import { PageTitle } from "../ui/PageTitle";

export default function Chat() {
  const chat = useLive((s) => s.chat);
  useStickToBottom(chat.length);
  return (
    <div className="flex max-w-3xl flex-col pb-20 lg:min-h-[calc(100dvh-3.75rem)] lg:pb-0">
      <PageTitle>Conversa</PageTitle>
      {chat.length === 0 && <p className="py-16 text-center text-dim">Nenhuma mensagem ainda.</p>}
      <ul className="flex flex-col gap-3">
        {chat.map((m, i) => (
          <Bubble key={`${m.t}-${m.who}-${i}`} entry={m} />
        ))}
      </ul>
      <Composer />
    </div>
  );
}
