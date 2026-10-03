"""Latency spike: Gemini Live text turn -> first audio, plus one tool call."""
import asyncio, os, sys, time
from google import genai
from google.genai import types

MODELS = sys.argv[1:] or ["gemini-3.8-live", "gemini-3.8-live-extended-thinking"]
PROMPTS = ["Oi Dobby, tudo bem?", "Que horas são agora?"]
TOOL = types.Tool(function_declarations=[types.FunctionDeclaration(
    name="get_current_time", description="Retorna a data e hora atuais do sistema.")])


async def run(model):
    global T0
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    cfg = {
        "response_modalities": ["AUDIO"],
        "system_instruction": "Você é o Dobby, robô sarcástico. Responda sempre em português do Brasil, em até 20 palavras. Para hora/data use a tool get_current_time.",
        "tools": [TOOL],
        "output_audio_transcription": {},
        "input_audio_transcription": {},
    }
    if "extended" in model:
        cfg["thinking_config"] = {"thinking_level": os.environ.get("LEVEL", "low")}
    print(f"\n=== {model}")
    async with client.aio.live.connect(model=model, config=cfg) as s:
        for p in PROMPTS:
            t0 = time.monotonic(); first = None; text = ""; tool_t = None; pending = False; first_after = None
            await s.send_realtime_input(text=p)
            t_after = None
            for _ in range(4):
                called = False
                async for r in s.receive():
                    if os.environ.get('DBG'): print('  evt', round(time.monotonic()-t0,2), [k for k,v in r.model_dump(exclude_none=True).items() if k!='usage_metadata'], (r.server_content and {k:v for k,v in r.server_content.model_dump(exclude_none=True).items() if k not in ('model_turn','output_transcription')}) or '')
                    if r.tool_call:
                        called = True
                        tool_t = tool_t or time.monotonic() - t0
                        await s.send_tool_response(function_responses=[types.FunctionResponse(
                            id=c.id, name=c.name, response={"time": "18:42", "weekday": "sábado"}) for c in r.tool_call.function_calls])
                        t_after = time.monotonic()
                    sc = r.server_content
                    if sc:
                        if sc.model_turn:
                            for part in sc.model_turn.parts:
                                if part.inline_data and (first is None or (t_after and not first_after)):
                                    if first is None: first = time.monotonic() - t0
                                    if t_after and not first_after: first_after = time.monotonic() - t_after
                        if sc.output_transcription:
                            text += sc.output_transcription.text or ""
                if not called:
                    break
            print(f"'{p}' first_audio={first and round(first,2)}s tool_call_at={tool_t and round(tool_t,2)}s audio_after_tool={first_after and round(first_after,2)}s total={round(time.monotonic()-t0,2)}s -> {text.strip()}")

async def main():
    for m in MODELS:
        try:
            await run(m)
        except Exception as e:
            print(f"{m} FAILED: {type(e).__name__}: {e}")

async def guarded():
    await asyncio.wait_for(main(), 90)
asyncio.run(guarded())
