// A fake bridge for frontend work while the real one is down: same /ws protocol, fake data,
// and an animated test picture on /api/video. Run `npm run mock`, then `npm run dev:mock`.
import { createServer } from "node:http";
import { WebSocketServer } from "ws";

const PORT = Number(process.env.MOCK_PORT || 18901);
const CATEGORIES = ["Conversa", "Visão", "Áudio", "Movimento", "Ferramentas", "Sistema", "Outros"];
const STATES = ["IDLE", "LISTENING", "THINKING", "EXECUTING", "LISTENING", "IDLE"];

let seq = 0;
let tick = 0;
const state = { sleeping: false, transition: "", head_tracking: true, video_enabled: true, log_level: "INFO" };
const chat = [
  { t: Date.now() / 1000 - 60, who: "you", text: "Dobby, que horas são?" },
  { t: Date.now() / 1000 - 58, who: "dobby", text: "Hora de você parar de me perguntar isso. 21h07." },
];
const logs = [];

function log(lv, cat, msg) {
  const line = { id: ++seq, t: Date.now() / 1000, lv, cat, msg };
  logs.push(line);
  if (logs.length > 3000) logs.shift();
  return line;
}
for (let i = 0; i < 400; i++) log([10, 20, 20, 30, 40][i % 5], CATEGORIES[i % 7], `mensagem de exemplo ${i} sobre ${["câmera", "microfone", "wake", "tool", "sessão"][i % 5]}`);

function status() {
  const speaking = tick % 20 < 8;
  return {
    type: "status",
    state: state.sleeping ? "IDLE" : STATES[Math.floor(tick / 4) % STATES.length],
    sleeping: state.sleeping,
    transition: state.transition,
    model: "gemini · gemini-3.8-live",
    people: tick % 30 < 25 ? [{ id: 1, name: "Rafael" }, { id: 2, name: null }] : [],
    speaker: speaking ? "Rafael" : null,
    camera: true,
    head_tracking: state.head_tracking,
    video_enabled: state.video_enabled,
    fps: state.video_enabled ? 8 : 0,
    mic: { rms: Math.abs(Math.sin(tick / 3)) * 0.03 },
    wake: { heard: state.sleeping && tick % 10 < 4 ? "o dobby tá aí?" : null },
    services: { ha: true, oc: tick % 40 < 35, pi: { cpu_load: 0.42, mem_used_gb: 1.3, mem_total_gb: 3.8 } },
    log_level: state.log_level,
  };
}

const server = createServer((req, res) => {
  if (req.url?.startsWith("/api/video")) return video(req, res);
  res.writeHead(404).end();
});

function frame(t) {
  const x = 260 + Math.sin(t / 10) * 120;
  return `<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3a3f5c"/><stop offset="1" stop-color="#14161f"/></linearGradient></defs><rect width="960" height="540" fill="url(#g)"/><rect x="${x}" y="120" width="180" height="260" rx="10" fill="none" stroke="#7cf0c5" stroke-width="4"/><text x="${x}" y="108" fill="#7cf0c5" font-family="monospace" font-size="22">Rafael 0.91</text><circle cx="${480 + Math.cos(t / 7) * 200}" cy="480" r="14" fill="#ffc56b"/><text x="20" y="40" fill="#fff" font-family="monospace" font-size="20">MOCK ${new Date().toLocaleTimeString("pt-BR")}</text></svg>`;
}

function video(req, res) {
  res.writeHead(200, { "Content-Type": "multipart/x-mixed-replace; boundary=frame", "Cache-Control": "no-store" });
  let t = 0;
  const timer = setInterval(() => {
    const body = Buffer.from(frame(t++));
    res.write(`--frame\r\nContent-Type: image/svg+xml\r\nContent-Length: ${body.length}\r\n\r\n`);
    res.write(body);
    res.write("\r\n");
  }, 125);
  req.on("close", () => clearInterval(timer));
}

const wss = new WebSocketServer({ server, path: "/ws" });
const sendAll = (m) => wss.clients.forEach((c) => c.readyState === 1 && c.send(JSON.stringify(m)));

wss.on("connection", (ws) => {
  ws.send(JSON.stringify({ type: "hello", categories: CATEGORIES, logs: logs.slice(-800), chat, status: status() }));
  ws.on("message", (raw) => {
    const m = JSON.parse(String(raw));
    const reply = (x) => ws.send(JSON.stringify(x));
    if (m.type === "chat") {
      if (state.sleeping) return reply({ type: "error", text: "o Dobby está dormindo: acorde-o primeiro" });
      const you = { t: Date.now() / 1000, who: "you", text: m.text };
      chat.push(you);
      sendAll({ type: "chat", entry: you });
      setTimeout(() => {
        const answer = { t: Date.now() / 1000, who: "dobby", text: `Você disse “${m.text}”. Fascinante. Sério.` };
        chat.push(answer);
        sendAll({ type: "chat", entry: answer });
      }, 1200);
    } else if (m.type === "cmd") {
      state.transition = m.name;
      setTimeout(() => {
        state.sleeping = m.name === "sleep";
        state.transition = "";
      }, 2500);
    } else if (m.type === "head_tracking") state.head_tracking = !!m.enabled;
    else if (m.type === "video_enabled") state.video_enabled = !!m.enabled;
    else if (m.type === "log_level") state.log_level = m.level;
    else if (m.type === "wake_test") {
      reply({ type: "wake_state", recording: true, seconds: 3 });
      setTimeout(() => reply({ type: "wake_state", recording: false, thinking: true }), 3000);
      setTimeout(() => {
        const hit = Math.random() > 0.4;
        reply({ type: "wake_result", text: hit ? "Dobby" : "toby", hit, silent: false });
        reply({ type: "wake_state", recording: false });
      }, 4200);
    }
  });
});

setInterval(() => {
  tick++;
  sendAll(status());
  const lines = Array.from({ length: state.log_level === "DEBUG" ? 6 : 2 }, (_, i) =>
    log(state.log_level === "DEBUG" && i % 2 ? 10 : [20, 20, 30][tick % 3], CATEGORIES[(tick + i) % 7], `evento ${tick}.${i}: tudo sob controle (mais ou menos)`),
  );
  sendAll({ type: "logs", lines });
}, 1000);

server.listen(PORT, "127.0.0.1", () => console.log(`mock bridge on http://127.0.0.1:${PORT}`));
