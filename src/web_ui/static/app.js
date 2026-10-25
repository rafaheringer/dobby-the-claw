"use strict";

const $ = (id) => document.getElementById(id);
const LEVEL_NAMES = { 10: "DEBUG", 20: "INFO", 30: "AVISO", 40: "ERRO", 50: "CRÍTICO" };
const CAT_COLORS = {
  "Conversa": "#a78bfa", "Visão": "#38bdf8", "Áudio": "#34d399", "Movimento": "#fb923c",
  "Ferramentas": "#f472b6", "Sistema": "#94a3b8", "Outros": "#9ca3af",
};
const MAX_LOG_DOM = 700;
const STATE_NAMES = {
  IDLE: "Em espera", LISTENING: "Ouvindo", THINKING: "Pensando", EXECUTING: "Executando",
  DELEGATING: "Delegando", CONFIRMING: "Aguardando confirmação", ERROR: "Erro",
};

let ws = null;
let retry = 0;
let page = "vision";
let logs = [];
let paused = false;
let activeCats = new Set();
let status = null;

/* ---------- helpers ---------- */
function toast(text) {
  const el = $("toast");
  el.textContent = text;
  el.classList.add("on");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => el.classList.remove("on"), 3200);
}
function clock(ts) {
  return new Date(ts * 1000).toLocaleTimeString("pt-BR", { hour12: false });
}
function setCard(id, text, tone) {
  const el = $(id);
  el.textContent = "";
  if (tone) {
    const dot = document.createElement("span");
    dot.className = "dot " + tone;
    el.appendChild(dot);
  }
  el.appendChild(document.createTextNode(text));
}
function send(message) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message));
  else toast("Sem conexão com o Dobby");
}

/* ---------- navigation ---------- */
function show(name) {
  page = name;
  document.querySelectorAll("#nav button").forEach((b) => b.classList.toggle("on", b.dataset.page === (name === "wake" ? "settings" : name)));
  document.querySelectorAll(".page").forEach((p) => p.classList.toggle("on", p.id === "page-" + name));
  updateVideo();
  if (name === "chat") $("chat").scrollTop = $("chat").scrollHeight;
  if (name === "logs") $("logs").scrollTop = $("logs").scrollHeight;
}
document.querySelectorAll("#nav button").forEach((b) => b.addEventListener("click", () => show(b.dataset.page)));
$("open-wake").addEventListener("click", () => show("wake"));
$("wake-back").addEventListener("click", () => show("settings"));
for (const suffix of ["", "-m"]) {
  $("wake" + suffix).addEventListener("click", () => send({ type: "cmd", name: "wake" }));
  $("sleep" + suffix).addEventListener("click", () => send({ type: "cmd", name: "sleep" }));
}

/* ---------- video ---------- */
function videoWanted() {
  return page === "vision" && !document.hidden && $("t-video").checked && status && status.camera && status.video_enabled !== false;
}
// Few fixed sizes: a width that drifts with the layout would restart the stream again and again.
function streamWidth(px) {
  return px <= 700 ? 640 : px <= 1100 ? 960 : 1280;
}
function updateVideo() {
  const img = $("video");
  const frame = img.parentElement;
  if (!videoWanted()) {
    if (img.getAttribute("src")) img.removeAttribute("src");
    frame.classList.remove("live");
    $("video-msg").textContent = status && !status.camera ? "Sem câmera" : "Câmera desligada";
    return;
  }
  const q = new URLSearchParams({
    people: $("t-people").checked ? 1 : 0,
    voice: $("t-voice").checked ? 1 : 0,
    tracking: $("t-tracking").checked ? 1 : 0,
    w: streamWidth(frame.clientWidth * (window.devicePixelRatio || 1)),
  });
  const url = "/api/video?" + q;
  if (img.dataset.url !== url || !img.getAttribute("src")) {
    img.dataset.url = url;
    img.src = url;
  }
}
$("video").addEventListener("load", () => $("video").parentElement.classList.add("live"));
$("video").addEventListener("error", () => {
  $("video").parentElement.classList.remove("live");
  $("video-msg").textContent = "Aguardando imagem…";
  $("video").removeAttribute("src");
  setTimeout(updateVideo, 3000);
});
["t-people", "t-voice", "t-tracking", "t-video"].forEach((id) => $(id).addEventListener("change", () => {
  if (id === "t-video") send({ type: "video_enabled", enabled: $("t-video").checked });
  updateVideo();
}));
$("t-head").addEventListener("change", () => send({ type: "head_tracking", enabled: $("t-head").checked }));
document.addEventListener("visibilitychange", updateVideo);
window.addEventListener("resize", () => { clearTimeout(updateVideo.t); updateVideo.t = setTimeout(updateVideo, 400); });

/* ---------- status ---------- */
function renderStatus(s) {
  status = s;
  const asleep = s.sleeping;
  const label = s.transition === "sleep" ? "Indo dormir…" : s.transition === "wake" ? "Acordando…" : asleep ? "Dormindo" : (STATE_NAMES[s.state] || s.state);
  for (const suffix of ["", "-m"]) {
    $("state-text" + suffix).textContent = label;
    $("state-dot" + suffix).className = s.transition ? "warn" : asleep ? "" : "good";
    $("wake" + suffix).disabled = !!s.transition;
    $("sleep" + suffix).disabled = !!s.transition;
  }
  $("w-sleep").disabled = !!s.transition;
  setCard("c-people", s.people.length ? `${s.people.length} (${s.people.map((p) => p.name || "?").join(", ")})` : "0", s.people.length ? "accent" : "");
  setCard("c-speaker", s.speaker || "—", s.speaker ? "warn" : "");
  if (s.mic) {
    setCard("c-mic", s.mic.rms.toFixed(4), s.mic.rms > 0.002 ? "good" : "warn");
    $("b-mic").style.width = Math.min(100, (s.mic.rms / 0.05) * 100) + "%";
  } else setCard("c-mic", "—", "");
  if (s.wake) {
    setCard("c-wake", asleep ? (s.wake.heard ? `ouviu “${s.wake.heard}”` : "ouvindo") : "acordado", asleep ? "accent" : "");
    $("w-heard").textContent = s.wake.heard ? `“${s.wake.heard}”` : "Nada nos últimos segundos.";
  } else setCard("c-wake", "desligado", "");
  $("w-awake").hidden = asleep || !!s.transition;

  setCard("s-state", asleep ? "Dormindo" : (STATE_NAMES[s.state] || s.state), asleep ? "" : "good");
  setCard("s-model", s.model, "accent");
  const sv = s.services || {};
  for (const [key, id] of [["ha", "s-ha"], ["oc", "s-oc"]]) {
    if (key in sv) setCard(id, sv[key] ? "online" : "fora do ar", sv[key] ? "good" : "bad");
    else setCard(id, "desligado", "");
  }
  const pi = sv.pi;
  setCard("s-pi", pi ? `cpu ${pi.cpu_load.toFixed(1)} · ${pi.mem_used_gb.toFixed(1)}/${pi.mem_total_gb.toFixed(1)} GB` : "sem resposta", pi ? "good" : "");
  setCard("s-fps", s.fps ? `${s.fps} fps` : "parado", s.fps >= 5 ? "good" : "");

  if (document.activeElement !== $("t-head")) $("t-head").checked = s.head_tracking;
  $("t-head").disabled = !s.camera;
  if (document.activeElement !== $("t-video")) $("t-video").checked = s.video_enabled;
  $("t-debug").checked = s.log_level === "DEBUG";
  updateVideo();
}

/* ---------- chat ---------- */
function addChat(entry) {
  const box = $("chat");
  const stick = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  const el = document.createElement("div");
  el.className = "msg " + entry.who;
  const small = document.createElement("small");
  small.textContent = `${entry.who === "you" ? "Você" : "Dobby"} · ${clock(entry.t)}`;
  el.appendChild(small);
  el.appendChild(document.createTextNode(entry.text));
  box.appendChild(el);
  if (stick) box.scrollTop = box.scrollHeight;
}
$("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("chat-input").value.trim();
  if (!text) return;
  send({ type: "chat", text });
  $("chat-input").value = "";
});

/* ---------- logs ---------- */
function matcher() {
  const level = Number($("log-level").value);
  const words = $("log-search").value.toLowerCase().split(/\s+/).filter(Boolean);
  const must = words.filter((w) => !w.startsWith("!"));
  const hide = words.filter((w) => w.startsWith("!") && w.length > 1).map((w) => w.slice(1));
  return (l) => {
    if (l.lv < level || !activeCats.has(l.cat)) return false;
    const text = l.msg.toLowerCase();
    return must.every((w) => text.includes(w)) && !hide.some((w) => text.includes(w));
  };
}
function logNode(l) {
  const row = document.createElement("div");
  row.className = "logline lv" + l.lv;
  const t = document.createElement("span");
  t.className = "t";
  t.textContent = clock(l.t) + " ";
  const c = document.createElement("span");
  c.textContent = l.cat + " ";
  c.style.color = CAT_COLORS[l.cat] || "inherit";
  const m = document.createElement("span");
  m.className = "m";
  m.textContent = l.msg;
  row.append(t, c, m);
  return row;
}
function renderLogs() {
  const box = $("logs");
  const match = matcher();
  const shown = logs.filter(match).slice(-MAX_LOG_DOM);
  box.textContent = "";
  const frag = document.createDocumentFragment();
  shown.forEach((l) => frag.appendChild(logNode(l)));
  box.appendChild(frag);
  box.scrollTop = box.scrollHeight;
}
function addLogs(lines) {
  logs.push(...lines);
  if (logs.length > 4000) logs = logs.slice(-3000);
  if (paused) return;
  const box = $("logs");
  const stick = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
  const match = matcher();
  lines.filter(match).forEach((l) => box.appendChild(logNode(l)));
  while (box.childElementCount > MAX_LOG_DOM) box.removeChild(box.firstChild);
  if (stick) box.scrollTop = box.scrollHeight;
}
function buildChips(categories) {
  const holder = $("chips");
  holder.textContent = "";
  categories.forEach((name) => {
    activeCats.add(name);
    const chip = document.createElement("button");
    chip.className = "chip on";
    chip.style.setProperty("--c", CAT_COLORS[name] || "#9ca3af");
    chip.textContent = name;
    chip.addEventListener("click", () => {
      chip.classList.toggle("on");
      chip.classList.contains("on") ? activeCats.add(name) : activeCats.delete(name);
      renderLogs();
    });
    holder.appendChild(chip);
  });
}
$("log-search").addEventListener("input", renderLogs);
$("log-level").addEventListener("change", renderLogs);
$("log-clear").addEventListener("click", () => { logs = []; $("logs").textContent = ""; });
$("log-pause").addEventListener("click", () => {
  paused = !paused;
  $("log-pause").textContent = paused ? "Continuar" : "Pausar";
  if (!paused) renderLogs();
});
$("t-debug").addEventListener("change", () => send({ type: "log_level", level: $("t-debug").checked ? "DEBUG" : "INFO" }));

/* ---------- waking the robot (wakeword test) ---------- */
let recTimer = null;
let recGuard = null;
function renderRecording(m) {
  const box = $("w-rec");
  $("w-test").disabled = m.recording || !!m.thinking;
  clearInterval(recTimer);
  clearTimeout(recGuard);
  if (m.thinking) { box.hidden = false; box.textContent = "Entendendo o que você disse…"; }
  else if (!m.recording) { box.hidden = true; return; }
  // If the server never answers (dropped socket), do not stay stuck.
  recGuard = setTimeout(() => {
    renderRecording({ recording: false });
    toast("O Dobby não respondeu ao teste. Recarregue a página e tente de novo.");
  }, ((m.seconds || 0) + 15) * 1000);
  if (m.thinking) return;
  let left = m.seconds;
  const paint = () => { box.textContent = `Diga “Dobby” agora…  ${Math.ceil(left)}`; };
  box.hidden = false;
  paint();
  recTimer = setInterval(() => { left = Math.max(0, left - 0.25); paint(); }, 250);
}
function addTry(m) {
  const li = document.createElement("li");
  li.className = m.hit ? "ok" : "no";
  const icon = document.createElement("b");
  icon.textContent = m.hit ? "✓" : "✕";
  const text = document.createElement("span");
  text.textContent = m.silent ? "Não ouvi nada. Chegue mais perto ou fale mais alto."
    : m.hit ? `Ouvi “${m.text}”. Eu acordaria.` : `Ouvi “${m.text || "…"}”. Não acordaria.`;
  li.append(icon, text);
  const list = $("w-tries");
  list.prepend(li);
  while (list.children.length > 6) list.lastChild.remove();
}
$("w-sleep").addEventListener("click", () => send({ type: "cmd", name: "sleep" }));
$("w-test").addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) { toast("Sem conexão com o Dobby. Recarregue a página."); return; }
  renderRecording({ recording: true, seconds: 3 });  // instant feedback; the server confirms
  send({ type: "wake_test" });
});

/* ---------- connection ---------- */
function connect() {
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onopen = () => { retry = 0; $("conn").textContent = "conectado"; };
  ws.onclose = () => {
    $("conn").textContent = "reconectando…";
    status = status && { ...status, video_enabled: false };
    updateVideo();
    setTimeout(connect, Math.min(10000, 800 * 2 ** retry++));
  };
  ws.onmessage = (event) => {
    const m = JSON.parse(event.data);
    if (m.type === "hello") {
      if (!activeCats.size) buildChips(m.categories);
      logs = m.logs;
      renderLogs();
      $("chat").textContent = "";
      m.chat.forEach(addChat);
      renderStatus(m.status);
    } else if (m.type === "status") renderStatus(m);
    else if (m.type === "logs") addLogs(m.lines);
    else if (m.type === "chat") addChat(m.entry);
    else if (m.type === "wake_result") addTry(m);
    else if (m.type === "wake_state") renderRecording(m);
    else if (m.type === "error") toast(m.text);
  };
}
connect();
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
