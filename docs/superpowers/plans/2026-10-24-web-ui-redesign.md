# Web UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the vanilla `src/web_ui/static/` page with a React SPA (soft-glass look, real routes, PWA) that includes a mocked 3D digital twin of the Reachy Mini.

**Architecture:** A Vite + React project in `src/web_ui/frontend/` builds to `frontend/dist/`, which `server.py` serves with an SPA fallback; `/ws` and `/api/video` are unchanged. Live data flows through one WebSocket into a Zustand store via a pure reducer. One persistent react-three-fiber canvas in the root layout renders Pollen's GLB in head mode (Home) or full mode (`/twin`), driven by a `RobotLink` interface implemented by `MockRobotLink`.

**Tech Stack:** React 19, Vite 7, TypeScript 5.9 (strict), React Router 7 (`react-router`, declarative `BrowserRouter`), Tailwind CSS 4 (`@tailwindcss/vite`), Motion 12 (`motion/react`), three 0.181, @react-three/fiber 9, @react-three/drei 10, Zustand 5, `radix-ui` 1.4, @tanstack/react-virtual 3, vite-plugin-pwa 1, vite-plugin-static-copy 3, @fontsource-variable/inter 5, Vitest 4, ESLint 9 + typescript-eslint 8. Python side: aiohttp 3.14, pytest 9.

**Spec:** `docs/superpowers/specs/2026-10-24-web-ui-redesign-design.md`

## Global Constraints

- All UI copy in pt-BR, short labels (one or two words), no explanatory subtitles. Server error texts are shown verbatim.
- Must run under the CSP: `default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; connect-src 'self' ws: wss:; frame-ancestors …`. No CDNs, no inline `<script>`, fonts self-hosted.
- Server binds `127.0.0.1` only; `access.py` guard unchanged except the two CSP directives.
- WebSocket protocol and `/api/video` query parameters unchanged (see Task 3 types).
- Routes: `/`, `/conversa`, `/logs`, `/sistema`, `/sistema/nome`, `/twin`; anything else redirects to `/`.
- Phone breakpoint default; desktop layout at `≥ 1024px` (Tailwind `lg:`). Must work at 390 px width with no horizontal scroll.
- One spring preset `{ type: "spring", stiffness: 380, damping: 34 }` and one fade `{ duration: 0.15 }`, both from `app/motion.ts`. `prefers-reduced-motion` → fades only.
- Palette tokens (CSS custom properties on `:root`): accent `#9aa8ff`; states: listening `#7cf0c5`, thinking `#ffc56b`, executing/delegating `#9aa8ff`, confirming `#ffc56b`, sleeping `#8a90a8`, error `#ff7a85`; background radial gradient `#2c3266 → #161a2c → #0f111b`.
- Pollen-derived files start with a header comment naming the upstream file, repo `pollen-robotics/reachy-mini-desktop-app`, commit `5796d5b94f07cd889e955e3b9e785eafee1cac89`, and "Apache-2.0".
- Python: ruff (100-char lines), match the surrounding style of `server.py`. Python commands run with `C:/GITHUB/dobby-the-claw/.venv/Scripts/python.exe` and `PYTHONPATH=src`.
- Frontend commands run in `src/web_ui/frontend/`. `npm run build` = `tsc --noEmit && vitest run && vite build`.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

- Deep link / reload on `/logs?q=…`, `/sistema/nome`, `/twin` (opened directly, no history): page renders, and "Voltar" on `/twin` goes to `/` instead of leaving the site. → test in Task 2 (server fallback keeps query) and Task 12 (`closeTwin` with no history).
- Path traversal on the SPA fallback (`/../access.py`, `/%2e%2e/…`): must never serve files outside `dist/`. → test in Task 2.
- Socket drops while a wake test is recording: the UI must not stay stuck "Gravando". → reducer test in Task 3 (disconnect clears `recording/thinking`), plus the 15 s guard in Task 9.
- Large log volume (backlog 3000 + bursts of 1000 lines/150 ms): list stays responsive and memory bounded. → reducer cap test in Task 3; virtualized list in Task 8.
- Status arriving with `camera: false` or `mic: null`/`wake: null` (camera or wakeword disabled): no crash, placeholders shown. → reducer/selector tests in Task 3 and Task 6.

---

### Task 1: Frontend scaffold, tokens and build

**Files:**
- Create: `src/web_ui/frontend/package.json`, `vite.config.ts`, `tsconfig.json`, `eslint.config.js`, `index.html`, `src/main.tsx`, `src/app/App.tsx`, `src/app/motion.ts`, `src/styles.css`, `src/vite-env.d.ts`, `src/app/motion.test.ts`
- Move: `src/web_ui/static/icon*.png`, `icon.svg` → `src/web_ui/frontend/public/`
- Modify: `.gitignore` (add `src/web_ui/frontend/node_modules/`, `src/web_ui/frontend/dist/`)

**Interfaces:**
- Produces: `spring`, `fade`, `useMotionPreset(): { spring: Transition; fade: Transition; reduced: boolean }` in `src/app/motion.ts` (reduced → spring replaced by fade). CSS tokens listed in Global Constraints as `--accent`, `--st-listening`, `--st-thinking`, `--st-executing`, `--st-confirming`, `--st-sleeping`, `--st-error`, `--glass-bg: rgba(255,255,255,.07)`, `--glass-border: rgba(255,255,255,.12)`, `--text`, `--text-dim: #aab0d6`, exposed to Tailwind via `@theme`.

- [ ] **Step 1:** `npm create vite@7 frontend -- --template react-ts` inside `src/web_ui/`, then install the stack packages from the header (runtime deps and dev deps) and `@vitejs/plugin-react`. Commit `package-lock.json`.
- [ ] **Step 2: Write the failing test** `src/app/motion.test.ts`: `spring` deep-equals `{ type: "spring", stiffness: 380, damping: 34 }`; `fade` deep-equals `{ duration: 0.15 }`.
- [ ] **Step 3:** Run `npx vitest run src/app/motion.test.ts` → FAIL (module missing).
- [ ] **Step 4:** Implement `motion.ts` (`useMotionPreset` uses Motion's `useReducedMotion`).
- [ ] **Step 5:** Configure `vite.config.ts`: react plugin, `@tailwindcss/vite`, `server.proxy` `{ "/ws": { target: "ws://127.0.0.1:18900", ws: true }, "/api": "http://127.0.0.1:18900" }`, `viteStaticCopy` of `node_modules/three/examples/jsm/libs/draco/gltf/*` to `draco/`, `VitePWA` with `registerType: "autoUpdate"`, manifest (`name`/`short_name` "Dobby", `lang: "pt-BR"`, `display: "standalone"`, `start_url: "/"`, `background_color`/`theme_color` `#0f111b`, the three icons with the same sizes/purposes as the old manifest), workbox `globPatterns: ["**/*.{js,css,html,svg,png,woff2,glb,wasm}"]`, `maximumFileSizeToCacheInBytes: 6_000_000`, `navigateFallback: "/index.html"`, `navigateFallbackDenylist: [/^\/ws/, /^\/api\//]`. Vitest `environment: "node"` default.
- [ ] **Step 6:** `styles.css`: `@import "tailwindcss"`, `@import "@fontsource-variable/inter"`, `:root` tokens, `@theme` mapping, body background gradient (fixed), `color-scheme: dark`, `-webkit-tap-highlight-color: transparent`, safe-area padding helpers. `index.html`: `lang="pt-BR"`, viewport with `viewport-fit=cover`, `theme-color #0f111b`, apple-touch-icon, title "Dobby". `App.tsx` renders "Dobby" for now.
- [ ] **Step 7:** `package.json` scripts: `dev: "vite"`, `build: "tsc --noEmit && vitest run && vite build"`, `test: "vitest"`, `lint: "eslint ."`. Run `npm run build` → PASS; `dist/index.html`, `dist/sw.js`, `dist/manifest.webmanifest`, `dist/draco/draco_decoder.wasm` exist.
- [ ] **Step 8:** Commit `feat(web-ui): scaffold React frontend with tokens and PWA build`.

### Task 2: Server serves the SPA; CSP gains WASM/worker directives

**Files:**
- Create: `src/web_ui/frontend_routes.py`, `tests/web_ui/test_frontend_routes.py`, `tests/conftest.py` (adds `src` to `sys.path` if not set)
- Modify: `src/web_ui/server.py` (remove `_STATIC`, `_index`, `_manifest`, `_service_worker`, `revalidate_static`, the `/static` route; call `add_frontend_routes`), `src/web_ui/access.py:110-114` (CSP)
- Delete: `src/web_ui/static/` (remaining files)

**Interfaces:**
- Produces: `FRONTEND_DIST: Path` (= `Path(__file__).parent / "frontend" / "dist"`), `add_frontend_routes(app: web.Application, dist: Path) -> None` — registers a catch-all `GET /{tail:.*}` **after** `/ws` and `/api/video`; `warn_if_missing(dist: Path) -> None` (logs one warning).

- [ ] **Step 1: Write the failing tests** (aiohttp `TestClient`/`TestServer` from `aiohttp.test_utils`, inside `asyncio.run` or `pytest`'s `async` via a small helper; a `tmp_path` dist with `index.html` = `"<html>app</html>"`, `assets/app-abc123.js`, `sw.js`, `manifest.webmanifest`):
  - `test_spa_route_returns_index`: `GET /logs?q=x` and `GET /sistema/nome` → 200, body `<html>app</html>`, `Cache-Control: no-cache`.
  - `test_hashed_asset_is_immutable`: `GET /assets/app-abc123.js` → 200, `Cache-Control: public, max-age=31536000, immutable`.
  - `test_root_files_no_cache`: `GET /sw.js` → 200, `no-cache`; `GET /manifest.webmanifest` → `Content-Type` `application/manifest+json`.
  - `test_missing_asset_is_404`: `GET /assets/nope.js` → 404 (no index fallback under `/assets/`).
  - `test_traversal_is_refused`: `GET /../frontend_routes.py`, `GET /%2e%2e/secret.txt` (with `tmp_path.parent / "secret.txt"` present) → never returns that file's content (200 with index body or 404).
  - `test_missing_dist_explains_build`: dist dir absent → `GET /` 503 with body containing `npm --prefix src/web_ui/frontend ci` and `run build`.
  - `test_csp_allows_wasm_and_blob_workers`: app built with `make_guard(AccessVerifier(AccessPolicy(team_domain="", audience="", allowed_emails=(), frame_ancestors=())))` and a dummy handler; local `GET /` → CSP contains `script-src 'self' 'wasm-unsafe-eval'` and `worker-src 'self' blob:`, and still `default-src 'self'`. (Check `AccessPolicy` field names/types in `access.py` before writing.)
- [ ] **Step 2:** Run `PYTHONPATH=src C:/GITHUB/dobby-the-claw/.venv/Scripts/python.exe -m pytest tests/web_ui -q` → FAIL.
- [ ] **Step 3:** Implement `frontend_routes.py`: resolve `(dist / tail).resolve()` and require `is_relative_to(dist.resolve())` and `is_file()`; `assets/` prefix → immutable header or 404; other existing files → `no-cache` (`.webmanifest` content type set explicitly); otherwise `index.html` with `no-cache`; dist missing → 503 text/html in pt-BR naming both npm commands. Edit the CSP string in `access.py`. Wire into `server.py` `build_app` and call `warn_if_missing(FRONTEND_DIST)` in `start_web_ui`; delete `static/`.
- [ ] **Step 4:** Run the pytest command → PASS; `ruff check src/web_ui tests` → clean.
- [ ] **Step 5:** Commit `feat(web-ui): serve the built SPA with history fallback`.

### Task 3: Live data — protocol types, reducer, socket client, store

**Files:**
- Create: `src/live/protocol.ts`, `src/live/reducer.ts`, `src/live/reducer.test.ts`, `src/live/socket.ts`, `src/live/socket.test.ts`, `src/live/store.ts`

**Interfaces:**
- Produces (`protocol.ts`), mirroring `server.py`:

```ts
export type RobotState = "IDLE" | "LISTENING" | "THINKING" | "EXECUTING" | "DELEGATING" | "CONFIRMING" | "ERROR";
export interface PiStats { cpu_load: number; mem_used_gb: number; mem_total_gb: number }
export interface Status {
  state: RobotState; sleeping: boolean; transition: "" | "sleep" | "wake"; model: string;
  people: { id: number | string; name: string | null }[]; speaker: string | null;
  camera: boolean; head_tracking: boolean; video_enabled: boolean; fps: number;
  mic: { rms: number } | null; wake: { heard: string | null } | null;
  services: { ha?: boolean | null; oc?: boolean | null; pi?: PiStats | null }; log_level: string;
}
export interface LogLine { id: number; t: number; lv: number; cat: string; msg: string }
export interface ChatEntry { t: number; who: "dobby" | "you"; text: string }
export type ServerMessage =
  | { type: "hello"; categories: string[]; logs: LogLine[]; chat: ChatEntry[]; status: Status }
  | ({ type: "status" } & Status)
  | { type: "logs"; lines: LogLine[] } | { type: "chat"; entry: ChatEntry }
  | { type: "wake_state"; recording: boolean; thinking?: boolean; seconds?: number }
  | { type: "wake_result"; text: string; hit: boolean; silent: boolean }
  | { type: "error"; text: string };
export type ClientMessage =
  | { type: "chat"; text: string } | { type: "cmd"; name: "wake" | "sleep" }
  | { type: "head_tracking"; enabled: boolean } | { type: "video_enabled"; enabled: boolean }
  | { type: "log_level"; level: "DEBUG" | "INFO" } | { type: "wake_test" };
```

- Produces (`reducer.ts`): `LiveState { connection: "connecting" | "open" | "closed"; status: Status | null; categories: string[]; logs: LogLine[]; chat: ChatEntry[]; wakeTest: { recording: boolean; thinking: boolean; seconds: number; tries: WakeTry[] }; toasts: { id: number; text: string }[] }`, `WakeTry = { text: string; hit: boolean; silent: boolean; t: number }`, `initialLive: LiveState`, `reduce(s: LiveState, m: ServerMessage, now?: number): LiveState`, `onConnection(s: LiveState, c: LiveState["connection"]): LiveState`, `MAX_LOGS = 4000`, `MAX_CHAT = 500`, `MAX_TRIES = 6`.
- Produces (`socket.ts`): `backoffMs(retry: number): number` (= `min(10000, 800 * 2 ** retry)`), `connectLive(url: string): { send(m: ClientMessage): boolean; close(): void }` dispatching into the store.
- Produces (`store.ts`): `useLive` (Zustand store of `LiveState`), `send(m: ClientMessage): void` (toast "Sem conexão com o Dobby" when not open), `pushToast(text: string)`, `dismissToast(id: number)`, `startLive(): void` (called once from `main.tsx`, URL from `location`).

- [ ] **Step 1: Write the failing tests** `reducer.test.ts`:
  - `hello replaces logs, chat, status and categories` (pre-existing logs are dropped).
  - `logs append and cap at MAX_LOGS` (push 4500 lines → length 4000, last id kept).
  - `chat appends and caps at MAX_CHAT`.
  - `status message replaces status without the type field`.
  - `wake_state recording true sets recording and seconds; thinking true clears recording`.
  - `wake_result prepends a try and keeps MAX_TRIES` (7 results → 6, newest first) and clears recording/thinking.
  - `error pushes a toast with the server text and an increasing id`.
  - `onConnection("closed") clears wake recording and thinking` (Review Focus).
  - `status with camera false and mic/wake null is stored as is`.
  `socket.test.ts`: `backoffMs(0) === 800`, `backoffMs(2) === 3200`, `backoffMs(10) === 10000`.
- [ ] **Step 2:** `npx vitest run src/live` → FAIL.
- [ ] **Step 3:** Implement `protocol.ts`, `reducer.ts` (pure, immutable updates), `socket.ts` (WebSocket with `wss:` when `https:`, `onopen` → `onConnection("open")`, `onclose` → `"closed"` + reconnect after `backoffMs(retry++)`, retry reset on open), `store.ts`.
- [ ] **Step 4:** `npx vitest run src/live` → PASS.
- [ ] **Step 5:** Commit `feat(web-ui): live socket, reducer and store`.

### Task 4: Pure helpers — log filter, URL params, camera URL

**Files:**
- Create: `src/lib/logFilter.ts`, `src/lib/logFilter.test.ts`, `src/lib/video.ts`, `src/lib/video.test.ts`

**Interfaces:**
- Produces: `LogLevel = 10 | 20 | 30 | 40`; `LogFilter { q: string; level: LogLevel; cats: string[] | null }` (`null` = all); `matchLog(f: LogFilter): (l: LogLine) => boolean`; `parseLogParams(sp: URLSearchParams): LogFilter` and `toLogParams(f: LogFilter): URLSearchParams` with params `q`, `nivel` ∈ `debug|info|aviso|erro` (default `info` = 20, omitted when default), `cat` = comma-separated enabled categories (omitted when `null`).
- Produces: `streamWidth(px: number): 640 | 960 | 1280` (≤700 → 640, ≤1100 → 960, else 1280); `videoUrl(o: { people: boolean; voice: boolean; tracking: boolean; width: number }): string` → `/api/video?people=1&voice=0&tracking=1&w=960`.

- [ ] **Step 1: Write the failing tests:** `matchLog` — level below threshold hidden; category not in `cats` hidden; `cats: null` shows all; `"camera !erro"` keeps lines containing "camera" and hides those containing "erro" (case-insensitive); a lone `"!"` is ignored. Params round trip: `toLogParams(parseLogParams(new URLSearchParams("q=a&nivel=erro&cat=Visão,Áudio")))` gives back the same `q`, `nivel`, `cat`; defaults produce an empty string; unknown `nivel` → 20. `streamWidth(700) === 640`, `streamWidth(701) === 960`, `streamWidth(1101) === 1280`; `videoUrl` exact string above.
- [ ] **Step 2:** `npx vitest run src/lib` → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** `npx vitest run src/lib` → PASS.
- [ ] **Step 5:** Commit `feat(web-ui): log filter, URL params and video helpers`.

### Task 5: UI kit and shell (routing, navigation, state pill, toasts, transitions)

**Files:**
- Create: `src/ui/Glass.tsx` (`Card`, `Panel`), `src/ui/Button.tsx`, `src/ui/Pill.tsx`, `src/ui/Switch.tsx`, `src/ui/Toasts.tsx`, `src/ui/Menu.tsx` (Radix dropdown), `src/app/routes.ts`, `src/app/routes.test.ts`, `src/app/Layout.tsx`, `src/app/TabBar.tsx`, `src/app/Sidebar.tsx`, `src/app/StatePill.tsx`, `src/app/WakeToggle.tsx`, `src/app/stateLabel.ts`, `src/app/stateLabel.test.ts`, placeholder `src/routes/{Home,Chat,Logs,System,SystemWake,Twin}.tsx`
- Modify: `src/app/App.tsx`, `src/main.tsx`

**Interfaces:**
- Consumes: `useLive`, `send` (Task 3); `useMotionPreset` (Task 1).
- Produces: `SECTIONS = [{ path: "/", label: "Início" }, { path: "/conversa", label: "Conversa" }, { path: "/logs", label: "Logs" }, { path: "/sistema", label: "Sistema" }]`; `sectionIndex(pathname: string): number` (`/sistema/nome` → 3, `/twin` → -1, unknown → -1); `stateLabel(s: Status | null, connection): { text: string; tone: "listening" | "thinking" | "executing" | "confirming" | "sleeping" | "error" | "offline"; pulse: boolean }` with texts: connection not open → "reconectando…" (offline); `transition` sleep → "Indo dormir…", wake → "Acordando…"; sleeping → "Dormindo"; states IDLE "Em espera", LISTENING "Ouvindo", THINKING "Pensando", EXECUTING "Executando", DELEGATING "Delegando", CONFIRMING "Confirmando", ERROR "Erro". `Card`/`Panel` props `{ className?: string; children }`; `Button` variants `"glass" | "primary"`, `busy?: boolean`.

- [ ] **Step 1: Write the failing tests:** `routes.test.ts` (`sectionIndex` cases above); `stateLabel.test.ts` (each mapping above, transition wins over sleeping, offline wins over all).
- [ ] **Step 2:** `npx vitest run src/app` → FAIL.
- [ ] **Step 3:** Implement `routes.ts`, `stateLabel.ts`, then the components. `App.tsx`: `BrowserRouter` → `Routes` with `Layout` as parent route and children for each path, `*` → `<Navigate to="/" replace />`. `Layout`: `Sidebar` (lg) / header + `TabBar` (mobile, floating glass, active indicator `motion.span layoutId="tab-indicator"`), `<AnimatePresence mode="popLayout">` keyed by section with x offset `±24px` based on `sectionIndex` direction (fade only when reduced), `Toasts`, and a slot for the twin stage (Task 11). `WakeToggle`: one button, label "Dormir" when awake / "Acordar" when sleeping, spinner + disabled during `transition` or when not connected. `main.tsx`: `startLive()`, register SW via `virtual:pwa-register`.
- [ ] **Step 4:** `npx vitest run src/app` → PASS; `npm run build` → PASS.
- [ ] **Step 5: Manual:** `npm run dev` with the bridge running: tabs switch with URL change; reload on each route keeps the route; browser back/forward walks the history; 390 px shows tab bar, 1280 px shows sidebar.
- [ ] **Step 6:** Commit `feat(web-ui): glass UI kit, shell and routes`.

### Task 6: Home screen (camera card, indicators)

**Files:**
- Create: `src/routes/Home.tsx`, `src/home/CameraCard.tsx`, `src/home/Indicators.tsx`, `src/home/MicMeter.tsx`, `src/home/indicators.ts`, `src/home/indicators.test.ts`, `src/twin/TwinSlot.tsx` (empty measured box, filled in Task 11)

**Interfaces:**
- Consumes: `videoUrl`, `streamWidth` (Task 4); `useLive` (Task 3); `Card`, `Menu` (Task 5).
- Produces: `indicatorsFrom(s: Status): { people: string; speaker: string; mic: number | null; heard: string | null }` — `people`: `"0"` or `"2 · Ana, ?"` (unknown names as `?`), `speaker`: name or `"—"`, `mic`: `min(1, rms / 0.05)` or `null` when `mic` is null, `heard`: `wake.heard` only while `sleeping`; `<TwinSlot className />` (a div with `data-twin-slot`, aspect square on phone).

- [ ] **Step 1: Write the failing test** `indicators.test.ts` for each rule above, including `mic: null` and `wake: null`.
- [ ] **Step 2:** `npx vitest run src/home` → FAIL.
- [ ] **Step 3:** Implement. `CameraCard`: `<img>` src from `videoUrl` only when on `/`, `document.visibilityState === "visible"`, `status.camera && status.video_enabled` and connection open; width from `streamWidth(card.clientWidth * devicePixelRatio)` recomputed on resize (debounced 400 ms); on `error` remove src, show "Aguardando imagem…", retry after 3000 ms; placeholders "Sem câmera" (`camera: false`) / "Câmera desligada"; options menu (Radix dropdown, icon button) with Pessoas / Voz / Rastreio checkboxes stored in `localStorage` (try/catch). Layout: phone = TwinSlot, CameraCard, indicator row (numbers animate on change with `motion` + `spring`); `lg:` grid `grid-cols-[1.4fr_1fr]` with camera spanning two rows, TwinSlot, and the last 4 chat messages.
- [ ] **Step 4:** `npx vitest run src/home` → PASS.
- [ ] **Step 5: Manual:** camera streams on `/`, stops (no `/api/video` request in DevTools) on other tabs and when the tab is hidden; toggling overlays restarts the stream with the new query.
- [ ] **Step 6:** Commit `feat(web-ui): home screen with camera and indicators`.

### Task 7: Chat screen

**Files:**
- Create: `src/routes/Chat.tsx`, `src/chat/Bubble.tsx`, `src/chat/Composer.tsx`, `src/chat/useStickToBottom.ts`

**Interfaces:**
- Consumes: `useLive` (`chat`, `status.sleeping`), `send` (Task 3); `spring` (Task 1).

- [ ] **Step 1:** Implement bubbles (Você right/accent, Dobby left/glass, time `HH:MM` pt-BR), `motion.div` entrance with `spring` (y 12 → 0, opacity), stick-to-bottom only when the user is within 60 px of the bottom. `Composer`: pinned bottom, `maxLength 2000`, `enterKeyHint="send"`, sends `{type:"chat", text}` trimmed, clears on send; keeps above the phone keyboard using `window.visualViewport` (`resize` → bottom offset); while `sleeping` replaced by "Dobby está dormindo" + `WakeToggle`.
- [ ] **Step 2:** `npm run build` → PASS.
- [ ] **Step 3: Manual:** at 390 px in Chrome device mode, typing a message sends it and the reply bubble appears; the input stays visible with the on-screen keyboard (Android Chrome or iOS Safari through the tunnel).
- [ ] **Step 4:** Commit `feat(web-ui): chat screen`.

### Task 8: Logs screen

**Files:**
- Create: `src/routes/Logs.tsx`, `src/logs/LogToolbar.tsx`, `src/logs/LogList.tsx`, `src/logs/categoryColors.ts`

**Interfaces:**
- Consumes: `matchLog`, `parseLogParams`, `toLogParams` (Task 4); `useLive` (`logs`, `categories`, `status.log_level`), `send` (Task 3).
- Produces: `CATEGORY_COLORS` (same map as the old `app.js`: Conversa `#a78bfa`, Visão `#38bdf8`, Áudio `#34d399`, Movimento `#fb923c`, Ferramentas `#f472b6`, Sistema `#94a3b8`, Outros `#9ca3af`).

- [ ] **Step 1:** Implement. Filter state comes from `useSearchParams` (`parseLogParams`), updates with `setSearchParams(toLogParams(f), { replace: true })` (search typing debounced 250 ms). Toolbar: search ("Buscar  ·  !termo esconde" placeholder), level select (Debug/Info/Avisos/Erros), category chips (toggle), Pausar/Continuar (frozen snapshot), Limpar (client-side: hides lines with `id ≤` the last id at clear time). `LogList`: `@tanstack/react-virtual`, monospace rows `HH:MM:SS  Categoria  mensagem`, level color for ≥30, sticks to bottom when at bottom.
- [ ] **Step 2:** `npm run build` → PASS.
- [ ] **Step 3: Manual:** set `?q=wake&nivel=debug`, reload → filters restored; back after changing filters returns to the previous route (replace keeps history clean); with server Debug on (Task 9 switch or `BRIDGE_LOG_LEVEL=DEBUG`), scrolling stays smooth with ~4000 lines.
- [ ] **Step 4:** Commit `feat(web-ui): logs screen with URL filters`.

### Task 9: System and wake-word test screens

**Files:**
- Create: `src/routes/System.tsx`, `src/routes/SystemWake.tsx`, `src/system/ServiceRow.tsx`, `src/system/CountdownRing.tsx`, `src/system/services.ts`, `src/system/services.test.ts`

**Interfaces:**
- Consumes: `useLive` (`status`, `wakeTest`), `send` (Task 3); `Switch`, `Card`, `Button` (Task 5).
- Produces: `serviceRows(s: Status): { key: string; label: string; value: string; tone: "good" | "bad" | "off" }[]` — rows: Estado (state label text), Modelo (`model`), Home Assistant / OpenClaw (`true` → "online"/good, `false|null` → "fora do ar"/bad, key absent → "desligado"/off), Raspberry Pi (`cpu 0.4 · 1.2/3.8 GB` good, null/absent → "sem resposta"/off), Câmera (`8 fps` good when ≥5, `0` → "parada"/off).

- [ ] **Step 1: Write the failing test** `services.test.ts` covering each row rule.
- [ ] **Step 2:** `npx vitest run src/system` → FAIL.
- [ ] **Step 3:** Implement (every switch and button that sends is disabled while the connection is not open). System: "Serviços" card with rows; "Controles" card with switches Cabeça segue (`head_tracking`, disabled without camera), Câmera ligada (`video_enabled`), Log em Debug (`log_level === "DEBUG"`; sends `log_level` DEBUG/INFO); link row "Teste do nome" → `/sistema/nome`. SystemWake: back link "Sistema"; when awake and no transition, notice "Ele está acordado e pode responder." + button "Colocar para dormir"; big "Testar" button (disabled while recording/thinking or disconnected) → optimistic `recording: true, seconds: 3` then `send({type:"wake_test"})`; `CountdownRing` (SVG stroke-dashoffset animated over `seconds`) with "Diga “Dobby”"; thinking → "Entendendo…"; 15 s guard (`seconds + 15` s) clears the state and toasts "O Dobby não respondeu ao teste. Tente de novo."; tries list ✓/✕ with "Ouvi “…”. Eu acordaria." / "Ouvi “…”. Não acordaria." / "Não ouvi nada. Chegue mais perto ou fale mais alto."; live "Ouvido enquanto dorme" line from `status.wake.heard`.
- [ ] **Step 4:** `npx vitest run src/system` → PASS.
- [ ] **Step 5: Manual:** each switch changes the server state (reflected after the next status tick); a wake test records, shows the ring, and adds a try.
- [ ] **Step 6:** Commit `feat(web-ui): system and wake-word test screens`.

### Task 10: RobotLink and MockRobotLink

**Files:**
- Create: `src/robot/types.ts`, `src/robot/pose.ts`, `src/robot/pose.test.ts`, `src/robot/emotions.ts`, `src/robot/emotions.test.ts`, `src/robot/mock.ts`, `src/robot/mock.test.ts`, `src/robot/index.ts`

**Interfaces:**
- Produces (`types.ts`): `RobotPose { headPose: number[] /*16, row-major*/; antennas: [number, number]; yawBody: number }`, `RobotTarget { x; y; z; roll; pitch; yaw; antennas: [number, number]; yawBody }` (metres, radians), `EmotionName = "feliz" | "curioso" | "sono" | "surpreso"`, `RobotLink { usePose(): RobotPose & { mode: "auto" | "manual"; simulated: boolean }; useTarget(): RobotTarget; setManual(on: boolean): void; setTarget(t: Partial<RobotTarget>): void; center(): void; playEmotion(name: EmotionName): void }`.
- Produces (`pose.ts`): `ZERO_TARGET`, `LIMITS = { x: [-0.02, 0.02], y: [-0.02, 0.02], z: [-0.04, 0.025], roll: [-0.5, 0.5], pitch: [-0.6, 0.6], yaw: [-1.1, 1.1], antenna: [-2.5, 2.5], yawBody: [-2.8, 2.8] }`, `clampTarget(t: RobotTarget): RobotTarget`, `poseMatrix(t: RobotTarget): number[]` (row-major 4×4, R = Rz(yaw)·Ry(pitch)·Rx(roll), translation in indices 3, 7, 11), `addTargets(a: RobotTarget, b: Partial<RobotTarget>): RobotTarget`, `idleOffset(tSec: number): Partial<RobotTarget>` (z ±0.003 at 0.25 Hz, pitch ±0.03, antennas ±0.25 out of phase at 0.45 Hz).
- Produces (`emotions.ts`): `EMOTIONS: Record<EmotionName | "atento", { duration: number; keys: { at: number; target: Partial<RobotTarget> }[] }>`, `sampleEmotion(name, tSec): Partial<RobotTarget> | null` (linear interpolation between keys; `null` after `duration`).
- Produces (`mock.ts`): `createMockRobotLink(opts?: { now?: () => number }): RobotLink & { tick(): RobotPose; onSpeaker(name: string | null): void }`; `index.ts` exports `robotLink` (the singleton mock, subscribed to `useLive` `status.speaker`) — the one place a real link replaces later.

- [ ] **Step 1: Write the failing tests:**
  - `pose.test.ts`: `poseMatrix(ZERO_TARGET)` equals identity; translation `x=0.01,z=-0.02` → `m[3]===0.01`, `m[11]===-0.02`; `yaw=π/2` → `m[0]≈0`, `m[1]≈-1`, `m[4]≈1`; `clampTarget` clamps every field to `LIMITS` (antennas per element); `idleOffset` stays within `|z|≤0.003`, `|pitch|≤0.03`, `|antenna|≤0.25` for t in 0..20 s step 0.1.
  - `emotions.test.ts`: every emotion's first key `at===0`, last key `at===duration`; `sampleEmotion("feliz", 0)` equals first key target; `sampleEmotion("feliz", duration + 0.01) === null`; midpoint interpolates.
  - `mock.test.ts` (fake `now`): auto mode `tick()` differs over time (idle motion); `setTarget({yaw: 5})` switches to manual and clamps yaw to 1.1; `center()` returns target to `ZERO_TARGET` but stays manual; `setManual(false)` back to auto; `onSpeaker("Ana")` then `onSpeaker("Ana")` plays "atento" once; `playEmotion` overrides for its duration then returns to the base pose; `usePose().simulated === true`.
- [ ] **Step 2:** `npx vitest run src/robot` → FAIL.
- [ ] **Step 3:** Implement (pose stored in a small Zustand store updated by `tick()`; `tick` is driven by the 3D frame loop in Task 11, at most once per frame).
- [ ] **Step 4:** `npx vitest run src/robot` → PASS.
- [ ] **Step 5:** Commit `feat(web-ui): RobotLink interface and mock`.

### Task 11: Persistent 3D stage with Pollen's robot (head mode, full mode, fallback)

**Files:**
- Create: `src/assets/robot/reachy_mini_viz.glb` (from Pollen at the pinned commit, via `https://media.githubusercontent.com/media/pollen-robotics/reachy-mini-desktop-app/5796d5b94f07cd889e955e3b9e785eafee1cac89/src/assets/robot-3d/reachy_mini_viz.glb`, 2 980 704 bytes), `src/assets/robot/NOTICE.md`, `src/twin/GLTFRobot.tsx` (adapted), `src/twin/headOnly.ts`, `src/twin/headOnly.test.ts`, `src/twin/framing.ts`, `src/twin/framing.test.ts`, `src/twin/TwinStage.tsx`, `src/twin/Scene.tsx`, `src/twin/HeadIllustration.tsx`, `src/twin/webgl.ts`
- Modify: `src/app/Layout.tsx` (mount `TwinStage` once), `src/twin/TwinSlot.tsx`

**Interfaces:**
- Consumes: `robotLink` (Task 10); `spring`, `useMotionPreset` (Task 1); `TwinSlot` (Task 6).
- Produces: `applyHeadOnly(root: THREE.Object3D, on: boolean, headName = "core001"): void` (visible = descendant of the node whose normalized name is `headName`, for every `Mesh`/`SkinnedMesh`); `StageMode = "head" | "full" | "hidden"`, `stageModeFor(pathname: string): StageMode` (`/` → head, `/twin` → full, else hidden); `CAMERA: Record<"head" | "full", { position: [number, number, number]; target: [number, number, number]; fov: number }>` (initial values to tune visually: head `{ position: [0, 0.23, 0.32], target: [0, 0.2, 0], fov: 30 }`, full `{ position: [0, 0.2, 0.62], target: [0, 0.12, 0], fov: 35 }`); `hasWebGL(): boolean`; `TwinSlot` registers its element in a tiny store `useTwinSlot` so the stage can measure it.

- [ ] **Step 1: Write the failing tests:** `headOnly.test.ts` — build a THREE tree `Armature > [Cable(SkinnedMesh), GLOBAL > Base > Core > [BodyMesh, Core.001 > [HeadMesh, Antenna.L.002 > AntMesh]]]` (`Core.001` named with the dot); `applyHeadOnly(root, true)` → HeadMesh, AntMesh visible, BodyMesh and Cable hidden; `applyHeadOnly(root, false)` → all visible. `framing.test.ts` — `stageModeFor` for `/`, `/twin`, `/logs`, `/sistema/nome`.
- [ ] **Step 2:** `npx vitest run src/twin` → FAIL.
- [ ] **Step 3:** Copy the GLB; write `NOTICE.md` (source URL, commit, Apache-2.0, "© Pollen Robotics"). Adapt `GLTFRobot.tsx` from upstream `src/components/viewer3d/GLTFRobot.tsx` with the license header: import GLB via `?url` from our assets; `DRACO_PATH = "/draco/"`; props `{ pose: RobotPose; headOnly: boolean }`; keep the constants, bone names, leg IK and smoothing as upstream; call `applyHeadOnly(model, headOnly)` in an effect on `[model, headOnly]`. Implement `headOnly.ts`, `framing.ts`, `webgl.ts`.
- [ ] **Step 4:** `npx vitest run src/twin` → PASS.
- [ ] **Step 5:** Implement `TwinStage`: fixed-position `motion.div` (z above page content, below tab bar in head mode; full screen in full mode) whose `top/left/width/height` animate with `spring` to the measured `TwinSlot` rect (ResizeObserver + scroll/resize listeners) in head mode or to the viewport in full mode; hidden mode → opacity 0 and `pointer-events: none`, canvas `frameloop="never"`. `<Canvas dpr={[1, 2]} gl={{ alpha: true, antialias: true }} frameloop={visible && !document.hidden ? "always" : "never"}>` with `Scene`: ambient + key light + accent rim light, drei `ContactShadows` (full mode only), `Environment` built from local `Lightformer`s (no HDR download), camera lerped toward `CAMERA[mode]` each frame (reduced motion → snap), `<Suspense>` + error boundary around `GLTFRobot` (`pose` from `robotLink.tick()` inside `useFrame`). In head mode the canvas is clickable → `navigate("/twin")`; an `IntersectionObserver` on the slot pauses rendering when the slot is off screen. Fallback: `!hasWebGL()` or GLB error → `HeadIllustration` (SVG head with antennas, ported from the brainstorm mockup) in the slot.
- [ ] **Step 6:** `npm run build` → PASS; check `dist/` has the GLB under `assets/` and `draco/draco_decoder.wasm`.
- [ ] **Step 7: Manual (bridge running, so the real CSP applies):** on `/` the head renders (no body/legs/cables), breathes and moves its antennas; no CSP errors in the console; switching to `/logs` hides the canvas (GPU idle in DevTools Performance); with WebGL disabled (`chrome://flags` or `--disable-webgl`) the 2D head appears.
- [ ] **Step 8:** Commit `feat(web-ui): persistent 3D stage with Pollen's Reachy Mini model`.

### Task 12: Twin route — transition, controls, keyboard, close

**Files:**
- Create: `src/routes/Twin.tsx`, `src/twin/controls/Joystick.tsx`, `src/twin/controls/CircularSlider.tsx`, `src/twin/controls/LinearSlider.tsx`, `src/twin/controls/EmotionMenu.tsx`, `src/twin/keys.ts`, `src/twin/keys.test.ts`, `src/twin/close.ts`, `src/twin/close.test.ts`
- Modify: `src/app/Layout.tsx` (blur/fade page content when on `/twin`)

**Interfaces:**
- Consumes: `robotLink`, `RobotTarget`, `LIMITS` (Task 10); `TwinStage` full mode (Task 11); `spring` (Task 1).
- Produces: `keyDelta(key: string): Partial<Record<"yaw" | "pitch" | "roll" | "z", number>> | null` — ArrowLeft/a → yaw +0.05, ArrowRight/d → yaw −0.05, ArrowUp/w → pitch −0.05, ArrowDown/s → pitch +0.05, q → roll −0.05, e → roll +0.05, r → z +0.002, f → z −0.002, others → null (case-insensitive letters); `closeTwin(navigate, historyIdx: number): void` — `historyIdx > 0` → `navigate(-1)`, else `navigate("/", { replace: true })` (`historyIdx` from `window.history.state?.idx`).

- [ ] **Step 1: Write the failing tests:** `keys.test.ts` for every mapping plus `"A"` behaving like `"a"` and `"x"` → null; `close.test.ts` with a spy `navigate`: idx 3 → called with -1; idx 0 and undefined → called with `"/", { replace: true }`.
- [ ] **Step 2:** `npx vitest run src/twin` → FAIL.
- [ ] **Step 3:** Implement `keys.ts`, `close.ts`, then the controls (Radix Slider for linear ones, pointer-event based Joystick and CircularSlider, all glass-styled, `touch-action: none`, values mapped from `LIMITS`; any interaction calls `robotLink.setTarget`, which switches to manual). `Twin.tsx` overlay over the full-mode stage: top bar with "‹ Voltar" (`closeTwin`), pill "simulado", Manual/Auto toggle; controls panel at the bottom (phone) / right side (`lg:`): Joystick (yaw/pitch), vertical LinearSlider (z), horizontal LinearSlider (roll), two CircularSliders (antena esq./dir.), CircularSlider (corpo), buttons "Centralizar" and "Emoções" (EmotionMenu with feliz, curioso, sono, surpreso). Entrance: controls `motion` with `staggerChildren: 0.04` after the stage reaches full size; exit reverses. Phone swipe down on the top bar (`drag="y"`, `dragConstraints {top:0,bottom:0}`, offset > 120 px) → `closeTwin`. `keydown` listener (ignored when focus is in an input) applies `keyDelta` to the current target. Layout: page content gets `filter: blur(12px)` + opacity 0 with `spring` while on `/twin`.
- [ ] **Step 4:** `npx vitest run src/twin` → PASS; `npm run build` → PASS.
- [ ] **Step 5: Manual:** tap the head on `/` → canvas grows to full screen, body rises, controls cascade in; joystick/sliders move the 3D head, antennas and body; keyboard works on desktop; browser back and "Voltar" reverse the animation to the head card; opening `/twin` directly then "Voltar" lands on `/` inside the app; swipe down closes on a phone; with reduced motion only fades are used.
- [ ] **Step 6:** Commit `feat(web-ui): full twin with manual controls (mocked)`.

### Task 13: Start-script build, docs and end-to-end verification

**Files:**
- Modify: `scripts/run-windows.ps1`, `CLAUDE.md` (Commands: frontend build/dev; Runtime Modes paragraph: `src/web_ui/frontend/` React SPA built to `dist/`; Web UI security: CSP note), `docs/features.md` ("Web page / phone app" bullet: home with Dobby's live 3D head, full twin with manual controls — simulated for now, real links that survive reload/back), `docs/web-ui.md` (build/dev section), `docs/architecture.md` (only if it mentions `static/` — check with grep)

**Interfaces:**
- Consumes: `npm ci`, `npm run build` (Task 1).

- [ ] **Step 1:** Add `Build-Frontend` to `run-windows.ps1`, called before starting the bridge: `$fe = Join-Path $root "src\web_ui\frontend"`; if `node_modules\.package-lock.json` is missing or older than `package-lock.json` → `npm ci` in `$fe`; if `dist\index.html` is missing or any file under `$fe` excluding `node_modules` and `dist` has `LastWriteTime` newer than it → `npm run build`; a non-zero `$LASTEXITCODE` → `throw "Build do painel web falhou."`. Comments in the script's style (pt-BR without accents, like the existing ones).
- [ ] **Step 2:** Run `powershell -ExecutionPolicy Bypass -File scripts/run-windows.ps1` twice: first run builds, second run skips both npm steps (no "npm" output), bridge starts and `http://localhost:18900` serves the new UI.
- [ ] **Step 3:** Update the docs listed above.
- [ ] **Step 4: End-to-end manual pass (Chrome, bridge running):** reload on every route (including `/logs?q=x&nivel=debug` and `/twin`); back/forward across tabs and twin; 390 px and 1280 px; install the PWA (Chrome "Install app") and open it; camera, chat, wake/sleep, wake test, logs with server Debug on/off; no CSP violations in the console; through the Cloudflare tunnel on the phone if available.
- [ ] **Step 5:** `ruff check .` clean; `npm run lint` clean; `PYTHONPATH=src C:/GITHUB/dobby-the-claw/.venv/Scripts/python.exe -m pytest tests/web_ui -q` PASS.
- [ ] **Step 6:** Commit `chore(web-ui): build frontend on start; docs for the new panel`.
