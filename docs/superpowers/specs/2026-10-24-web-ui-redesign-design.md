# Web UI redesign — design

Date: 2026-10-24 · Status: approved in brainstorming, pending spec review

## Goal

Rebuild the Dobby web UI (PWA) from scratch as the control panel for the robot, used on the
desktop and, mostly, on the phone from anywhere (through Cloudflare Access). The current page
(`src/web_ui/static/`: vanilla HTML/JS/CSS) was made in a hurry: inconsistent look, wordy copy,
and pages switched by CSS class with no URL, so reload and browser back/forward do not work.

Success criteria:

- Same functions as today, with a consistent "soft glass" look, short copy and good motion design.
- Responsive, mobile first (390 px phone up to wide desktop).
- Real URLs: reload, deep links and back/forward work on every screen.
- Still an installable PWA.
- A 3D digital twin of the Reachy Mini: the head (no body) lives on the home screen; tapping it
  opens the full twin with manual controls. Mocked for now; a later session wires it to the robot.
- The structure must take future features (real twin, camera control, …) without another rewrite.

Out of scope now: realtime twin and sending commands to the robot (only the `RobotLink`
interface and a mock), camera control, light theme (tokens are ready for it), protocol changes.

## Visual direction

"Soft glass": deep blue/indigo radial gradients, translucent panels with backdrop blur, large
radii, floating tab bar, one accent (periwinkle `#9aa8ff`), state colors (listening green,
thinking amber, sleeping slate, error red). Font: Inter (self-hosted via `@fontsource`). Dark
theme only for now, built on CSS custom-property tokens.

Copy: pt-BR, short labels (one or two words), no explanatory subtitles, empty states only where
useful. Server error texts stay as they are.

## Stack

React 19 + Vite + TypeScript (strict), React Router (declarative `BrowserRouter`), Tailwind CSS
v4, Motion (framer-motion), `@react-three/fiber` + `@react-three/drei` + `three`, Zustand,
Radix UI primitives (slider, switch, dialog, dropdown) styled in-house, `@tanstack/react-virtual`
for the log list, `vite-plugin-pwa` (Workbox), Vitest.

Chosen because it is the stack of Pollen Robotics' official app
(`pollen-robotics/reachy-mini-desktop-app`, Apache-2.0), whose 3D viewer (`GLTFRobot`), robot
model (`reachy_mini_viz.glb`, ~3 MB, Draco) we adapt, and whose Controller widgets we model ours
on. Upstream commit: `5796d5b94f07cd889e955e3b9e785eafee1cac89`. Alternatives considered:
SvelteKit + Threlte and Vue + TresJS (lighter, but no reuse of Pollen's code).

## Layout of the code

```
src/web_ui/
  server.py              serves frontend/dist (SPA fallback); /ws and /api/video unchanged
  access.py, logbus.py   unchanged
  frontend/
    package.json, vite.config.ts, tsconfig.json, index.html
    public/              icons (moved from static/), robot GLB
    src/
      main.tsx
      app/               router, root layout (persistent 3D canvas + navigation), motion presets
      routes/            Home, Chat, Logs, System, SystemWake, Twin
      twin/              Scene, GLTFRobot (adapted from Pollen), head/full framing, controls
      ui/                glass components: Card, Pill, Button, Switch, Slider, Toast, Sheet…
      live/              WebSocket client + Zustand store (status, chat, logs, wake test)
      robot/             RobotLink interface + MockRobotLink
      lib/               pure helpers (log filter, URL filter params, pose math)
    dist/                build output (git-ignored)
```

`src/web_ui/static/` is removed (old html/js/css/sw.js); icons move to `frontend/public/`.

Files adapted from Pollen's repo keep a header naming the source file, the upstream commit and
the Apache-2.0 license. The GLB gets a `NOTICE`/attribution entry next to it.

## Backend changes (`server.py`)

- Static root becomes `frontend/dist`.
- `/assets/*` (hashed file names): `Cache-Control: public, max-age=31536000, immutable`.
- `/sw.js`, `/manifest.webmanifest`, `/index.html` and root files in `dist/`: `no-cache`.
- SPA fallback: any `GET` that is not `/ws`, `/api/*`, `/assets/*` or an existing file in
  `dist/` returns `index.html` (`no-cache`).
- When `dist/` is missing, `/` returns a short HTML page saying the frontend must be built
  (`npm --prefix src/web_ui/frontend ci && npm --prefix src/web_ui/frontend run build`) and the
  bridge logs a warning once at startup.
- WebSocket protocol, `/api/video` and the access guard stay as they are.
- CSP (`access.py`) gains exactly two directives, needed by the Draco decoder of Pollen's GLB
  (WebAssembly compiled inside a Blob worker): `script-src 'self' 'wasm-unsafe-eval'` and
  `worker-src 'self' blob:`. `'wasm-unsafe-eval'` allows compiling WebAssembly only, not JS
  `eval`. Everything else must fit the existing policy: no CDNs, no inline scripts, fonts
  self-hosted, Draco decoder files served from `/draco/` (copied from
  `three/examples/jsm/libs/draco/gltf/` at build time). Re-encoding the GLB without Draco was
  tried and rejected: 12.6 MB (5.7 MB gzip) and the optimizer pruned the skin.

## Build and run

- `scripts/run-windows.ps1` before starting the bridge: run `npm ci` in `frontend/` when
  `package-lock.json` is newer than `node_modules/.package-lock.json` (or `node_modules` is
  missing), and `npm run build` when any file under `frontend/` (excluding `node_modules`,
  `dist`) is newer than `dist/index.html`. A build failure stops the script with the error.
- `npm run build` = `tsc --noEmit && vitest run && vite build`.
- Dev: `npm run dev` in `frontend/`; Vite proxies `/ws` (WebSocket) and `/api` to
  `http://127.0.0.1:18900`. Hot reload; the bridge keeps running.
- `CLAUDE.md` and `docs/features.md` are updated (commands and the user-facing description).

## PWA

`vite-plugin-pwa` generates the manifest (same name, icons, `pt-BR`, `standalone`, theme color
of the new palette) and a Workbox service worker: precache of the built assets (including the
GLB), navigation fallback to `index.html`, `/ws` and `/api/*` never cached, `registerType:
autoUpdate` (a new version activates on the next load).

## Routes and screens

| Route | Screen |
|---|---|
| `/` | Home |
| `/conversa` | Chat |
| `/logs` | Logs (filters in the query string: `q`, `nivel`, `cat`) |
| `/sistema` | System |
| `/sistema/nome` | Wake-word test |
| `/twin` | Full twin (over the current layout) |
| anything else | redirect to `/` |

**Shell.** Phone: header with "Dobby" and the state pill, floating glass tab bar (Início,
Conversa, Logs, Sistema) with a sliding active indicator. Desktop (≥ 1024 px): glass sidebar with
the same sections, the state pill and the wake/sleep button. The state pill shows Ouvindo /
Pensando / Executando / Delegando / Confirmando / Em espera / Dormindo / Erro / Acordando… /
Indo dormir…, colored and pulsing per state. One wake/sleep toggle (spinner while
`transition` is set). Toasts for server `error` messages. A discreet "reconectando…" state when
the socket is down.

**Home.** The 3D head (head mode) with a subtle ring; tap opens `/twin`. Camera card: MJPEG
`/api/video` with overlays, plus an options menu with the Pessoas / Voz / Rastreio overlay
toggles. Indicators: people in view (names), who is speaking, mic level meter, and — while
asleep — what the wake word detector heard. Desktop: grid with a large camera, the head and the
latest chat messages.

**Chat.** Bubbles (Você / Dobby) with time, spring entrance, auto-scroll when at the bottom.
Input pinned at the bottom, keyboard-safe on phones (`visualViewport`). While asleep the input is
replaced by "Dobby está dormindo" and a wake button.

**Logs.** Search (`!term` hides), level select (Debug/Info/Avisos/Erros), category chips,
pause, clear. Virtualized list (up to the 3000-line backlog plus live lines, capped at 4000).
Filters live in the URL, so reload/back keep them.

**System.** "Serviços": state, model, Home Assistant, OpenClaw, Raspberry Pi (CPU, memory), camera
FPS. "Controles": Cabeça segue (head tracking), Câmera ligada (video enabled), Log em Debug (server
log level). Item "Teste do nome" → `/sistema/nome`.

**Wake-word test.** Big "Testar" button; during the 3 s recording an animated countdown ring,
then "Entendendo…"; result ✓/✕ with the heard phrase; last 6 tries. When awake, a short notice
with "Colocar para dormir". Same 15 s client guard as today if the server never answers.

## Live data (`live/`)

One WebSocket to `/ws` (same protocol as today). Messages in: `hello`, `status`, `logs`,
`chat`, `wake_state`, `wake_result`, `error`. Messages out: `chat`, `cmd` (`wake`/`sleep`),
`head_tracking`, `video_enabled`, `log_level`, `wake_test`. A pure reducer applies incoming
messages to the Zustand store; components subscribe to slices. Reconnect with exponential
backoff 0.8 s → 10 s; on reconnect `hello` replaces the state. Sending while disconnected shows a
toast and controls that send commands are disabled.

Camera stream: only open while on `/` (or the desktop grid shows it) and the tab is visible;
width picked from 640 / 960 / 1280 by the card size × DPR (fixed steps, as today); on error a
placeholder and retry after 3 s; "Sem câmera" / "Câmera desligada" placeholders from status.

## Twin (`twin/`, `robot/`)

**One persistent canvas.** The R3F `<Canvas>` lives in the root layout, never remounts. The route
decides its frame: on `/` it sits inside the head card (head mode); on `/twin` it fills the
screen (full mode). Motion layout animation moves/resizes the container; inside the scene the
camera and body visibility animate between the two framings. On other routes the canvas is
hidden and its render loop paused (`frameloop="demand"` / not rendering).

**Robot.** `GLTFRobot` adapted from Pollen: takes `headPose` (4×4), `antennas` [l, r] and
`yawBody`, solves the Stewart-platform legs itself, smooths changes. Head mode hides the body
and leg meshes and frames head + antennas; full mode shows everything. Lighting: soft
environment, contact shadow, faint accent rim. Transparent background so the glass shows through.

**`RobotLink`** — the seam the next session implements for real:

```ts
interface RobotPose { headPose: Mat4; antennas: [number, number]; yawBody: number }
interface RobotTarget { x: number; y: number; z: number; roll: number; pitch: number; yaw: number;
                        antennas: [number, number]; yawBody: number }
interface RobotLink {
  usePose(): RobotPose & { mode: "auto" | "manual"; simulated: boolean }
  setManual(on: boolean): void
  setTarget(target: Partial<RobotTarget>): void
  center(): void
  playEmotion(name: EmotionName): void
}
```

**`MockRobotLink`.** Auto mode: idle "alive" motion (breathing, antenna sway) and a short
"atento" head tilt + antennas up when `status.speaker` changes to a new name (so it already
reacts to real data; the status carries no speaker direction, so it does not turn toward them).
Manual mode: controls move the pose locally with smoothing. Emotions: short keyframed moves
(feliz, curioso, sono, surpreso). Sends nothing to the robot; `/twin` shows a "simulado" pill.

**Head mode** hides every mesh that is not a descendant of the head bone `Core.001`
(normalized `core001`), including the two skinned cables under `Armature`.

**Controls on `/twin`** (modeled on Pollen's Controller widgets, rewritten on Radix + our glass
styles because Pollen's are built on MUI): 2D joystick (head yaw/pitch),
vertical slider (z), slider (roll), two circular sliders (antennas), one circular slider (body
yaw), "Centralizar", "Emoções" menu, manual/auto toggle. Desktop keyboard: arrows/WASD for the
joystick, Q/E roll, R/F height. Touching any control switches to manual.

**Fallback.** No WebGL or GLB load failure: a 2D head illustration on Home; `/twin` shows the
controls with a notice.

## Motion

One spring preset for the whole app (Motion `type: "spring"`, ~`stiffness 380, damping 34`) and
one short fade (150 ms). Defined once in `app/motion.ts`.

- Open twin: the rest of the screen blurs and fades, the canvas container animates from the card
  to full screen, the 3D camera pulls back while the body rises and fades in, controls enter
  with a 40 ms stagger. Close (back button, browser back, swipe down on the phone) plays it in
  reverse.
- Tab change: fade + short slide in the direction of the tab; the tab-bar indicator slides
  (`layoutId`).
- Micro: state pill pulse per state, animated number changes, chat bubble springs, wake-test
  countdown ring.
- `prefers-reduced-motion`: no slides/springs, fades only.
- Performance: 3D render paused when the tab is hidden or the head card is off screen; DPR
  capped at 2.

## Errors and edge cases

| Case | Behavior |
|---|---|
| Socket down | pill "reconectando…", command controls disabled, camera paused, backoff reconnect |
| Camera missing / off / stream error | placeholder; retry after 3 s on error |
| No WebGL / GLB fails | 2D head on Home; controls-only twin with a notice |
| Unknown route | redirect to `/` |
| `dist/` missing | server explains how to build; warning in the bridge log |
| Server `error` message | toast with the server text |

## Verification

- `tsc --noEmit` and ESLint clean; `ruff check` clean on changed Python.
- Vitest for pure units: log filter (`!term`, level, categories), WS message reducer, URL filter
  params, mock pose math (pose → matrix, smoothing bounds).
- Manual, with the bridge running, through Chrome: reload on every route; back/forward including
  closing `/twin`; phone width (390 px) and desktop; PWA install; camera, chat, wake/sleep,
  wake-word test, logs with server Debug on/off.
