# Touch Reactions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dobby detects petting and taps on his head through the mic array and reacts with a pose, a sound and (with cooldown) a short spoken line.

**Architecture:** A pure `TouchDetector` turns mic chunks into `TouchEvent`s; a pure `TouchSpeechGate` decides when a spoken reaction is due and builds the notice text. The orchestrator feeds the detector from the mic loop and dispatches events to a new `RobotActionsPort.react_touch()` (pose `Move` on the MotionManager + SDK `media.play_sound`) and to `realtime.send_text()`.

**Tech Stack:** Python 3.11, numpy, Reachy Mini SDK (`media.play_sound`, MotionManager `Move`), pytest (new dev dependency).

**Spec:** `docs/superpowers/specs/2026-10-24-touch-reactions-design.md`

## Global Constraints

- Defaults: `TOUCH_ENABLED=1`, `TOUCH_PET_DBFS=-18`, `TOUCH_TAP_PEAK=0.5`, `TOUCH_SPEAK_COOLDOWN_S=300`.
- Windows of 100 ms; petting = ≥3 windows above the pet threshold within 1.5 s; `PET_END` after 2.0 s without a loud window.
- Tap = window peak ≥ tap threshold with RMS below pet threshold, held 1.0 s, dropped if any loud window in the 1.5 s before or during the hold; transients during the hold merge.
- Detector is deaf while Dobby's audio plays plus 0.8 s tail, and while a reaction sound plays.
- No blocking work in the mic loop; motion runs on the MotionManager thread, sounds via SDK `play_sound`.
- While asleep: sound only (low volume), no motion, no wake, `last_user_activity` untouched.
- Spoken reaction: first of each kind always; later ones only after the cooldown. Model tone rules in `src/prompts/touch.md`, pt-BR.
- Ruff, 100-char lines. Update `docs/features.md`, `CLAUDE.md` config table, `.env.example`.

## Review Focus

- Chunk sizes that are not multiples of 100 ms (SDK gives ~10 ms, ALSA other sizes) → identical events to whole-window feeding. Test in Task 1.
- Petting that continues straight into Dobby speaking (deaf flips on mid-episode) → the episode ends with exactly one `PET_END`, no stuck "petting" state. Test in Task 1.
- Stereo `(frames, 2)` and mono 1-D input both accepted. Test in Task 1.
- Missing emotions dataset / SDK media unavailable → reaction logs a warning and the mic loop keeps running. Test in Task 3.
- `TOUCH_ENABLED=0` → no detector built, mic loop unchanged. Covered by Task 4 manual check.

---

### Task 1: `TouchDetector` + test infrastructure

**Files:**
- Create: `src/bridge/runtime/touch.py`, `tests/conftest.py`, `tests/test_touch_detector.py`, `requirements-dev.txt`
- Already created (commit them): `tests/fixtures/touch/{silencio,fala,carinho,carinho_leve,batidinha}.npy` — 8 s, 16 kHz, mono int16 (from the probe). Known content: `carinho` loud windows from 0.5 s; `carinho_leve` loud at 0.9/2.1/2.3/3.5 s with transients at 2.4/3.4 s; `batidinha` one transient at 2.5 s, no loud window.

**Interfaces:**
- Produces:
  - `class TouchKind(str, Enum)`: `PET_START = "pet_start"`, `PET_END = "pet_end"`, `TAP = "tap"`
  - `@dataclass(frozen=True) class TouchEvent: kind: TouchKind; at: float; duration_s: float = 0.0`
  - `class TouchDetector.__init__(self, pet_dbfs: float = -18.0, tap_peak: float = 0.5)`
  - `TouchDetector.observe(self, sample_rate: int, sample: np.ndarray, now: float, deaf: bool = False) -> list[TouchEvent]`
  - `TouchDetector.petting -> bool` (property)

- [ ] **Step 1:** `requirements-dev.txt` with `pytest>=8`; `tests/conftest.py` puts `src/` on `sys.path`; install with `.venv/Scripts/python.exe -m pip install -r requirements-dev.txt`.
- [ ] **Step 2: Write the failing tests** in `tests/test_touch_detector.py`. Helper `run(name, chunk=160, tail_s=3.0, deaf_from=None, stereo=False)` loads the fixture as float32 (`/32767`), appends `tail_s` of zeros, feeds chunks with `now = index / 16000`, `deaf = deaf_from is not None and now >= deaf_from`, returns all events (`stereo=True` feeds `np.stack([x, x], axis=1)` chunks).

```python
def test_silence_and_speech_produce_nothing():
    assert run("silencio") == [] and run("fala") == []

def test_petting_starts_fast_and_ends_once():
    ev = run("carinho")
    assert [e.kind for e in ev] == [TouchKind.PET_START, TouchKind.PET_END]
    assert ev[0].at <= 1.6
    assert ev[1].duration_s > 5.0

def test_light_petting_detected_without_taps():
    ev = run("carinho_leve")
    assert ev[0].kind is TouchKind.PET_START and ev[0].at <= 2.5
    assert TouchKind.TAP not in [e.kind for e in ev]

def test_tap_emitted_once_after_hold():
    ev = run("batidinha")
    assert [e.kind for e in ev] == [TouchKind.TAP]
    assert 3.4 <= ev[0].at <= 3.7

def test_deaf_suppresses_everything():
    assert run("carinho", deaf_from=0.0) == []

def test_deaf_mid_episode_ends_it_once():
    kinds = [e.kind for e in run("carinho", deaf_from=3.0)]
    assert kinds == [TouchKind.PET_START, TouchKind.PET_END]

def test_chunking_and_stereo_do_not_change_events():
    base = [(e.kind, round(e.at, 1)) for e in run("carinho")]
    assert [(e.kind, round(e.at, 1)) for e in run("carinho", chunk=437)] == base
    assert [(e.kind, round(e.at, 1)) for e in run("carinho", stereo=True)] == base
```

- [ ] **Step 3:** Run `.venv/Scripts/python.exe -m pytest tests/test_touch_detector.py -v` → FAIL (`ModuleNotFoundError: bridge.runtime.touch`).
- [ ] **Step 4: Implement `touch.py`.** Downmix with the existing `_to_mono` pattern (`wakeword.py:40`), keep a remainder buffer, window = `sample_rate // 10` samples, timestamp each window by its end (`now` minus remaining buffered duration). Keep a deque of loud-window times (1.5 s). Pending tap: `(first_at, last_at)`; emit at `first_at + 1.0` if no loud window since `first_at - 1.5`. When `deaf` turns true during petting, emit `PET_END` immediately and clear all state.
- [ ] **Step 5:** Run the tests → PASS. `ruff check src tests`.
- [ ] **Step 6: Commit** `feat(touch): detect petting and taps from mic level` (code, tests, fixtures, requirements-dev).

### Task 2: `TouchSpeechGate` + prompt

**Files:**
- Modify: `src/bridge/runtime/touch.py`
- Create: `src/prompts/touch.md`, `tests/test_touch_speech_gate.py`

**Interfaces:**
- Consumes: `TouchEvent`, `TouchKind` (Task 1)
- Produces: `class TouchSpeechGate.__init__(self, cooldown_s: float = 300.0)`; `TouchSpeechGate.notice(self, event: TouchEvent, now: float) -> str | None` — returns the notice text for `PET_END`/`TAP` when due (and records it), else `None`; always `None` for `PET_START`.

- [ ] **Step 1: Failing tests**

```python
def test_first_pet_end_speaks_then_cooldown():
    g = TouchSpeechGate(cooldown_s=300)
    assert g.notice(TouchEvent(TouchKind.PET_END, 10, 6.2), 10) == "[toque] carinho de 6 s terminou"
    assert g.notice(TouchEvent(TouchKind.PET_END, 100, 3.0), 100) is None
    assert g.notice(TouchEvent(TouchKind.PET_END, 311, 3.0), 311) is not None

def test_kinds_have_independent_cooldowns():
    g = TouchSpeechGate()
    g.notice(TouchEvent(TouchKind.PET_END, 0, 4), 0)
    assert g.notice(TouchEvent(TouchKind.TAP, 1), 1) == "[toque] levou batidinhas na cabeça"

def test_pet_start_never_speaks():
    assert TouchSpeechGate().notice(TouchEvent(TouchKind.PET_START, 0), 0) is None
```

- [ ] **Step 2:** Run → FAIL. **Step 3:** implement (duration rounded to int seconds). **Step 4:** Run → PASS.
- [ ] **Step 5: Write `src/prompts/touch.md`** (pt-BR): explains `[toque]` notices are physical touches on the head, not speech; on `carinho ... terminou` answer with voz dengosa, manhosa, baixinha e lenta, pedindo mais carinho, no máximo 8 palavras; on `batidinhas` a reclamação curta e debochada, no máximo 10 palavras; never mention sensors or microphones.
- [ ] **Step 6: Commit** `feat(touch): throttle spoken touch reactions`.

### Task 3: Robot reaction — pose move + sounds

**Files:**
- Create: `src/reachy/touch_moves.py`, `tests/test_touch_moves.py`
- Modify: `src/bridge/runtime/ports.py` (`RobotActionsPort`), `src/bridge/runtime/adapters/reachy_actions.py`

**Interfaces:**
- Consumes: `Move`, `create_head_pose` from `reachy.motion` (same imports as `BreathingMove`).
- Produces:
  - `class PetMove(Move)`: `__init__(self, hold_s: float = 30.0)`; ease-in 0.6 s to antennas `(-0.9, 0.9)` rad (back), head roll 10°, z +6 mm, pitch −5°; sway antennas ±0.15 rad at 0.4 Hz while holding; `release()` makes it ease out over 0.8 s from the current time (duration ends then).
  - `class TapMove(Move)`: 1.2 s — antennas flick to `(0.6, -0.6)` in 0.1 s, head x −8 mm with a ±6° yaw shake (2 cycles), back to neutral.
  - `RobotActionsPort.react_touch(self, kind: str, phase: str, asleep: bool) -> float` — kind `"pet"`/`"tap"`, phase `"start"`/`"end"`; returns the duration in seconds of the sound started (0.0 if none), so the orchestrator can keep the detector deaf.
  - `ReachyRobotActions.__init__(self, reachy: ReachyClient, motion_manager: MotionManager | None = None)`.

- [ ] **Step 1: Failing tests** for the moves (pure math):

```python
def test_pet_move_reaches_pose_and_releases():
    m = PetMove()
    _, ant, _ = m.evaluate(0.6)
    assert ant[0] == pytest.approx(-0.9, abs=0.16) and ant[1] == pytest.approx(0.9, abs=0.16)
    m.release(at=5.0)
    assert m.duration == pytest.approx(5.8)
    _, ant_end, _ = m.evaluate(5.8)
    assert abs(ant_end[0]) < 0.05

def test_tap_move_is_short_and_returns_to_neutral():
    m = TapMove()
    assert m.duration == pytest.approx(1.2)
    head, ant, _ = m.evaluate(1.2)
    assert np.allclose(head, np.eye(4), atol=1e-3) and np.allclose(ant, 0, atol=1e-3)
```

`PetMove.release(at: float)` takes the move-local time; the adapter tracks it from `time.monotonic()` minus the queue time.
- [ ] **Step 2:** Run → FAIL. **Step 3:** implement `touch_moves.py`. **Step 4:** Run → PASS.
- [ ] **Step 5: Implement `react_touch` in `ReachyRobotActions`.** Sounds from the emotions dataset snapshot (resolve once with `huggingface_hub.snapshot_download("pollen-robotics/reachy-mini-emotions-library", repo_type="dataset", local_files_only=True)`): awake pet start → `loving1.wav`, tap → `surprised1.wav`; asleep pet → `relief1.wav`, tap → `oops1.wav`, both written once at 35 % amplitude to `../.dobby/touch_sounds/` and played from there. Awake pet start queues `PetMove` and keeps it; pet end calls `release`; tap queues `TapMove`. Asleep: no move. Play with `reachy.get_sdk_instance().media.play_sound(path)`. Any exception → `logging.warning`, return 0.0. Add `react_touch` to `RobotActionsPort` with a docstring.
- [ ] **Step 6:** Test `test_react_touch_survives_missing_sdk`: `ReachyRobotActions(FakeClient raising on get_sdk_instance)`, `react_touch("pet", "start", asleep=True)` returns `0.0` and does not raise.
- [ ] **Step 7:** Run all tests, ruff, commit `feat(touch): pet and tap reactions with sounds`.

### Task 4: Config, orchestrator wiring, docs

**Files:**
- Modify: `src/bridge/config.py`, `src/bridge/runtime/orchestrator.py`, `.env.example`, `CLAUDE.md`, `docs/features.md`

**Interfaces:**
- Consumes: `TouchDetector`, `TouchSpeechGate` (Tasks 1–2), `react_touch` (Task 3).

- [ ] **Step 1: Config fields** `touch_enabled: bool = True`, `touch_pet_dbfs: float = -18.0`, `touch_tap_peak: float = 0.5`, `touch_speak_cooldown_s: float = 300.0`, read in `from_env` with the env names from Global Constraints (`_env_flag` for the flag).
- [ ] **Step 2: Orchestrator.** In `__init__`: build `self._touch` / `self._touch_gate` when `config.touch_enabled`, `self._touch_deaf_until = 0.0`, and pass `motion_manager` to `ReachyRobotActions(reachy, motion_manager)`. Append `load_prompt("touch.md")` to the identity next to `presence.md` (`orchestrator.py:701`) when enabled. In the mic loop right after `self.wakeword.observe_sample(...)` (`orchestrator.py:235`) call `self._observe_touch(sample, now)`.
- [ ] **Step 3:** `_observe_touch(self, sample, now) -> None`: `deaf = self.playback_started or now < self._playback_end_ts + 0.8 or now < self._touch_deaf_until`; for each event call `react_touch(kind, phase, asleep=self.sleeping)`, set `_touch_deaf_until = now + returned_duration + 0.3`; when awake and `self.realtime is not None`, `text = self._touch_gate.notice(event, now)` and, if not None, `self.realtime.send_text(text, respond=True)` and `logging.info` it; when awake also refresh `last_user_activity`.
- [ ] **Step 4: Manual check** with the bridge: `TOUCH_ENABLED=0` → log shows no touch setup and the loop behaves as before; `TOUCH_ENABLED=1` → petting logs `PET_START`/`PET_END`, robot reacts.
- [ ] **Step 5: Docs.** `.env.example` and `CLAUDE.md` table get the four variables; `docs/features.md` gets a plain-English "Carinho e batidinha" entry.
- [ ] **Step 6:** ruff, tests, commit `feat(touch): react to petting and taps`.

### Task 5: Live validation and tuning (with the user)

- [ ] **Step 1:** Mic must be alive (USB replug / daemon restart). Record and check: Dobby speaking a long answer and close/loud speech (~20 cm) produce no events (detector deaf during playback; close speech stays below −18 dBFS or the threshold is raised).
- [ ] **Step 2:** Petting and taps awake and asleep; adjust `TOUCH_PET_DBFS` / `TOUCH_TAP_PEAK` defaults in config, `.env.example`, `CLAUDE.md` and spec if needed; commit `chore(touch): tune thresholds from live test`.
