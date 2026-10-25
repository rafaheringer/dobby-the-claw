# Touch reactions (petting and taps) — design

Date: 2026-10-24 · Status: approved in chat, pending spec review

## Goal

Dobby notices when someone pets the top of his head (where the mic array sits) or gives it a
few quick taps, and reacts like a pet would: physically, with a sound, and — when petting stops —
by asking for more in a sweet voice. Taps get a startled, grumpy reaction.

## Feasibility findings (probe, 2026-10-24)

Recorded 8 s clips from the Reachy Mini Audio WASAPI input (48 kHz, 2 channels) on the Windows host:

- The two channels are identical (L/R correlation 1.000): the XMOS output is one processed beam,
  so there is no spatial cue. Raw capsules are not exposed on Windows.
- Petting **saturates** the array: 100 ms windows reach −7 dBFS and samples clip at 1.0–1.49.
  Speech at ~1 m never exceeded −22.7 dBFS (peak 0.30). The XMOS gates the gaps between strokes
  down to −80…−130 dBFS, giving a burst/silence pattern.
- Spectral cues (flatness, high-band ratio) did **not** separate well: the loud petting windows are
  low-frequency and tonal, like speech. Level is the discriminator.
- Rule "≥3 windows above −18 dBFS within 1.5 s" fired on normal petting (1.5 s) and light petting
  (2.4 s) with no false positive on silence, speech or taps.
- Taps are short: 100 ms RMS stays around −21…−27 dBFS, but the sample peak reaches 0.70 vs 0.30
  for speech, so taps need a transient (peak) detector, not the RMS rule.
- Not yet measured: Dobby's own voice through the speaker and very close/loud speech.
- **AGC collapse (found in the first live test, 2026-10-25):** petting saturates the array and the
  XMOS AGC (`PP_AGCGAIN`) drops from ~2.0 to ~0.04 within a second, then barely recovers in silence:
  Dobby goes deaf (no more touches, no speech, idle sleep). The mic "going silent" during the probe
  was the same effect. Hence the gain restore after every petting/tap.
- In the same test the petting hand in front of the camera was reported as an "unknown person"
  (presence greeting mid-petting), finger antenna tracking reacted to the approaching hand, and head
  tracking/speech offsets kept moving the head under the petting pose.

## Behaviour

| Situation | Petting | Taps |
|---|---|---|
| Awake, quiet | Antennas fold back to the sleep-pose angles (±3.05 rad), head tilts ~10° and lifts slightly "into the hand", `loving1.wav` plays; antennas sway slowly while petting lasts. On stop: pose eases back; with cooldown, Dobby asks for more in a soft, sweet voice. | Startle: antennas flick up, head pulls back with a small shake, `surprised1.wav` plays; with cooldown, Dobby makes a short sarcastic complaint. |
| Awake, speaking | Ignored (detector deaf while playback runs plus ~0.8 s tail). | Ignored. |
| Asleep (motors off) | `relief1.wav` at low volume. No motion, no wake, not counted as user activity. | A soft grumble sound at low volume. Same rules. |

While a touch move runs it owns head and antennas: face tracking, speech wobble and finger antenna
offsets fade out (0.4 s) and the listening antenna freeze is bypassed. Presence notices are absorbed
(committed unsent) from `PET_START` until 15 s after the touch ends. After every `PET_END`/`TAP` the
mic AGC gain is written back to `TOUCH_AGC_RESTORE_GAIN`. Finger antenna control ignores hands that
span more than 40 % of the frame.

Cooldown for the spoken reaction: the first one always speaks; later ones only if ≥5 min passed
since the last spoken reaction of that kind. Physical reaction and sound always run.

Sounds come from the cached `pollen-robotics/reachy-mini-emotions-library` dataset (already used by
the `emotion` tool); the recorded moves themselves are not used.

## Components

### `src/bridge/runtime/touch.py` — `TouchDetector`

Pure, I/O-free, unit-testable.

- `observe(sample_rate, sample, now, deaf: bool) -> list[TouchEvent]`; downmixes to mono and slices
  into 100 ms windows (keeps a remainder buffer across chunks).
- Petting: `PET_START` when ≥3 windows above `TOUCH_PET_DBFS` (default −18) within 1.5 s;
  `PET_END(duration_s)` after 2 s without a loud window.
- Tap: a window whose sample peak ≥ `TOUCH_TAP_PEAK` (default 0.5) while its RMS stays below the pet
  threshold. It is held for 1.0 s and emitted as one `TAP` only if no petting window occurred in the
  1.5 s before it nor during the hold; further transients during the hold merge into the same `TAP`.
  (In the probe only one of the 2–3 taps peaked above 0.6, and light petting produces isolated
  transients right next to its loud windows — hence single-transient taps with an isolation rule.)
  Petting and taps are mutually exclusive.
- `deaf` (Dobby speaking) drops windows, resets partial state and ends a petting episode with
  `PET_END(interrupted=True)`, which never earns a spoken notice. `muted` (our own reaction sound is
  playing) only drops windows: an ongoing petting episode continues.

### Robot reactions — `RobotActionsPort`

Two new port methods, implemented in `ReachyRobotActions`:

- `react_touch(kind: str, phase: str, asleep: bool)` — kind `pet`/`tap`, phase `start`/`end`.
  Awake: queues a new `Move` subclass on the MotionManager (pose targets for head + antennas,
  smooth in/out; sway loop while petting) and plays the sound. Asleep: plays the sound only, at low
  volume.
- Sounds are resolved (quiet copies written to `~/.dobby/touch_sounds`) once at construction and
  played with SDK `media.play_sound` on a single worker thread, off the mic loop; `react_touch`
  returns the sound length so the orchestrator mutes the detector meanwhile.
- `PetMove`/`TapMove` are `exclusive` moves (see Behaviour); `PetMove.release_now()` measures time on
  the move's own clock, since it may have waited in the queue.

### Orchestrator wiring

- `TouchDetector` is fed at the same point as `wakeword.observe_sample` in the mic loop
  (awake and asleep), with `deaf = playback_started or now < playback_end + 0.8` and
  `muted = reaction sound playing`.
- Events are handled without blocking: robot reactions run on the MotionManager thread / media.
- On `PET_END` / `TAP` while awake and past cooldown, send a notice with
  `realtime.send_text(text, respond=True)` (same path as presence notices). Text is short, e.g.
  `[toque] carinho de 6 s terminou`; tone rules live in `src/prompts/touch.md` (sweet, soft,
  ≤8 words asking for more; taps: short sarcastic complaint in Dobby's voice).
- Touch events do not update `last_user_activity` while asleep.

### Configuration

| Variable | Default | Notes |
|---|---|---|
| `TOUCH_ENABLED` | `1` | Petting/tap detection through the mic array |
| `TOUCH_PET_DBFS` | `-18` | 100 ms RMS level counted as a petting window |
| `TOUCH_TAP_PEAK` | `0.5` | Sample peak counted as a tap transient |
| `TOUCH_SPEAK_COOLDOWN_S` | `300` | Minimum gap between spoken touch reactions of the same kind |
| `TOUCH_AGC_RESTORE_GAIN` | `2.0` | XMOS AGC gain written back after petting/taps (`0` disables) |

## Testing

- New `tests/` with pytest (first tests in the repo): `TouchDetector` against the probe clips
  (`tests/fixtures/touch/*.npy`, downsampled to 16 kHz mono to keep them small): silence/speech → no
  event; petting/light petting → `PET_START` then `PET_END`; taps → one `TAP`; deaf → nothing.
- End to end: the probe clips through the real detector, gate and `_observe_touch` with the robot
  and model faked (`tests/test_touch_end_to_end.py`).
- Live check on the robot: petting and taps awake and asleep, plus Dobby speaking and close speech
  to tune thresholds.

## Docs

Update `docs/features.md`, the `CLAUDE.md` configuration table and `.env.example`.

## Out of scope

Raw-capsule access, localisation of the hand, ML classifiers, interrupting Dobby mid-speech.
