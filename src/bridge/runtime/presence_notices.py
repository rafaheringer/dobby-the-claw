"""Turn the people in view into short, throttled presence notices for the conversation model.

Only changes are announced (someone arrived, was recognized, or left), so a quiet room costs
no tokens. Arrivals may ask the model to respond (it decides whether to greet); departures are
context only. Pure logic: the orchestrator feeds observations and sends the notices.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

UNKNOWN_KEY = ""  # Counter key for people in view who were not recognized
PET_PREFIX = "pet:"  # Counter keys for animals, e.g. "pet:cat"
_PET_WORDS = {"cat": ("um gato", "gatos"), "dog": ("um cachorro", "cachorros")}


@dataclass(frozen=True)
class PresenceNotice:
    """One notice ready to send."""

    text: str
    respond: bool
    greeted: tuple[str, ...] = ()  # labels to mark as greeted once sent
    announced: Counter = field(default_factory=Counter, compare=False)  # state after sending


class PresenceNotifier:
    """Track who is in view and decide when (and whether) to tell the model."""

    def __init__(
        self,
        greet_cooldown_s: float = 3 * 3600.0,
        unknown_greet_cooldown_s: float = 600.0,
        pet_greet_cooldown_s: float = 1800.0,
        id_wait_s: float = 5.0,
        leave_s: float = 20.0,
        min_interval_s: float = 15.0,
        baseline_s: float = 10.0,
    ) -> None:
        self._greet_cooldown_s = greet_cooldown_s
        self._unknown_greet_cooldown_s = unknown_greet_cooldown_s
        self._pet_greet_cooldown_s = pet_greet_cooldown_s
        self._id_wait_s = id_wait_s
        self._leave_s = leave_s
        self._min_interval_s = min_interval_s
        self._baseline_s = baseline_s
        self._announced: Counter[str] = Counter()
        self._present: Counter[str] = Counter()
        self._missing_since: dict[str, float] = {}
        self._greeted_at: dict[str, float] = {}
        self._last_notice_ts = 0.0
        self._baseline_until = 0.0
        self._baseline_seen: Counter[str] = Counter()

    def reset(self, now: float) -> None:
        """Forget who was announced (e.g. after waking up); whoever is in view right away is not greeted."""
        self._announced.clear()
        self._present.clear()
        self._missing_since.clear()
        self._baseline_seen.clear()
        self._baseline_until = now + self._baseline_s

    def mark_greeted(self, name: Optional[str], now: float) -> None:
        """Record that the model already interacted with this person (no greeting needed)."""
        self._greeted_at[name or UNKNOWN_KEY] = now

    def observe(self, people: Iterable[tuple[Optional[str], float]], now: float, pets: Iterable[str] = ()) -> None:
        """Fold in the people in view as (recognized name or None, seen-since timestamp) and pet labels."""
        present: Counter[str] = Counter(PET_PREFIX + label for label in pets)
        for name, since in people:
            if name:
                present[name] += 1
            elif now - since >= self._id_wait_s:
                present[UNKNOWN_KEY] += 1  # give face recognition a moment before calling them unknown
        self._present = present
        if now < self._baseline_until:
            self._baseline_seen |= present
        for label in set(self._announced) | set(present):
            if present[label] < self._announced[label]:
                self._missing_since.setdefault(label, now)
            else:
                self._missing_since.pop(label, None)

    def pending(self, now: float) -> Optional[PresenceNotice]:
        """Return the notice to send now, or None. Call `commit` once it was sent."""
        present, announced = self._present, self._announced
        arrived = present - announced
        departed = Counter({
            label: count
            for label, count in (announced - present).items()
            if now - self._missing_since.get(label, now) >= self._leave_s
        })
        if not arrived and not departed:
            return None
        if now - self._last_notice_ts < self._min_interval_s and now >= self._baseline_until:
            return None

        # An unknown person who becomes recognized is the same person, not a new arrival.
        recognized: list[str] = []
        spare = announced[UNKNOWN_KEY] - present[UNKNOWN_KEY]
        for name in sorted(n for n in arrived if _is_name(n)):
            if spare <= 0:
                break
            recognized.append(name)
            spare -= 1
        arrived_names = sorted(n for n in arrived if _is_name(n) and n not in recognized)
        arrived_unknown = arrived[UNKNOWN_KEY]
        arrived_pets = _pets(arrived)
        departed_names = sorted(n for n in departed if _is_name(n))
        departed_unknown = max(0, departed[UNKNOWN_KEY] - len(recognized))
        departed_pets = _pets(departed)

        next_announced = Counter(announced)
        for label in arrived:
            next_announced[label] = present[label]
        if recognized:
            next_announced[UNKNOWN_KEY] = max(present[UNKNOWN_KEY], announced[UNKNOWN_KEY] - len(recognized))
        for label in departed:
            next_announced[label] = present[label]
        next_announced = +next_announced

        parts = []
        if recognized:
            parts.append(f"a pessoa não reconhecida é {_join(recognized)}")
        arrivals = arrived_names + _unknown_phrase(arrived_unknown) + _pet_phrases(arrived_pets)
        if arrivals:
            count = len(arrived_names) + arrived_unknown + sum(arrived_pets.values())
            parts.append(f"{'chegaram' if count > 1 else 'chegou'} {_join(arrivals)}")
        departures = departed_names + _unknown_phrase(departed_unknown) + _pet_phrases(departed_pets)
        if departures:
            count = len(departed_names) + departed_unknown + sum(departed_pets.values())
            parts.append(f"{'saíram' if count > 1 else 'saiu'} {_join(departures)}")
        if not parts:
            if next_announced == announced:
                return None
            return PresenceNotice(text="", respond=False, announced=next_announced)

        to_greet = [
            label for label in arrived_names + recognized + ([UNKNOWN_KEY] if arrived_unknown else []) + list(arrived_pets)
            if label not in self._baseline_seen and self._may_greet(label, now)
        ]
        # Just greeted as an unknown person: being recognized a moment later needs no second greeting.
        just_greeted = [n for n in recognized if now - self._greeted_at.get(UNKNOWN_KEY, -1e12) < 120.0]
        respond = any(label not in just_greeted for label in to_greet)
        text = (
            f"SYSTEM NOTICE (presença, {time.strftime('%H:%M', time.localtime(now))}): "
            f"{'; '.join(parts)}. Agora à vista: {_describe(present)}."
        )
        greeted = tuple(dict.fromkeys(to_greet + just_greeted))
        return PresenceNotice(text=text, respond=respond, greeted=greeted, announced=next_announced)

    def commit(self, notice: PresenceNotice, now: float) -> None:
        """Apply `notice` as announced, after it was sent (or skipped when it has no text)."""
        for label in self._announced - notice.announced:
            self._baseline_seen.pop(label, None)  # left: greet them again when they come back
        self._announced = Counter(notice.announced)
        for label in notice.greeted:
            self._greeted_at[label] = now
        if notice.text:
            self._last_notice_ts = now

    def _may_greet(self, label: str, now: float) -> bool:
        if label.startswith(PET_PREFIX):
            cooldown = self._pet_greet_cooldown_s
        elif label == UNKNOWN_KEY:
            cooldown = self._unknown_greet_cooldown_s
        else:
            cooldown = self._greet_cooldown_s
        return now - self._greeted_at.get(label, -1e12) >= cooldown


def _describe(present: Counter[str]) -> str:
    items = sorted(n for n in present if _is_name(n)) + _unknown_phrase(present[UNKNOWN_KEY]) + _pet_phrases(_pets(present))
    return _join(items) if items else "ninguém"


def _is_name(label: str) -> bool:
    return label != UNKNOWN_KEY and not label.startswith(PET_PREFIX)


def _pets(counts: Counter[str]) -> Counter[str]:
    return Counter({label: n for label, n in counts.items() if label.startswith(PET_PREFIX) and n > 0})


def _pet_phrases(pets: Counter[str]) -> list[str]:
    phrases = []
    for label, count in sorted(pets.items()):
        kind = label[len(PET_PREFIX):]
        one, many = _PET_WORDS.get(kind, (f"um {kind}", kind))
        phrases.append(one if count == 1 else f"{count} {many}")
    return phrases


def _unknown_phrase(count: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return ["uma pessoa não reconhecida"]
    return [f"{count} pessoas não reconhecidas"]


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " e " + items[-1]
