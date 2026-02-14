from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionClass(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass(frozen=True)
class Envelope:
    id: str
    ts: str
    type: str
    session_id: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class IntentPayload:
    intent: str
    action_class: ActionClass
    summary: str
    actions: List[Dict[str, Any]]


@dataclass(frozen=True)
class ConfirmationRequest:
    action_class: ActionClass
    summary: str
    token: str


@dataclass(frozen=True)
class ConfirmationResponse:
    accepted: bool
    token: str


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    message: str
    data: Optional[Dict[str, Any]] = None
