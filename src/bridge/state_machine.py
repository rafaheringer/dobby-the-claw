from dataclasses import dataclass
from enum import Enum
from typing import Optional


class State(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    EXECUTING = "EXECUTING"
    CONFIRMING = "CONFIRMING"
    ERROR = "ERROR"


class Event(str, Enum):
    WAKE_WORD = "WAKE_WORD"
    STT_RECEIVED = "STT_RECEIVED"
    TIMEOUT = "TIMEOUT"
    RESPONSE_READY = "RESPONSE_READY"
    MODEL_ERROR = "MODEL_ERROR"
    NEED_CONFIRMATION = "NEED_CONFIRMATION"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


@dataclass
class Transition:
    from_state: State
    to_state: State
    event: Event


class StateMachine:
    def __init__(self) -> None:
        self.state = State.IDLE

    def transition(self, event: Event) -> State:
        if self.state == State.IDLE:
            if event == Event.WAKE_WORD:
                self.state = State.LISTENING
        elif self.state == State.LISTENING:
            if event == Event.STT_RECEIVED:
                self.state = State.THINKING
            elif event == Event.TIMEOUT:
                self.state = State.IDLE
        elif self.state == State.THINKING:
            if event == Event.RESPONSE_READY:
                self.state = State.EXECUTING
            elif event == Event.MODEL_ERROR:
                self.state = State.ERROR
        elif self.state == State.EXECUTING:
            if event == Event.NEED_CONFIRMATION:
                self.state = State.CONFIRMING
            elif event == Event.RESPONSE_READY:
                self.state = State.IDLE
        elif self.state == State.CONFIRMING:
            if event == Event.CONFIRMED:
                self.state = State.EXECUTING
            elif event in (Event.REJECTED, Event.TIMEOUT):
                self.state = State.IDLE
        elif self.state == State.ERROR:
            self.state = State.IDLE

        return self.state
