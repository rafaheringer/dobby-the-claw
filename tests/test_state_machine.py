"""A conversation turn goes all the way round and back to IDLE."""

from bridge.state_machine import Event, State, StateMachine


def test_a_turn_ends_idle():
    sm = StateMachine()
    for event, state in [
        (Event.WAKE_WORD, State.LISTENING),
        (Event.STT_RECEIVED, State.THINKING),
        (Event.RESPONSE_READY, State.EXECUTING),  # the answer is out, being played
        (Event.RESPONSE_READY, State.IDLE),  # played to the end
    ]:
        assert sm.transition(event) == state


def test_speaking_over_the_answer_starts_a_new_turn():
    sm = StateMachine()
    sm.state = State.EXECUTING
    assert sm.transition(Event.WAKE_WORD) == State.LISTENING
    assert sm.transition(Event.STT_RECEIVED) == State.THINKING


def test_a_typed_turn_thinks_straight_away():
    sm = StateMachine()
    assert sm.transition(Event.STT_RECEIVED) == State.THINKING
