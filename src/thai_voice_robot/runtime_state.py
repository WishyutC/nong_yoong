"""Thread-safe runtime state shared by wake word, ASR, API, and playback."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import Condition, RLock
from time import monotonic


class RobotState(StrEnum):
    STARTING = "starting"
    IDLE = "idle"
    RECORDING = "recording"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"
    OFFLINE = "offline"
    STOPPING = "stopping"


class InvalidStateTransition(RuntimeError):
    """Raised when one component attempts an unsafe state transition."""


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    state: RobotState
    revision: int
    changed_at: float


class RobotStateMachine:
    """Serialize lifecycle transitions across the robot's worker threads.

    Wake-word detection is allowed only while IDLE. In particular, THINKING and
    SPEAKING remain exclusive until acoustic echo cancellation and barge-in are
    implemented.
    """

    _ALLOWED: dict[RobotState, frozenset[RobotState]] = {
        RobotState.STARTING: frozenset(
            {RobotState.IDLE, RobotState.ERROR, RobotState.OFFLINE, RobotState.STOPPING}
        ),
        RobotState.IDLE: frozenset(
            {
                RobotState.RECORDING,
                RobotState.THINKING,
                RobotState.ERROR,
                RobotState.OFFLINE,
                RobotState.STOPPING,
            }
        ),
        RobotState.RECORDING: frozenset(
            {RobotState.THINKING, RobotState.IDLE, RobotState.ERROR, RobotState.STOPPING}
        ),
        RobotState.THINKING: frozenset(
            {RobotState.SPEAKING, RobotState.ERROR, RobotState.OFFLINE, RobotState.STOPPING}
        ),
        RobotState.SPEAKING: frozenset(
            {RobotState.IDLE, RobotState.ERROR, RobotState.STOPPING}
        ),
        RobotState.ERROR: frozenset(
            {RobotState.IDLE, RobotState.OFFLINE, RobotState.STOPPING}
        ),
        RobotState.OFFLINE: frozenset(
            {RobotState.IDLE, RobotState.ERROR, RobotState.STOPPING}
        ),
        RobotState.STOPPING: frozenset(),
    }

    def __init__(self, initial: RobotState = RobotState.STARTING) -> None:
        self._condition = Condition(RLock())
        self._snapshot = StateSnapshot(initial, 0, monotonic())

    @property
    def state(self) -> RobotState:
        with self._condition:
            return self._snapshot.state

    @property
    def can_accept_wake_word(self) -> bool:
        return self.state is RobotState.IDLE

    def snapshot(self) -> StateSnapshot:
        with self._condition:
            return self._snapshot

    def transition(
        self,
        new_state: RobotState,
        *,
        expected: RobotState | None = None,
    ) -> StateSnapshot:
        """Move atomically to ``new_state`` and return the new snapshot.

        ``expected`` protects asynchronous workers from completing a stale turn.
        A transition to the current state is idempotent and does not increment
        the revision.
        """

        with self._condition:
            current = self._snapshot.state
            if expected is not None and current is not expected:
                raise InvalidStateTransition(
                    f"expected {expected.value}, found {current.value}"
                )
            if new_state is current:
                return self._snapshot
            if new_state not in self._ALLOWED[current]:
                raise InvalidStateTransition(
                    f"cannot transition from {current.value} to {new_state.value}"
                )

            self._snapshot = StateSnapshot(
                state=new_state,
                revision=self._snapshot.revision + 1,
                changed_at=monotonic(),
            )
            self._condition.notify_all()
            return self._snapshot

    def wait_for(self, state: RobotState, timeout: float | None = None) -> bool:
        """Wait until ``state`` is reached, returning false after ``timeout``."""

        with self._condition:
            return self._condition.wait_for(
                lambda: self._snapshot.state is state,
                timeout=timeout,
            )
