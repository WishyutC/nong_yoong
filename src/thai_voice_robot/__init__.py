"""Shared foundations for the Nong Yoong Thai voice robot."""

from .runtime_state import InvalidStateTransition, RobotState, RobotStateMachine

__version__ = "1.0.0"

__all__ = [
    "InvalidStateTransition",
    "RobotState",
    "RobotStateMachine",
    "__version__",
]
