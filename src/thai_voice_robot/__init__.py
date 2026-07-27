"""Shared foundations for the Nong Yoong Thai voice robot."""

from .runtime_state import InvalidStateTransition, RobotState, RobotStateMachine

__all__ = ["InvalidStateTransition", "RobotState", "RobotStateMachine"]
