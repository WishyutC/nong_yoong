import threading
import time
import unittest

from thai_voice_robot.runtime_state import (
    InvalidStateTransition,
    RobotState,
    RobotStateMachine,
)


class RobotStateMachineTests(unittest.TestCase):
    def test_normal_voice_turn_blocks_wake_word_until_playback_finishes(self) -> None:
        state = RobotStateMachine()
        state.transition(RobotState.IDLE)
        self.assertTrue(state.can_accept_wake_word)

        state.transition(RobotState.RECORDING, expected=RobotState.IDLE)
        self.assertFalse(state.can_accept_wake_word)
        state.transition(RobotState.THINKING, expected=RobotState.RECORDING)
        self.assertFalse(state.can_accept_wake_word)
        state.transition(RobotState.SPEAKING, expected=RobotState.THINKING)
        self.assertFalse(state.can_accept_wake_word)
        state.transition(RobotState.IDLE, expected=RobotState.SPEAKING)

        self.assertTrue(state.can_accept_wake_word)
        self.assertEqual(state.snapshot().revision, 5)

    def test_overlapping_recording_is_rejected(self) -> None:
        state = RobotStateMachine(RobotState.IDLE)
        state.transition(RobotState.RECORDING)

        with self.assertRaises(InvalidStateTransition):
            state.transition(RobotState.RECORDING, expected=RobotState.IDLE)

    def test_speaking_cannot_skip_back_to_thinking(self) -> None:
        state = RobotStateMachine(RobotState.SPEAKING)

        with self.assertRaises(InvalidStateTransition):
            state.transition(RobotState.THINKING)

    def test_wait_for_observes_worker_transition(self) -> None:
        state = RobotStateMachine(RobotState.THINKING)

        def finish_inference() -> None:
            time.sleep(0.01)
            state.transition(RobotState.SPEAKING)

        worker = threading.Thread(target=finish_inference)
        worker.start()
        self.assertTrue(state.wait_for(RobotState.SPEAKING, timeout=0.5))
        worker.join()


if __name__ == "__main__":
    unittest.main()
