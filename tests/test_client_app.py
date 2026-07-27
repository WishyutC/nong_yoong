from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import write_test_wav
from thai_voice_robot.client.app import VoiceRobot
from thai_voice_robot.client.events import EventLogger
from thai_voice_robot.contracts import AudioDescriptor, ConversationResponse
from thai_voice_robot.runtime_state import RobotState


class FakeWorkstation:
    def __init__(self) -> None:
        self.health_calls = 0

    def health(self):
        self.health_calls += 1
        return {"status": "ready"}

    def respond(self, request):
        return ConversationResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            response_text="รับทราบครับ",
            audio=AudioDescriptor(url=f"/v1/audio/{request.request_id}.wav", sample_rate_hz=16_000),
        )

    def download_audio(self, response, destination_dir: Path) -> Path:
        path = destination_dir / f"{response.request_id}.wav"
        write_test_wav(path)
        return path


class FakePlayer:
    def __init__(self) -> None:
        self.played: list[Path] = []

    def play(self, path: Path) -> None:
        self.played.append(path)


class ClientAppTests(unittest.TestCase):
    def test_text_turn_returns_to_idle_only_after_playback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workstation = FakeWorkstation()
            player = FakePlayer()
            robot = VoiceRobot(
                workstation=workstation,
                player=player,
                events=EventLogger(root / "events.jsonl"),
                audio_dir=root / "audio",
            )
            robot.start()
            response = robot.process_text("ทดสอบระบบ")

            self.assertEqual(response.response_text, "รับทราบครับ")
            self.assertEqual(robot.state.state, RobotState.IDLE)
            self.assertEqual(len(player.played), 1)
            self.assertEqual(workstation.health_calls, 1)
            self.assertIn("turn_completed", (root / "events.jsonl").read_text())


if __name__ == "__main__":
    unittest.main()
