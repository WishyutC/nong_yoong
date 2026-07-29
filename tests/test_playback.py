from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from thai_voice_robot.client.playback import SystemAudioPlayer


class SystemAudioPlayerTests(unittest.TestCase):
    def test_auto_prefers_ffplay_under_wsl(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            player = SystemAudioPlayer()
            with (
                patch.object(player, "_is_wsl", return_value=True),
                patch(
                    "thai_voice_robot.client.playback.shutil.which",
                    side_effect=lambda name: f"/usr/bin/{name}",
                ),
                patch("thai_voice_robot.client.playback.subprocess.run") as run,
            ):
                player.play(Path(audio.name))

        self.assertEqual(run.call_args.args[0][0], "/usr/bin/ffplay")

    def test_explicit_aplay_keeps_alsa_device(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            player = SystemAudioPlayer("plughw:1,0", "aplay")
            with (
                patch(
                    "thai_voice_robot.client.playback.shutil.which",
                    side_effect=lambda name: f"/usr/bin/{name}",
                ),
                patch("thai_voice_robot.client.playback.subprocess.run") as run,
            ):
                player.play(Path(audio.name))

        self.assertIn("plughw:1,0", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
