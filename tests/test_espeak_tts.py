from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from thai_voice_robot.server.tts import EspeakThaiTTS, inspect_pcm16_wav


@unittest.skipUnless(shutil.which("espeak-ng") or shutil.which("espeak"), "eSpeak missing")
class EspeakThaiTTSTests(unittest.TestCase):
    def test_generates_final_format_thai_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thai.wav"
            info = EspeakThaiTTS().synthesize("สวัสดีครับ", path)

            self.assertEqual(info.channels, 1)
            self.assertEqual(info.sample_width_bytes, 2)
            self.assertGreater(path.stat().st_size, 44)
            self.assertEqual(inspect_pcm16_wav(path), info)


if __name__ == "__main__":
    unittest.main()
