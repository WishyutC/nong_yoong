from __future__ import annotations

import unittest
from unittest.mock import patch

from thai_voice_robot.server.tts import MmsThaiTTS


class MmsThaiTTSTests(unittest.TestCase):
    def test_health_reports_missing_optional_dependencies(self) -> None:
        with patch("thai_voice_robot.server.tts.find_spec", return_value=None):
            ready, detail = MmsThaiTTS().health()

        self.assertFalse(ready)
        self.assertIn("torch", detail)
        self.assertIn("transformers", detail)

    def test_health_is_ready_before_lazy_model_load(self) -> None:
        with patch("thai_voice_robot.server.tts.find_spec", return_value=object()):
            ready, detail = MmsThaiTTS().health()

        self.assertTrue(ready)
        self.assertIn("lazy load pending", detail)


if __name__ == "__main__":
    unittest.main()
