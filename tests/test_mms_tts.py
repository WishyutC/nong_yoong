from __future__ import annotations

import unittest
from unittest.mock import Mock
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

    def test_optimizer_values_are_applied_to_model(self) -> None:
        model = Mock()
        model.eval.return_value = model
        tts = MmsThaiTTS(
            speaking_rate=0.95,
            noise_scale=0.55,
            noise_scale_duration=0.65,
            seed=555,
        )

        configured = tts._configure_model(model)

        self.assertIs(configured, model)
        self.assertEqual(model.speaking_rate, 0.95)
        self.assertEqual(model.noise_scale, 0.55)
        self.assertEqual(model.noise_scale_duration, 0.65)

    def test_invalid_optimizer_value_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MmsThaiTTS(speaking_rate=0.0)


if __name__ == "__main__":
    unittest.main()
