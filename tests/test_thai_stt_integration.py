from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from thai_voice_robot.client.stt import ThaiZipformerRecognizer
from thai_voice_robot.config import PROJECT_ROOT
from thai_voice_robot.server.tts import EspeakThaiTTS


MODEL_DIR = PROJECT_ROOT / "models" / "sherpa-onnx-zipformer-thai-2024-06-20"
HAS_MODEL = (MODEL_DIR / ThaiZipformerRecognizer.ENCODER).is_file()
HAS_RUNTIME = importlib.util.find_spec("sherpa_onnx") is not None


@unittest.skipUnless(HAS_MODEL, "Thai Zipformer model not downloaded")
@unittest.skipUnless(HAS_RUNTIME, "sherpa-onnx runtime not installed")
@unittest.skipUnless(shutil.which("espeak-ng") or shutil.which("espeak"), "eSpeak missing")
class ThaiSTTIntegrationTests(unittest.TestCase):
    def test_real_model_decodes_pcm16_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "thai.wav"
            EspeakThaiTTS().synthesize("สวัสดีครับ วันนี้อากาศดี", wav_path)
            transcript = ThaiZipformerRecognizer(MODEL_DIR).transcribe(wav_path)

            self.assertIsInstance(transcript, str)
            self.assertTrue(transcript)


if __name__ == "__main__":
    unittest.main()
