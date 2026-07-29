from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from thai_voice_robot.server.backends import BackendUnavailable
from thai_voice_robot.server.tts import ThonburianThaiTTS


def wav_bytes() -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
        with wave.open(temporary.name, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24_000)
            output.writeframes(b"\x00\x00" * 240)
        return Path(temporary.name).read_bytes()


class ThonburianThaiTTSTests(unittest.TestCase):
    def test_health_calls_local_sidecar(self) -> None:
        response = Mock()
        response.json.return_value = {"detail": "ThonburianTTS loaded on cpu"}
        response.raise_for_status.return_value = None
        with patch("thai_voice_robot.server.tts.requests.get", return_value=response):
            ready, detail = ThonburianThaiTTS("http://127.0.0.1:5055").health()

        self.assertTrue(ready)
        self.assertIn("loaded on cpu", detail)

    def test_synthesis_streams_and_validates_wav(self) -> None:
        response = Mock()
        response.headers = {"content-type": "audio/wav"}
        response.raise_for_status.return_value = None
        response.iter_content.return_value = [wav_bytes()]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "reply.wav"
            with patch(
                "thai_voice_robot.server.tts.requests.post", return_value=response
            ):
                info = ThonburianThaiTTS("http://127.0.0.1:5055").synthesize(
                    "สวัสดี", destination
                )

        self.assertEqual(info.sample_rate_hz, 24_000)
        self.assertEqual(info.channels, 1)

    def test_non_wav_response_is_rejected(self) -> None:
        response = Mock()
        response.headers = {"content-type": "application/json"}
        response.raise_for_status.return_value = None
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "thai_voice_robot.server.tts.requests.post", return_value=response
            ):
                with self.assertRaises(BackendUnavailable):
                    ThonburianThaiTTS("http://127.0.0.1:5055").synthesize(
                        "สวัสดี", Path(directory) / "reply.wav"
                    )


if __name__ == "__main__":
    unittest.main()
