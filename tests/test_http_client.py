from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import requests

from tests.helpers import write_test_wav
from thai_voice_robot.client.http_client import WorkstationAPIError, WorkstationClient
from thai_voice_robot.contracts import AudioDescriptor, ConversationRequest, ConversationResponse


class FakeResponse:
    def __init__(self, *, payload=None, content: bytes = b"", status_code: int = 200):
        self.payload = payload
        self.content = content
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self.content), max(1, min(chunk_size, 17))):
            yield self.content[start : start + max(1, min(chunk_size, 17))]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeSession:
    def __init__(self, response_payload: dict, audio: bytes):
        self.response_payload = response_payload
        self.audio = audio

    def post(self, *args, **kwargs):
        return FakeResponse(payload=self.response_payload)

    def get(self, url, *args, **kwargs):
        if url.endswith("/health/ready"):
            return FakeResponse(payload={"status": "ready"})
        return FakeResponse(content=self.audio)


class HTTPClientTests(unittest.TestCase):
    def test_correlated_response_and_streamed_wav_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            write_test_wav(source, frames=1000)
            request = ConversationRequest.new("สวัสดี")
            payload = ConversationResponse(
                request_id=request.request_id,
                session_id=request.session_id,
                response_text="สวัสดีครับ",
                audio=AudioDescriptor(
                    url=f"/v1/audio/{request.request_id}.wav", sample_rate_hz=16_000
                ),
            ).to_payload()
            client = WorkstationClient("http://workstation:5000", chunk_size=64)
            client._session = FakeSession(payload, source.read_bytes())

            response = client.respond(request)
            destination = client.download_audio(response, root / "downloads")

            self.assertEqual(response.request_id, request.request_id)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertFalse(destination.with_suffix(".wav.part").exists())

    def test_external_audio_host_is_rejected(self) -> None:
        response = ConversationResponse(
            request_id=str(uuid4()),
            session_id=str(uuid4()),
            response_text="ทดสอบ",
            audio=AudioDescriptor(url="http://unexpected.example/audio.wav"),
        )
        client = WorkstationClient("http://workstation:5000")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(WorkstationAPIError):
                client.download_audio(response, Path(directory))


if __name__ == "__main__":
    unittest.main()
