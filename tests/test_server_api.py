from __future__ import annotations

import tempfile
import unittest
import asyncio
from pathlib import Path
from uuid import uuid4

import httpx

from tests.helpers import CountingLLM, FakeTTS
from thai_voice_robot.config import ServerSettings
from thai_voice_robot.server.app import create_app
from thai_voice_robot.server.service import ResponseService
from thai_voice_robot.server.store import ConversationStore


API_TOKEN = "test-token-that-is-at-least-32-characters-long"


class ServerAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.settings = ServerSettings(
            audio_dir=root / "audio",
            database_path=root / "conversations.sqlite3",
            llm_backend="echo",
            tts_backend="espeak",
            api_token=API_TOKEN,
        )
        self.llm = CountingLLM()
        self.service = ResponseService(
            self.settings,
            ConversationStore(self.settings.database_path),
            self.llm,
            FakeTTS(),
        )
        self.app = create_app(self.settings, self.service)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def request(self, method: str, path: str, *, authenticated: bool = True, **kwargs):
        async def send():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                headers = dict(kwargs.pop("headers", {}))
                if authenticated:
                    headers["Authorization"] = f"Bearer {API_TOKEN}"
                kwargs["headers"] = headers
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def test_response_audio_history_and_idempotency(self) -> None:
        request_id = str(uuid4())
        session_id = str(uuid4())
        payload = {
            "request_id": request_id,
            "session_id": session_id,
            "text": "สวัสดี",
            "language": "th",
        }

        first = self.request("POST", "/v1/responses", json=payload)
        second = self.request("POST", "/v1/responses", json=payload)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json(), second.json())
        self.assertEqual(self.llm.calls, 1)
        self.assertEqual(first.json()["audio"]["encoding"], "pcm_s16le")

        audio = self.request("GET", first.json()["audio"]["url"])
        self.assertEqual(audio.status_code, 200)
        self.assertTrue(audio.content.startswith(b"RIFF"))

        history = self.request("GET", f"/v1/sessions/{session_id}/events").json()
        self.assertEqual([event["role"] for event in history], ["user", "assistant"])
        summary = self.request("GET", "/v1/dashboard/summary").json()
        self.assertEqual(summary["conversation_turns"], 1)
        self.assertEqual(summary["session_count"], 1)

    def test_readiness_reports_components(self) -> None:
        response = self.request("GET", "/health/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")

    def test_bearer_token_protects_api_but_not_liveness(self) -> None:
        live = self.request("GET", "/health/live", authenticated=False)
        missing = self.request("GET", "/health/ready", authenticated=False)
        invalid = self.request(
            "GET",
            "/v1/dashboard/summary",
            authenticated=False,
            headers={"Authorization": "Bearer wrong-token"},
        )

        self.assertEqual(live.status_code, 200)
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.headers["www-authenticate"], "Bearer")
        self.assertEqual(missing.json()["code"], "unauthorized")
        self.assertEqual(invalid.status_code, 401)

    def test_same_request_id_cannot_move_to_another_session(self) -> None:
        request_id = str(uuid4())
        first = {
            "request_id": request_id,
            "session_id": str(uuid4()),
            "text": "หนึ่ง",
        }
        second = {**first, "session_id": str(uuid4()), "text": "สอง"}
        self.assertEqual(self.request("POST", "/v1/responses", json=first).status_code, 200)
        self.assertEqual(self.request("POST", "/v1/responses", json=second).status_code, 409)

    def test_same_request_id_cannot_change_text(self) -> None:
        payload = {
            "request_id": str(uuid4()),
            "session_id": str(uuid4()),
            "text": "ข้อความแรก",
        }
        self.assertEqual(self.request("POST", "/v1/responses", json=payload).status_code, 200)
        changed = {**payload, "text": "ข้อความใหม่"}
        self.assertEqual(self.request("POST", "/v1/responses", json=changed).status_code, 409)


if __name__ == "__main__":
    unittest.main()
