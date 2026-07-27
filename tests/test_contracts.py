import unittest
from uuid import uuid4

from thai_voice_robot.contracts import (
    AudioDescriptor,
    ConversationRequest,
    ConversationResponse,
)


class ContractTests(unittest.TestCase):
    def test_request_factory_creates_serializable_thai_payload(self) -> None:
        request = ConversationRequest.new("  สวัสดีครับ  ")
        payload = request.to_payload()

        self.assertEqual(payload["text"], "สวัสดีครับ")
        self.assertEqual(payload["language"], "th")
        self.assertNotEqual(payload["request_id"], payload["session_id"])

    def test_response_keeps_request_and_session_correlation(self) -> None:
        request_id = str(uuid4())
        session_id = str(uuid4())
        response = ConversationResponse(
            request_id=request_id,
            session_id=session_id,
            response_text="สวัสดีครับ",
            audio=AudioDescriptor(url=f"http://workstation:5000/v1/audio/{request_id}.wav"),
        )

        payload = response.to_payload()
        self.assertEqual(payload["request_id"], request_id)
        self.assertEqual(payload["audio"]["encoding"], "pcm_s16le")
        self.assertEqual(payload["audio"]["sample_rate_hz"], 24_000)

    def test_empty_transcript_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ConversationRequest.new("   ")

    def test_non_uuid_request_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ConversationRequest("not-an-id", str(uuid4()), "สวัสดี")


if __name__ == "__main__":
    unittest.main()
