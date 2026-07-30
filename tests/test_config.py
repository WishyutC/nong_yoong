from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from thai_voice_robot.config import ClientSettings, ServerSettings


class AuthenticationConfigTests(unittest.TestCase):
    def test_shared_strong_token_loads_for_server_and_client(self) -> None:
        token = "a" * 64
        with patch.dict(os.environ, {"NONG_YOONG_API_TOKEN": token}):
            self.assertEqual(ServerSettings.from_env().api_token, token)
            self.assertEqual(ClientSettings.from_env().api_token, token)

    def test_short_configured_token_is_rejected(self) -> None:
        with patch.dict(os.environ, {"NONG_YOONG_API_TOKEN": "too-short"}):
            with self.assertRaisesRegex(ValueError, "at least 32 characters"):
                ServerSettings.from_env()


if __name__ == "__main__":
    unittest.main()
