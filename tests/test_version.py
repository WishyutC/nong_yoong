from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from thai_voice_robot import __version__
from thai_voice_robot.server.app import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VersionTests(unittest.TestCase):
    def test_package_metadata_and_api_versions_match(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as source:
            project_version = tomllib.load(source)["project"]["version"]

        self.assertEqual(project_version, "1.0.0")
        self.assertEqual(__version__, project_version)
        self.assertEqual(app.version, project_version)


if __name__ == "__main__":
    unittest.main()
