"""Command-line entry point for the workstation API."""

from __future__ import annotations

import uvicorn

from thai_voice_robot.config import ServerSettings


def main() -> None:
    settings = ServerSettings.from_env()
    uvicorn.run(
        "thai_voice_robot.server.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
