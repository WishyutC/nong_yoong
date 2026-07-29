"""Blocking audio playback so the robot remains SPEAKING until completion."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Protocol


class AudioPlayer(Protocol):
    def play(self, path: Path) -> None: ...


class NullAudioPlayer:
    def play(self, path: Path) -> None:
        return None


class SystemAudioPlayer:
    def __init__(
        self, output_device: str | None = None, preferred_player: str = "auto"
    ) -> None:
        if preferred_player not in {"auto", "aplay", "ffplay"}:
            raise ValueError("audio player must be auto, aplay, or ffplay")
        self.output_device = output_device
        self.preferred_player = preferred_player

    @staticmethod
    def _is_wsl() -> bool:
        try:
            return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
        except OSError:
            return sys.platform.startswith("win")

    def play(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        aplay = shutil.which("aplay")
        ffplay = shutil.which("ffplay")
        selected = self.preferred_player
        if selected == "auto":
            # WSL has no native ALSA sound card in many installations. FFplay
            # routes through WSLg/PulseAudio and resamples 24 kHz speech safely.
            selected = (
                "ffplay"
                if self._is_wsl() and not self.output_device and ffplay
                else "aplay"
                if aplay
                else "ffplay"
            )

        if selected == "aplay" and aplay:
            command = [aplay, "-q"]
            if self.output_device:
                command.extend(["-D", self.output_device])
            command.append(str(path))
        elif selected == "ffplay" and ffplay:
            command = [ffplay, "-nodisp", "-autoexit", "-loglevel", "error", str(path)]
        else:
            raise RuntimeError(f"requested audio player is not installed: {selected}")
        subprocess.run(command, check=True)
