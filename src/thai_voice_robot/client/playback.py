"""Blocking audio playback so the robot remains SPEAKING until completion."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Protocol


class AudioPlayer(Protocol):
    def play(self, path: Path) -> None: ...


class NullAudioPlayer:
    def play(self, path: Path) -> None:
        return None


class SystemAudioPlayer:
    def __init__(self, output_device: str | None = None) -> None:
        self.output_device = output_device

    def play(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        aplay = shutil.which("aplay")
        if aplay:
            command = [aplay, "-q"]
            if self.output_device:
                command.extend(["-D", self.output_device])
            command.append(str(path))
        else:
            ffplay = shutil.which("ffplay")
            if not ffplay:
                raise RuntimeError("neither aplay nor ffplay is installed")
            command = [ffplay, "-nodisp", "-autoexit", "-loglevel", "error", str(path)]
        subprocess.run(command, check=True)
