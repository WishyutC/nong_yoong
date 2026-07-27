from __future__ import annotations

import wave
from pathlib import Path

from thai_voice_robot.server.tts import WavInfo


def write_test_wav(path: Path, frames: int = 160) -> WavInfo:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16_000)
        wav_file.writeframes(b"\x00\x00" * frames)
    return WavInfo(16_000, 1, 2)


class FakeTTS:
    def synthesize(self, text: str, destination: Path) -> WavInfo:
        return write_test_wav(destination)

    def health(self) -> tuple[bool, str]:
        return True, "fake TTS ready"


class CountingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages) -> str:
        self.calls += 1
        return "สวัสดีครับ"

    def health(self) -> tuple[bool, str]:
        return True, "fake LLM ready"
