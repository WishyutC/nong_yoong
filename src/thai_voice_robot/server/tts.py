"""Thai TTS backends that write final mono PCM16 WAV files."""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

from thai_voice_robot.server.backends import BackendUnavailable


@dataclass(frozen=True, slots=True)
class WavInfo:
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int


def inspect_pcm16_wav(path: Path) -> WavInfo:
    try:
        with wave.open(str(path), "rb") as wav_file:
            info = WavInfo(
                sample_rate_hz=wav_file.getframerate(),
                channels=wav_file.getnchannels(),
                sample_width_bytes=wav_file.getsampwidth(),
            )
            compression = wav_file.getcomptype()
            frames = wav_file.getnframes()
    except (OSError, wave.Error) as exc:
        raise BackendUnavailable(f"invalid WAV output: {exc}") from exc

    if info.channels != 1 or info.sample_width_bytes != 2 or compression != "NONE":
        raise BackendUnavailable("TTS output must be mono uncompressed PCM16 WAV")
    if frames <= 0:
        raise BackendUnavailable("TTS output contains no audio frames")
    return info


class TTSBackend(Protocol):
    def synthesize(self, text: str, destination: Path) -> WavInfo: ...

    def health(self) -> tuple[bool, str]: ...


class EspeakThaiTTS:
    """Always-lightweight Thai fallback using the workstation's eSpeak NG."""

    def __init__(self, voice: str = "th", speed: int = 155) -> None:
        self.voice = voice
        self.speed = speed

    def synthesize(self, text: str, destination: Path) -> WavInfo:
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if not executable:
            raise BackendUnavailable("espeak-ng is not installed")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    executable,
                    "-v",
                    self.voice,
                    "-s",
                    str(self.speed),
                    "-w",
                    str(destination),
                    text,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise BackendUnavailable(f"eSpeak synthesis failed: {exc}") from exc
        return inspect_pcm16_wav(destination)

    def health(self) -> tuple[bool, str]:
        executable = shutil.which("espeak-ng") or shutil.which("espeak")
        if executable:
            return True, f"Thai eSpeak available at {executable}"
        return False, "espeak-ng is not installed"


class MmsThaiTTS:
    """Neural Thai MMS TTS. The model is loaded lazily on first use.

    facebook/mms-tts-tha is CC-BY-NC-4.0 and therefore must not be used for a
    commercial deployment without appropriate permission.
    """

    def __init__(self, model_name: str = "facebook/mms-tts-tha") -> None:
        self.model_name = model_name
        self._pipeline = None
        self._lock = Lock()

    def _load(self):
        with self._lock:
            if self._pipeline is None:
                try:
                    from transformers import pipeline

                    self._pipeline = pipeline("text-to-speech", model=self.model_name)
                except Exception as exc:
                    raise BackendUnavailable(f"could not load MMS Thai TTS: {exc}") from exc
            return self._pipeline

    def synthesize(self, text: str, destination: Path) -> WavInfo:
        try:
            import numpy as np

            result = self._load()(text)
            samples = np.asarray(result["audio"], dtype=np.float32).reshape(-1)
            sample_rate = int(result["sampling_rate"])
            samples = np.clip(samples, -1.0, 1.0)
            pcm = (samples * 32767.0).astype("<i2").tobytes()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(destination), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(pcm)
        except BackendUnavailable:
            raise
        except Exception as exc:
            raise BackendUnavailable(f"MMS synthesis failed: {exc}") from exc
        return inspect_pcm16_wav(destination)

    def health(self) -> tuple[bool, str]:
        if self._pipeline is None:
            return True, f"MMS model {self.model_name} configured; lazy load pending"
        return True, f"MMS model {self.model_name} loaded"
