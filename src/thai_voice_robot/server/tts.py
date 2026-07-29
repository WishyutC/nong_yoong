"""Thai TTS backends that write final mono PCM16 WAV files."""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from threading import Lock
from typing import Protocol

import requests

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

    def __init__(
        self,
        model_name: str = "facebook/mms-tts-tha",
        speaking_rate: float = 0.95,
        noise_scale: float = 0.55,
        noise_scale_duration: float = 0.65,
        seed: int = 555,
    ) -> None:
        if not 0.5 <= speaking_rate <= 2.0:
            raise ValueError("MMS speaking rate must be between 0.5 and 2.0")
        if not 0.0 <= noise_scale <= 2.0:
            raise ValueError("MMS noise scale must be between 0.0 and 2.0")
        if not 0.0 <= noise_scale_duration <= 2.0:
            raise ValueError("MMS duration noise must be between 0.0 and 2.0")
        if seed < 0:
            raise ValueError("MMS seed must be zero or greater")
        self.model_name = model_name
        self.speaking_rate = speaking_rate
        self.noise_scale = noise_scale
        self.noise_scale_duration = noise_scale_duration
        self.seed = seed
        self._model = None
        self._tokenizer = None
        self._lock = Lock()

    def _configure_model(self, model):
        model.speaking_rate = self.speaking_rate
        model.noise_scale = self.noise_scale
        model.noise_scale_duration = self.noise_scale_duration
        return model.eval()

    def _load(self):
        with self._lock:
            if self._model is None or self._tokenizer is None:
                try:
                    from transformers import AutoTokenizer, VitsModel

                    try:
                        tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name, local_files_only=True
                        )
                        model = VitsModel.from_pretrained(
                            self.model_name, local_files_only=True
                        )
                    except OSError:
                        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                        model = VitsModel.from_pretrained(self.model_name)
                    self._tokenizer = tokenizer
                    self._model = self._configure_model(model)
                except Exception as exc:
                    raise BackendUnavailable(f"could not load MMS Thai TTS: {exc}") from exc
            return self._tokenizer, self._model

    def synthesize(self, text: str, destination: Path) -> WavInfo:
        try:
            import numpy as np
            import torch

            tokenizer, model = self._load()
            inputs = tokenizer(text, return_tensors="pt")
            torch.manual_seed(self.seed)
            with torch.inference_mode():
                waveform = model(
                    **inputs, speaking_rate=self.speaking_rate
                ).waveform
            samples = np.asarray(
                waveform.squeeze().detach().cpu().numpy(), dtype=np.float32
            ).reshape(-1)
            sample_rate = int(model.config.sampling_rate)
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
        missing = [name for name in ("torch", "transformers") if find_spec(name) is None]
        if missing:
            return False, (
                "MMS dependencies missing: "
                + ", ".join(missing)
                + "; install the mms optional dependencies"
            )
        if self._model is None:
            state = "configured; lazy load pending"
        else:
            state = "loaded"
        return True, (
            f"MMS model {self.model_name} {state}; rate={self.speaking_rate}, "
            f"noise={self.noise_scale}, duration_noise={self.noise_scale_duration}, "
            f"seed={self.seed}"
        )


class ThonburianThaiTTS:
    """Client for the isolated, workstation-local Thonburian F5-TTS service."""

    MAX_AUDIO_BYTES = 128 * 1024 * 1024

    def __init__(self, base_url: str, timeout_seconds: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def synthesize(self, text: str, destination: Path) -> WavInfo:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with requests.post(
                f"{self.base_url}/synthesize",
                json={"text": text},
                stream=True,
                timeout=(5.0, self.timeout_seconds),
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0]
                if content_type != "audio/wav":
                    raise BackendUnavailable("Thonburian service returned non-WAV data")
                try:
                    declared_size = int(response.headers.get("content-length", "0"))
                except ValueError as exc:
                    raise BackendUnavailable(
                        "Thonburian service returned an invalid content length"
                    ) from exc
                if declared_size > self.MAX_AUDIO_BYTES:
                    raise BackendUnavailable("Thonburian WAV exceeds the size limit")
                received = 0
                with destination.open("wb") as output:
                    for chunk in response.iter_content(64 * 1024):
                        if chunk:
                            received += len(chunk)
                            if received > self.MAX_AUDIO_BYTES:
                                raise BackendUnavailable(
                                    "Thonburian WAV exceeds the size limit"
                                )
                            output.write(chunk)
        except BackendUnavailable:
            destination.unlink(missing_ok=True)
            raise
        except (OSError, requests.RequestException) as exc:
            destination.unlink(missing_ok=True)
            raise BackendUnavailable(f"Thonburian synthesis failed: {exc}") from exc
        return inspect_pcm16_wav(destination)

    def health(self) -> tuple[bool, str]:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=3.0)
            response.raise_for_status()
            payload = response.json()
        except (ValueError, requests.RequestException) as exc:
            return False, f"Thonburian service unavailable: {exc}"
        return True, str(payload.get("detail", "Thonburian service ready"))
