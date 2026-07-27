#!/usr/bin/env python3
"""Report readiness without starting a microphone or changing system state."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys

from thai_voice_robot.config import ClientSettings, ServerSettings
from thai_voice_robot.client.stt import ThaiZipformerRecognizer


def mark(ok: bool) -> str:
    return "OK" if ok else "MISSING"


def main() -> None:
    client = ClientSettings.from_env()
    server = ServerSettings.from_env()
    checks = {
        "Python 3.11": sys.version_info[:2] == (3, 11),
        "espeak-ng": bool(shutil.which("espeak-ng") or shutil.which("espeak")),
        "aplay or ffplay": bool(shutil.which("aplay") or shutil.which("ffplay")),
        "FastAPI": importlib.util.find_spec("fastapi") is not None,
        "requests": importlib.util.find_spec("requests") is not None,
        "sherpa-onnx (Pi voice mode)": importlib.util.find_spec("sherpa_onnx") is not None,
        "sounddevice (Pi voice mode)": importlib.util.find_spec("sounddevice") is not None,
        "WebRTC VAD (Pi voice mode)": importlib.util.find_spec("webrtcvad") is not None,
    }
    model_files = [
        ThaiZipformerRecognizer.ENCODER,
        ThaiZipformerRecognizer.DECODER,
        ThaiZipformerRecognizer.JOINER,
        ThaiZipformerRecognizer.TOKENS,
    ]
    checks["Thai STT model"] = all(
        (client.stt_model_dir / filename).is_file() for filename in model_files
    )

    print(f"Platform: {platform.platform()}")
    print(f"Server URL: {client.server_url}")
    print(f"Ollama: {server.ollama_url} model={server.ollama_model}")
    print(f"TTS backend: {server.tts_backend}")
    for name, ok in checks.items():
        print(f"[{mark(ok):7}] {name}")


if __name__ == "__main__":
    main()
