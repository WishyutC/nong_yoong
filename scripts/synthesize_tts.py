#!/usr/bin/env python3
"""Generate a Thai WAV directly, without starting the API server."""

from __future__ import annotations

import argparse
from pathlib import Path

from thai_voice_robot.config import ServerSettings
from thai_voice_robot.server.tts import EspeakThaiTTS, MmsThaiTTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "text",
        nargs="?",
        default="สวัสดีครับ ฉันคือน้องยูง ผู้ช่วยภาษาไทยของคุณ",
    )
    parser.add_argument(
        "--backend", choices=("mms", "espeak"), default="mms"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tts_samples/mms-thai.wav"),
    )
    parser.add_argument("--model", default="facebook/mms-tts-tha")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ServerSettings.from_env()
    if args.backend == "mms":
        backend = MmsThaiTTS(
            args.model,
            settings.mms_speaking_rate,
            settings.mms_noise_scale,
            settings.mms_noise_scale_duration,
            settings.mms_seed,
        )
    else:
        backend = EspeakThaiTTS()
    output = args.output.expanduser().resolve()
    info = backend.synthesize(args.text, output)
    print(
        f"Created {output} "
        f"({info.sample_rate_hz} Hz, {info.channels} channel, PCM{info.sample_width_bytes * 8})"
    )


if __name__ == "__main__":
    main()
