#!/usr/bin/env python3
"""Isolated local HTTP service for the heavyweight Thonburian F5-TTS model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

SOURCE_DIR = Path(
    os.getenv(
        "NONG_YOONG_THONBURIAN_SOURCE",
        PROJECT_ROOT / "models" / "thonburian-tts",
    )
).expanduser().resolve()
REFERENCE_AUDIO = Path(
    os.getenv(
        "NONG_YOONG_THONBURIAN_REF_AUDIO", SOURCE_DIR / "assets" / "000000.wav"
    )
).expanduser().resolve()
REFERENCE_TEXT = os.getenv("NONG_YOONG_THONBURIAN_REF_TEXT", "ใครเป็นผู้รับ")
CHECKPOINT = os.getenv(
    "NONG_YOONG_THONBURIAN_CHECKPOINT",
    "hf://biodatlab/ThonburianTTS/megaF5/mega_f5_last.safetensors",
)
VOCAB_FILE = os.getenv(
    "NONG_YOONG_THONBURIAN_VOCAB",
    "hf://biodatlab/ThonburianTTS/megaF5/mega_vocab.txt",
)
DEVICE = os.getenv("NONG_YOONG_THONBURIAN_DEVICE", "auto")
NFE_STEP = int(os.getenv("NONG_YOONG_THONBURIAN_NFE_STEP", "8"))
CFG_STRENGTH = float(os.getenv("NONG_YOONG_THONBURIAN_CFG", "2.5"))
SPEED = float(os.getenv("NONG_YOONG_THONBURIAN_SPEED", "1.0"))
TEMP_DIR = Path(
    os.getenv(
        "NONG_YOONG_THONBURIAN_TEMP",
        PROJECT_ROOT / "data" / "thonburian-temp",
    )
).expanduser().resolve()

if not 4 <= NFE_STEP <= 64:
    raise ValueError("NONG_YOONG_THONBURIAN_NFE_STEP must be between 4 and 64")
if SPEED <= 0:
    raise ValueError("NONG_YOONG_THONBURIAN_SPEED must be greater than zero")

os.environ.setdefault("MPLCONFIGDIR", str(TEMP_DIR / "matplotlib"))


class Engine:
    def __init__(self) -> None:
        self.pipeline: Any = None
        self.device = ""
        self.lock = Lock()

    def preload(self) -> None:
        if self.pipeline is not None:
            return
        if not (SOURCE_DIR / "flowtts" / "inference.py").is_file():
            raise RuntimeError(f"Thonburian source missing: {SOURCE_DIR}")
        if not REFERENCE_AUDIO.is_file():
            raise RuntimeError(f"Thonburian reference WAV missing: {REFERENCE_AUDIO}")
        sys.path.insert(0, str(SOURCE_DIR))
        import torch
        from flowtts.inference import AudioConfig, FlowTTSPipeline, ModelConfig

        self.device = (
            DEVICE
            if DEVICE != "auto"
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        model_config = ModelConfig(
            language="th",
            model_type="F5",
            checkpoint=CHECKPOINT,
            vocab_file=VOCAB_FILE,
            vocoder="vocos",
            device=self.device,
        )
        audio_config = AudioConfig(
            silence_threshold=-45,
            cfg_strength=CFG_STRENGTH,
            nfe_step=NFE_STEP,
            speed=SPEED,
        )
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.pipeline = FlowTTSPipeline(
            model_config=model_config,
            audio_config=audio_config,
            temp_dir=str(TEMP_DIR),
        )

    def synthesize(self, text: str, destination: Path) -> None:
        with self.lock:
            self.pipeline(
                text=text,
                ref_voice=str(REFERENCE_AUDIO),
                ref_text=REFERENCE_TEXT,
                output_file=str(destination),
                speed=SPEED,
            )


def json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    engine: Engine

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self.send_json(404, {"detail": "not found"})
            return
        self.send_json(
            200,
            {
                "status": "ready",
                "detail": (
                    f"ThonburianTTS loaded on {self.engine.device}; "
                    f"NFE={NFE_STEP}, speed={SPEED}"
                ),
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/synthesize":
            self.send_json(404, {"detail": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 64 * 1024:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(content_length))
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip() or len(text) > 12_000:
                raise ValueError("text must contain 1 to 12000 characters")
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                suffix=".wav", dir=TEMP_DIR, delete=False
            ) as temporary:
                output_path = Path(temporary.name)
            try:
                self.engine.synthesize(text.strip(), output_path)
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(output_path.stat().st_size))
                self.end_headers()
                with output_path.open("rb") as audio:
                    shutil.copyfileobj(audio, self.wfile, length=64 * 1024)
            finally:
                output_path.unlink(missing_ok=True)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"detail": str(exc)})
        except Exception as exc:
            self.send_json(503, {"detail": f"synthesis failed: {exc}"})

    def log_message(self, message: str, *args: Any) -> None:
        print(f"Thonburian HTTP: {message % args}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default=os.getenv("NONG_YOONG_THONBURIAN_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("NONG_YOONG_THONBURIAN_PORT", "5055")),
    )
    parser.add_argument(
        "--preload", action="store_true", help="load/download models, then exit"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = Engine()
    engine.preload()
    print(f"ThonburianTTS loaded on {engine.device} (NFE={NFE_STEP}, speed={SPEED})")
    if args.preload:
        return
    Handler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"ThonburianTTS listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
