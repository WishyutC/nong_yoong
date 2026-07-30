#!/usr/bin/env python3
"""Isolated HTTP service for Qwen3-TTS 1.7B Thai voice cloning."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import soundfile as sf

from qwen_sidecar_utils import clean_text, silence_sample_count, split_text_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def env_path(name: str, default: Path) -> Path:
    path = Path(os.getenv(name, str(default))).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


MODEL_ID = os.getenv(
    "NONG_YOONG_QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
)
DEVICE = os.getenv("NONG_YOONG_QWEN_TTS_DEVICE", "cuda:0")
DTYPE = os.getenv("NONG_YOONG_QWEN_TTS_DTYPE", "bfloat16")
ATTENTION = os.getenv("NONG_YOONG_QWEN_TTS_ATTN", "sdpa")
LANGUAGE = os.getenv("NONG_YOONG_QWEN_TTS_LANGUAGE", "Auto")
MAX_NEW_TOKENS = int(os.getenv("NONG_YOONG_QWEN_TTS_MAX_NEW_TOKENS", "2048"))
SPEAKING_SPEED = float(os.getenv("NONG_YOONG_QWEN_TTS_SPEED", "0.90"))
MAX_CHUNK_CHARS = int(os.getenv("NONG_YOONG_QWEN_TTS_MAX_CHUNK_CHARS", "160"))
CHUNK_PAUSE_MS = int(os.getenv("NONG_YOONG_QWEN_TTS_CHUNK_PAUSE_MS", "180"))
END_PADDING_MS = int(os.getenv("NONG_YOONG_QWEN_TTS_END_PADDING_MS", "500"))
REFERENCE_AUDIO = env_path(
    "NONG_YOONG_QWEN_TTS_REF_AUDIO",
    PROJECT_ROOT / "data" / "tts_samples" / "mms-thai-optimized.wav",
)
REFERENCE_TEXT = os.getenv(
    "NONG_YOONG_QWEN_TTS_REF_TEXT",
    "สวัสดีค่ะ ฉันคือน้องยูง ยินดีที่ได้รู้จักค่ะ",
).strip()
TEMP_DIR = env_path(
    "NONG_YOONG_QWEN_TTS_TEMP", PROJECT_ROOT / "data" / "qwen-tts-temp"
)

if not 128 <= MAX_NEW_TOKENS <= 4096:
    raise ValueError("NONG_YOONG_QWEN_TTS_MAX_NEW_TOKENS must be 128 to 4096")
if not 0.75 <= SPEAKING_SPEED <= 1.25:
    raise ValueError("NONG_YOONG_QWEN_TTS_SPEED must be between 0.75 and 1.25")
if not 60 <= MAX_CHUNK_CHARS <= 500:
    raise ValueError("NONG_YOONG_QWEN_TTS_MAX_CHUNK_CHARS must be 60 to 500")
if not 0 <= CHUNK_PAUSE_MS <= 1000:
    raise ValueError("NONG_YOONG_QWEN_TTS_CHUNK_PAUSE_MS must be 0 to 1000")
if not 0 <= END_PADDING_MS <= 3000:
    raise ValueError("NONG_YOONG_QWEN_TTS_END_PADDING_MS must be 0 to 3000")

os.environ.setdefault("MPLCONFIGDIR", str(TEMP_DIR / "matplotlib"))


def write_pcm16_wav(waveform: Any, sample_rate: int, destination: Path) -> None:
    samples = np.asarray(waveform, dtype=np.float32).squeeze()
    if samples.ndim != 1 or samples.size == 0:
        raise RuntimeError("Qwen3-TTS generated empty or non-mono audio")
    if not np.isfinite(samples).all():
        raise RuntimeError("Qwen3-TTS generated non-finite audio")
    peak = float(np.max(np.abs(samples)))
    if peak <= 0:
        raise RuntimeError("Qwen3-TTS generated silent audio")
    if peak > 0.98:
        samples = samples * (0.98 / peak)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, samples, sample_rate, subtype="PCM_16")


def append_end_padding(
    waveform: Any, sample_rate: int, padding_ms: int = END_PADDING_MS
) -> np.ndarray:
    """Append silence so playback hardware cannot swallow the last syllable."""

    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    padding = np.zeros(silence_sample_count(sample_rate, padding_ms), dtype=np.float32)
    return np.concatenate((samples, padding))


class Engine:
    def __init__(self) -> None:
        self.model: Any = None
        self.voice_prompt: Any = None
        self.lock = Lock()

    def preload(self) -> None:
        if self.model is not None:
            return
        if not REFERENCE_AUDIO.is_file():
            raise RuntimeError(f"Qwen3-TTS reference WAV missing: {REFERENCE_AUDIO}")
        if not REFERENCE_TEXT:
            raise RuntimeError("NONG_YOONG_QWEN_TTS_REF_TEXT cannot be empty")

        import torch
        from qwen_tts import Qwen3TTSModel

        dtype = getattr(torch, DTYPE, None)
        if dtype is None:
            raise RuntimeError(f"unsupported PyTorch dtype: {DTYPE}")
        self.model = Qwen3TTSModel.from_pretrained(
            MODEL_ID,
            device_map=DEVICE,
            dtype=dtype,
            attn_implementation=ATTENTION,
        )
        self.voice_prompt = self.model.create_voice_clone_prompt(
            ref_audio=str(REFERENCE_AUDIO),
            ref_text=REFERENCE_TEXT,
            x_vector_only_mode=False,
        )

    def synthesize(self, text: str, destination: Path) -> None:
        chunks = split_text_chunks(text, max_chars=MAX_CHUNK_CHARS)
        if not chunks:
            raise ValueError("text contains no speakable characters")
        generated: list[np.ndarray] = []
        sample_rate: int | None = None
        with self.lock:
            for chunk in chunks:
                waveforms, chunk_sample_rate = self.model.generate_voice_clone(
                    text=chunk,
                    language=LANGUAGE,
                    voice_clone_prompt=self.voice_prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                if sample_rate is None:
                    sample_rate = int(chunk_sample_rate)
                elif sample_rate != int(chunk_sample_rate):
                    raise RuntimeError("Qwen3-TTS returned inconsistent sample rates")
                generated.append(np.asarray(waveforms[0], dtype=np.float32).squeeze())

        if sample_rate is None:
            raise RuntimeError("Qwen3-TTS generated no audio chunks")
        pause = np.zeros(int(sample_rate * CHUNK_PAUSE_MS / 1000), dtype=np.float32)
        pieces: list[np.ndarray] = []
        for index, waveform in enumerate(generated):
            if index:
                pieces.append(pause)
            pieces.append(waveform)
        combined = np.concatenate(pieces)
        if SPEAKING_SPEED != 1.0:
            import librosa

            combined = librosa.effects.time_stretch(combined, rate=SPEAKING_SPEED)
        combined = append_end_padding(combined, sample_rate)
        write_pcm16_wav(combined, sample_rate, destination)


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
                    f"Qwen3-TTS {MODEL_ID} loaded on {DEVICE}; "
                    f"speed={SPEAKING_SPEED}, chunk={MAX_CHUNK_CHARS} chars, "
                    f"tail={END_PADDING_MS} ms"
                ),
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/synthesize":
            self.send_json(404, {"detail": "not found"})
            return
        output_path: Path | None = None
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
            self.engine.synthesize(text, output_path)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(output_path.stat().st_size))
            self.end_headers()
            with output_path.open("rb") as audio:
                shutil.copyfileobj(audio, self.wfile, length=64 * 1024)
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"detail": str(exc)})
        except Exception as exc:
            self.send_json(503, {"detail": f"synthesis failed: {exc}"})
        finally:
            if output_path is not None:
                output_path.unlink(missing_ok=True)

    def log_message(self, message: str, *args: Any) -> None:
        print(f"Qwen3-TTS HTTP: {message % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preload", action="store_true")
    action.add_argument("--synthesize", metavar="TEXT")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "tts_samples" / "qwen3-thai.wav",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = Engine()
    engine.preload()
    print(
        f"Qwen3-TTS {MODEL_ID} loaded on {DEVICE} "
        f"(speed={SPEAKING_SPEED}, chunk={MAX_CHUNK_CHARS} chars, "
        f"tail={END_PADDING_MS} ms)"
    )
    if args.preload:
        return
    if args.synthesize:
        output = args.output.expanduser().resolve()
        engine.synthesize(args.synthesize, output)
        print(f"Created {output}")
        return
    Handler.engine = engine
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Qwen3-TTS listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
