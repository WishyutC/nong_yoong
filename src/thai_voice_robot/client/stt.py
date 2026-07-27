"""Thai Zipformer speech recognition using sherpa-onnx."""

from __future__ import annotations

import sys
import wave
from array import array
from pathlib import Path
from threading import Lock


class ThaiZipformerRecognizer:
    ENCODER = "encoder-epoch-12-avg-5.int8.onnx"
    DECODER = "decoder-epoch-12-avg-5.onnx"
    JOINER = "joiner-epoch-12-avg-5.int8.onnx"
    TOKENS = "tokens.txt"

    def __init__(self, model_dir: Path, *, num_threads: int = 2) -> None:
        self.model_dir = model_dir
        self.num_threads = num_threads
        self._recognizer = None
        self._lock = Lock()

    def _model_path(self, filename: str) -> Path:
        path = self.model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"missing Thai STT model file {path}; run scripts/download_models.py"
            )
        return path

    def _load(self):
        with self._lock:
            if self._recognizer is not None:
                return self._recognizer
            try:
                import sherpa_onnx
            except ImportError as exc:
                raise RuntimeError("sherpa-onnx is not installed") from exc

            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(self._model_path(self.ENCODER)),
                decoder=str(self._model_path(self.DECODER)),
                joiner=str(self._model_path(self.JOINER)),
                tokens=str(self._model_path(self.TOKENS)),
                num_threads=self.num_threads,
                sample_rate=16_000,
                feature_dim=80,
                decoding_method="greedy_search",
                max_active_paths=4,
                provider="cpu",
            )
            return self._recognizer

    def transcribe(self, wav_path: Path) -> str:
        with wave.open(str(wav_path), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise ValueError("Thai STT input must be mono PCM16 WAV")
            sample_rate = wav_file.getframerate()
            pcm = array("h")
            pcm.frombytes(wav_file.readframes(wav_file.getnframes()))
        if sys.byteorder != "little":
            pcm.byteswap()
        samples = [sample / 32768.0 for sample in pcm]

        recognizer = self._load()
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        recognizer.decode_stream(stream)
        result = getattr(stream, "result", None)
        if result is None and hasattr(recognizer, "get_result"):
            result = recognizer.get_result(stream)
        return str(getattr(result, "text", "")).strip()
