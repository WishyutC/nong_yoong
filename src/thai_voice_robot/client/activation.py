"""Keyboard and continuous overlapping-window TFLite activation detectors."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Protocol


class Activator(Protocol):
    def wait_for_activation(self) -> None: ...


class KeyboardActivator:
    def wait_for_activation(self) -> None:
        input("Press Enter, then speak Thai... ")


class TFLiteWakeWordActivator:
    def __init__(
        self,
        model_path: Path,
        *,
        sample_rate: int = 16_000,
        threshold: float = 0.90,
        invert_probability: bool = True,
        input_device: str | int | None = None,
    ) -> None:
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.invert_probability = invert_probability
        self.input_device = input_device
        self._interpreter = None

    def _load(self):
        if self._interpreter is not None:
            return self._interpreter
        if not self.model_path.is_file():
            raise FileNotFoundError(f"wake-word model not found: {self.model_path}")
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError as exc:
                raise RuntimeError(
                    "install tflite-runtime or ai-edge-litert for wake-word mode"
                ) from exc
        self._interpreter = Interpreter(model_path=str(self.model_path))
        self._interpreter.allocate_tensors()
        return self._interpreter

    def wait_for_activation(self) -> None:
        try:
            import librosa
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError("wake-word audio dependencies are missing") from exc

        interpreter = self._load()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        hop_samples = self.sample_rate // 4
        window_samples = self.sample_rate
        chunks: deque = deque()
        buffered = 0

        print("Listening for the wake word...")
        with sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=hop_samples,
            device=self.input_device,
            channels=1,
            dtype="float32",
        ) as stream:
            while True:
                chunk, _overflowed = stream.read(hop_samples)
                flat = np.asarray(chunk, dtype=np.float32).reshape(-1)
                chunks.append(flat)
                buffered += len(flat)
                while buffered > window_samples and chunks:
                    removed = chunks.popleft()
                    buffered -= len(removed)
                if buffered < window_samples:
                    continue

                audio = np.concatenate(tuple(chunks))[-window_samples:]
                mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
                if mfcc.shape[1] < 32:
                    mfcc = np.pad(mfcc, ((0, 0), (0, 32 - mfcc.shape[1])))
                tensor = mfcc[:, :32][np.newaxis, ..., np.newaxis].astype(np.float32)
                interpreter.set_tensor(input_detail["index"], tensor)
                interpreter.invoke()
                probability = float(
                    np.asarray(interpreter.get_tensor(output_detail["index"])).reshape(-1)[0]
                )
                if self.invert_probability:
                    probability = 1.0 - probability
                if probability >= self.threshold:
                    print(f"Wake word detected ({probability:.3f})")
                    return
