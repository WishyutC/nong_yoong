"""Microphone recording with WebRTC VAD endpoint detection."""

from __future__ import annotations

import wave
from collections import deque
from pathlib import Path
from time import monotonic


class NoSpeechDetected(RuntimeError):
    pass


class VoiceActivityRecorder:
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        aggressiveness: int = 2,
        silence_seconds: float = 1.2,
        speech_start_timeout_seconds: float = 5.0,
        max_utterance_seconds: float = 30.0,
        input_device: str | int | None = None,
    ) -> None:
        if sample_rate not in {8_000, 16_000, 32_000, 48_000}:
            raise ValueError("WebRTC VAD requires 8, 16, 32, or 48 kHz audio")
        if aggressiveness not in {0, 1, 2, 3}:
            raise ValueError("VAD aggressiveness must be from 0 to 3")
        self.sample_rate = sample_rate
        self.aggressiveness = aggressiveness
        self.silence_seconds = silence_seconds
        self.speech_start_timeout_seconds = speech_start_timeout_seconds
        self.max_utterance_seconds = max_utterance_seconds
        self.input_device = input_device

    def record(self, destination: Path) -> Path:
        try:
            import sounddevice as sd
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError(
                "Pi audio dependencies are missing; install the project with the pi extra"
            ) from exc

        frame_ms = 30
        frame_samples = self.sample_rate * frame_ms // 1000
        silence_frames_needed = max(1, round(self.silence_seconds * 1000 / frame_ms))
        pre_roll = deque(maxlen=max(1, 300 // frame_ms))
        frames: list[bytes] = []
        vad = webrtcvad.Vad(self.aggressiveness)
        started_at = monotonic()
        speech_started_at: float | None = None
        silent_frames = 0

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=frame_samples,
            device=self.input_device,
            channels=1,
            dtype="int16",
        ) as stream:
            while True:
                data, _overflowed = stream.read(frame_samples)
                raw = bytes(data)
                is_speech = vad.is_speech(raw, self.sample_rate)
                elapsed = monotonic() - started_at

                if speech_started_at is None:
                    pre_roll.append(raw)
                    if is_speech:
                        speech_started_at = monotonic()
                        frames.extend(pre_roll)
                        pre_roll.clear()
                    elif elapsed >= self.speech_start_timeout_seconds:
                        raise NoSpeechDetected("no speech detected after activation")
                    continue

                frames.append(raw)
                silent_frames = 0 if is_speech else silent_frames + 1
                utterance_seconds = monotonic() - speech_started_at
                if silent_frames >= silence_frames_needed and utterance_seconds >= 0.3:
                    break
                if utterance_seconds >= self.max_utterance_seconds:
                    break

        destination.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destination), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b"".join(frames))
        return destination
