"""Environment-backed configuration shared by server and Pi client."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class ServerSettings:
    host: str = "0.0.0.0"
    port: int = 5000
    public_base_url: str = ""
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma3:12b"
    llm_backend: str = "ollama"
    tts_backend: str = "espeak"
    mms_model: str = "facebook/mms-tts-tha"
    mms_speaking_rate: float = 0.95
    mms_noise_scale: float = 0.55
    mms_noise_scale_duration: float = 0.65
    mms_seed: int = 555
    espeak_voice: str = "th"
    espeak_speed: int = 155
    audio_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "audio")
    database_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "conversations.sqlite3"
    )
    max_history_messages: int = 20
    request_timeout_seconds: float = 180.0
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )

    @classmethod
    def from_env(cls) -> ServerSettings:
        origins = tuple(
            item.strip()
            for item in os.getenv(
                "NONG_YOONG_CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if item.strip()
        )
        return cls(
            host=os.getenv("NONG_YOONG_HOST", "0.0.0.0"),
            port=_env_int("NONG_YOONG_PORT", 5000),
            public_base_url=os.getenv("NONG_YOONG_PUBLIC_BASE_URL", "").rstrip("/"),
            ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            llm_backend=os.getenv("NONG_YOONG_LLM_BACKEND", "ollama").lower(),
            tts_backend=os.getenv("NONG_YOONG_TTS_BACKEND", "espeak").lower(),
            mms_model=os.getenv("NONG_YOONG_MMS_MODEL", "facebook/mms-tts-tha"),
            mms_speaking_rate=_env_float("NONG_YOONG_MMS_SPEAKING_RATE", 0.95),
            mms_noise_scale=_env_float("NONG_YOONG_MMS_NOISE_SCALE", 0.55),
            mms_noise_scale_duration=_env_float(
                "NONG_YOONG_MMS_NOISE_SCALE_DURATION", 0.65
            ),
            mms_seed=_env_int("NONG_YOONG_MMS_SEED", 555),
            espeak_voice=os.getenv("NONG_YOONG_ESPEAK_VOICE", "th"),
            espeak_speed=_env_int("NONG_YOONG_ESPEAK_SPEED", 155),
            audio_dir=_env_path(
                "NONG_YOONG_AUDIO_DIR", PROJECT_ROOT / "data" / "audio"
            ),
            database_path=_env_path(
                "NONG_YOONG_DATABASE_PATH",
                PROJECT_ROOT / "data" / "conversations.sqlite3",
            ),
            max_history_messages=_env_int("NONG_YOONG_MAX_HISTORY", 20),
            request_timeout_seconds=_env_float("NONG_YOONG_LLM_TIMEOUT", 180.0),
            cors_origins=origins,
        )


@dataclass(frozen=True, slots=True)
class ClientSettings:
    server_url: str = "http://127.0.0.1:5000"
    activation_mode: str = "keyboard"
    stt_model_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT
        / "models"
        / "sherpa-onnx-zipformer-thai-2024-06-20"
    )
    wake_model_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "models" / "keyword_model.tflite"
    )
    audio_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "pi_audio")
    events_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "pi_events.jsonl"
    )
    sample_rate: int = 16_000
    vad_aggressiveness: int = 2
    silence_seconds: float = 1.2
    speech_start_timeout_seconds: float = 5.0
    max_utterance_seconds: float = 30.0
    request_timeout_seconds: float = 240.0
    stream_chunk_bytes: int = 64 * 1024
    retain_microphone_audio: bool = False
    playback_enabled: bool = True
    wake_threshold: float = 0.90
    wake_invert_probability: bool = True
    input_device: str | None = None
    output_device: str | None = None
    audio_player: str = "auto"

    @classmethod
    def from_env(cls) -> ClientSettings:
        return cls(
            server_url=os.getenv("NONG_YOONG_SERVER_URL", "http://127.0.0.1:5000").rstrip("/"),
            activation_mode=os.getenv("NONG_YOONG_ACTIVATION", "keyboard").lower(),
            stt_model_dir=_env_path(
                "NONG_YOONG_STT_MODEL_DIR",
                PROJECT_ROOT / "models" / "sherpa-onnx-zipformer-thai-2024-06-20",
            ),
            wake_model_path=_env_path(
                "NONG_YOONG_WAKE_MODEL", PROJECT_ROOT / "models" / "keyword_model.tflite"
            ),
            audio_dir=_env_path(
                "NONG_YOONG_PI_AUDIO_DIR", PROJECT_ROOT / "data" / "pi_audio"
            ),
            events_path=_env_path(
                "NONG_YOONG_PI_EVENTS", PROJECT_ROOT / "data" / "pi_events.jsonl"
            ),
            sample_rate=_env_int("NONG_YOONG_SAMPLE_RATE", 16_000),
            vad_aggressiveness=_env_int("NONG_YOONG_VAD_AGGRESSIVENESS", 2),
            silence_seconds=_env_float("NONG_YOONG_SILENCE_SECONDS", 1.2),
            speech_start_timeout_seconds=_env_float(
                "NONG_YOONG_SPEECH_START_TIMEOUT", 5.0
            ),
            max_utterance_seconds=_env_float("NONG_YOONG_MAX_UTTERANCE", 30.0),
            request_timeout_seconds=_env_float("NONG_YOONG_REQUEST_TIMEOUT", 240.0),
            stream_chunk_bytes=_env_int("NONG_YOONG_STREAM_CHUNK", 64 * 1024),
            retain_microphone_audio=_env_bool("NONG_YOONG_RETAIN_MIC_AUDIO", False),
            playback_enabled=_env_bool("NONG_YOONG_PLAYBACK", True),
            wake_threshold=_env_float("NONG_YOONG_WAKE_THRESHOLD", 0.90),
            wake_invert_probability=_env_bool("NONG_YOONG_WAKE_INVERT", True),
            input_device=os.getenv("NONG_YOONG_INPUT_DEVICE") or None,
            output_device=os.getenv("NONG_YOONG_OUTPUT_DEVICE") or None,
            audio_player=os.getenv("NONG_YOONG_AUDIO_PLAYER", "auto").lower(),
        )
