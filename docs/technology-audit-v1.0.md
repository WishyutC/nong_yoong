# Nong Yoong 1.0.0 technology audit

This document records what the 1.0.0 software release actively uses, what is
kept as a fallback or development tool, and what can be removed later. It does
not authorize deletion of user recordings, conversation history, or generated
audio.

## Active workstation path

- WSL Ubuntu 24.04 and Python 3.11
- FastAPI, Uvicorn, Pydantic, and Requests for the workstation HTTP API
- Ollama with `gemma3:12b` for local language-model inference
- Qwen3-TTS `Qwen/Qwen3-TTS-12Hz-1.7B-Base` for Thai voice cloning
- A separate `.venv-qwen-tts` with PyTorch CUDA 12.8, BF16, SDPA attention,
  NumPy, SoundFile, and Librosa
- A standard-library threaded HTTP sidecar on loopback port 5055
- Sentence-bounded TTS generation, pitch-preserving speed adjustment, pause
  insertion, and final silence padding
- Mono 24 kHz signed PCM16 WAV output streamed in bounded HTTP chunks
- SQLite structured conversation, response, latency, request, and session data
- UUID request/session correlation and serialized response generation
- systemd service templates for Qwen3-TTS and the workstation API

The configured voice reference for this installation is
`data/audio/girl_voice_2.wav`. Reference recordings are user-owned assets and
must not be removed or overwritten during cleanup.

## Active Raspberry Pi/client path

- Python 3.11
- SoundDevice/PortAudio for microphone capture
- WebRTC VAD for speech start/end detection
- sherpa-onnx Thai INT8 Zipformer for offline Thai STT
- TFLite Runtime, Librosa MFCCs, and the local TFLite wake-word model
- Requests for health, response, and bounded WAV download calls
- `aplay` on Raspberry Pi or FFplay under WSL for blocking playback
- A shared state machine that mutes activation during recording, inference, and
  playback
- JSONL client event logs with raw microphone deletion by default
- systemd client service template

The software adapters and model files are present and covered by integration
tests. Physical Pi microphone, speaker, noise, heat, and wake-word acceptance
remain hardware-validation work described in `docs/hardware-validation.md`.

## Development and release tooling

- Standard-library `unittest` suite
- GitHub Actions on Ubuntu 24.04 and Python 3.11
- Compile checks, dependency checks, model downloader, doctor utility, direct
  TTS synthesizer, and local Git wrapper
- Local Git metadata under `.gitdata` only because the Codex workspace exposes
  `.git` as a read-only empty mount; normal clones use ordinary `.git`

## Present but inactive fallbacks

These are not used while `.env` selects `ollama` and `qwen`, but they are
intentional recovery or test paths:

- Meta MMS Thai TTS, its CPU PyTorch/Transformers implementation, and its
  cached model (non-commercial model license)
- eSpeak NG Thai emergency TTS
- Deterministic Echo LLM used by automated and offline tests
- Keyboard activation and text-only client modes used for bring-up
- Legacy `POST /generate` compatibility endpoint
- The first `girl_voice.wav` profile, which is stored but not selected

Removing these is a product decision, not routine cleanup.

## Installed leftovers not used by project code

The main `.venv` is about 2.4 GB and contains packages not declared by the
current project and not imported by its source. They appear to be remnants of
earlier model experiments. Confirmed cleanup candidates include:

- Training/quantization: `bitsandbytes`, `datasets`, `wandb`, `accelerate`
- Old flow/audio model stack: `cached_path`, `ema-pytorch`, `encodec`,
  `torchdiffeq`, `vocos`, `x-transformers`, `transformers-stream-generator`
- Old text stack: `pypinyin`, `pythainlp`, `rjieba`, `Unidecode`
- Unused application helpers: `pydub`, `matplotlib`, `hydra-core`,
  `python-multipart`, `orjson`, `torchcodec`
- Dependency subtrees pulled by those leftovers, including Boto3/Google Cloud
  packages and Pandas/PyArrow dataset packages

The safest cleanup is to recreate `.venv` from `pyproject.toml` instead of
uninstalling this graph package-by-package. Do not remove the Pi extras from an
actual Pi environment.

The isolated Qwen environment also contains Gradio and related web packages
that the headless sidecar does not directly import. They are mandatory
dependencies declared by `qwen-tts==0.1.1`, so they should remain unless the
sidecar moves to a custom minimal Qwen runtime.

## Unused model caches

- `Systran/faster-whisper-small` (about 464 MB): no active Whisper code
- `charactr/vocos-mel-24khz` (about 52 MB): leftover from an earlier TTS test
- `facebook/mms-tts-tha` (about 139 MB): inactive but retained MMS fallback

The 4.3 GB Qwen3-TTS cache is active and must remain. The 154 MB sherpa-onnx
Thai model and wake-word TFLite model are active client assets.

## Not implemented in 1.0.0

These ideas belong to the roadmap and are not part of the current runtime:

- PostgreSQL migration
- Web dashboard frontend
- IoT action execution and validation
- Mood/emotion inference, medical or wellness diagnosis
- Speaker/user voice identification
- Acoustic echo cancellation, beamforming, and barge-in
- Progressive TTS playback, SSE, or WebSocket audio streaming
- Automatic multi-reference voice-profile selection

## Removed legacy technology

Vosk English STT, Whisper as the primary STT, Base64 WAV delivery, Pi-side
Pydub/FFmpeg conversion, and earlier experimental Thai TTS engines are not part
of the 1.0.0 tree.
