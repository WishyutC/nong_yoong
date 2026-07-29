# Nong Yoong Thai Voice Robot

Nong Yoong is a Python 3.11 edge voice robot split between a Raspberry Pi and a
workstation.

```text
Pi microphone → activation → WebRTC VAD → Thai Zipformer STT
              → workstation Ollama → Thai workstation TTS
              → streamed PCM16 WAV → blocking Pi playback
```

The shared runtime state is:

```text
STARTING → IDLE → RECORDING → THINKING → SPEAKING → IDLE
```

Wake-word recognition is inactive during `RECORDING`, `THINKING`, and
`SPEAKING`, preventing the robot from recognizing its own reply. Text mode can
move directly from `IDLE` to `THINKING` for setup and API testing.

## What is implemented

- Versioned FastAPI server with liveness and dependency readiness checks
- Ollama `/api/chat` integration with `gemma3:12b` as the default model
- Neural `facebook/mms-tts-tha` Thai TTS on the workstation
- Lightweight Thai eSpeak NG emergency fallback
- Final mono PCM16 WAV generation and bounded-chunk HTTP delivery
- UUID request idempotency and session-scoped LLM conversation history
- Structured SQLite events, real dashboard counts, and audio references
- Pi API client with retries, correlation validation, bounded downloads, and
  rejection of unexpected cross-host audio URLs
- WebRTC VAD recording and endpointing
- Thai INT8 sherpa-onnx Zipformer STT adapter
- Keyboard activation and continuous overlapping-window TFLite wake detection
- Blocking `aplay`/`ffplay` playback and microphone muting through state
- Text-only end-to-end mode for testing without Pi audio hardware
- systemd templates, environment examples, model downloader, and doctor tool

Raw microphone audio is deleted after recognition by default. The workstation
stores generated reply audio and structured text events; it does not store the
user's raw voice.

## Workstation quick start

Ubuntu packages:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv espeak-ng alsa-utils
```

Project setup:

```bash
cd ~/nong_yoong
./scripts/setup_workstation.sh
cp .env.workstation.example .env
```

Edit `.env`, especially `NONG_YOONG_PUBLIC_BASE_URL`, to use the workstation's
LAN IP. Ensure Ollama is running and the configured model exists:

```bash
ollama list
ollama pull gemma3:12b
```

Start the API:

```bash
source .venv/bin/activate
nong-yoong-server
```

Smoke-test it from another terminal without a microphone or speaker:

```bash
source .venv/bin/activate
nong-yoong-client --once "สวัสดี น้องยูง" --no-playback
```

For a deterministic test when Ollama is unavailable, temporarily set
`NONG_YOONG_LLM_BACKEND=echo` in `.env`.

### Thai TTS

The workstation setup installs the CPU-only neural runtime and selects MMS in
`.env.workstation.example`. Its `facebook/mms-tts-tha` weights download on the
first synthesis and are then cached for offline use. Test the voice without
starting the server:

```bash
.venv/bin/python scripts/synthesize_tts.py
aplay data/tts_samples/mms-thai.wav
```

This model is CC-BY-NC-4.0 and is not appropriate for commercial use without
permission. Set `NONG_YOONG_TTS_BACKEND=espeak` for a lightweight, fully local
emergency fallback; that voice is intentionally more robotic.

## Raspberry Pi setup

On a 64-bit Raspberry Pi OS installation with Python 3.11:

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv libportaudio2 portaudio19-dev \
  libopenblas0 alsa-utils

cd ~/nong_yoong
./scripts/setup_pi.sh
cp .env.pi.example .env
.venv/bin/python scripts/download_models.py
```

The downloader uses the official sherpa-onnx Thai model archive but extracts
only the INT8 encoder/joiner, decoder, and token table needed at runtime.

Start with keyboard activation so audio devices and Thai STT can be tested
before wake-word tuning:

```bash
source .venv/bin/activate
nong-yoong-client --mode keyboard
```

The Pi setup installs `tflite-runtime` for wake-word mode. Place the trained
model at `models/keyword_model.tflite`. The existing model can be
copied during setup with:

```bash
.venv/bin/python scripts/download_models.py \
  --copy-wake-model /path/to/keyword_model.tflite
nong-yoong-client --mode wake
```

Use `NONG_YOONG_INPUT_DEVICE` and `NONG_YOONG_OUTPUT_DEVICE` in `.env` when the
USB audio adapter is not the system default. Run `.venv/bin/python
scripts/doctor.py` to check dependencies and model files.

## API and data

- `POST /v1/responses`: correlated LLM and TTS turn
- `GET /v1/audio/{request_id}.wav`: streamed playback audio
- `GET /health/live`: process health
- `GET /health/ready`: Ollama/model and TTS readiness
- `GET /v1/sessions/{session_id}/events`: conversation history
- `GET /v1/dashboard/summary`: real turn/session totals
- `POST /generate`: temporary non-Base64 legacy adapter

The complete payload contract is in [docs/api-contract.md](docs/api-contract.md).

## Testing

```bash
source .venv/bin/activate
python -m unittest discover -s tests -t . -v
python -m compileall -q src tests scripts
```

GitHub Actions repeats the Python 3.11 core and workstation tests on every push
and pull request. Hardware/model integration tests automatically run when their
ignored model assets and optional Pi dependencies are present; otherwise those
device-specific cases are reported as skipped.

Hardware acceptance steps and metrics are in
[docs/hardware-validation.md](docs/hardware-validation.md).

## Version control in this workspace

The Codex environment exposes the normal `.git` path as a read-only empty
mount. This checkout therefore keeps equivalent Git metadata in the ignored
`.gitdata` directory. Use `./scripts/git-local.sh status`, `add`, `commit`,
`fetch`, and `push` here. A normal clone from GitHub uses ordinary `git`
commands and does not need this wrapper.
