#!/usr/bin/env bash
set -euo pipefail

python3.11 -m venv .venv-qwen-tts
.venv-qwen-tts/bin/python -m pip install --upgrade pip
.venv-qwen-tts/bin/python -m pip install 'torch>=2.7,<3' 'torchaudio>=2.7,<3' \
  --index-url https://download.pytorch.org/whl/cu128
.venv-qwen-tts/bin/python -m pip install -r requirements-qwen-tts.txt
echo "Qwen3-TTS ready. Start it with:"
echo ".venv-qwen-tts/bin/python scripts/qwen_tts_sidecar.py"
echo "The first start downloads the 1.7B voice-cloning model."
