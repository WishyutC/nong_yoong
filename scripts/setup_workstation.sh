#!/usr/bin/env bash
set -euo pipefail

python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Install the CPU wheel first so pip does not pull multi-gigabyte CUDA packages.
.venv/bin/python -m pip install 'torch>=2.1,<3' \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[mms]'
mkdir -p data/audio
echo "Workstation ready with neural Thai MMS TTS. Copy .env.workstation.example to .env."
