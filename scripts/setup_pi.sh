#!/usr/bin/env bash
set -euo pipefail

python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[pi,wake]'
mkdir -p data/pi_audio models
echo "Pi Python environment ready. Run scripts/download_models.py next."
