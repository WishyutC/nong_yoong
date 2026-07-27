#!/usr/bin/env bash
set -euo pipefail

python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
mkdir -p data/audio
echo "Workstation ready. Copy .env.workstation.example to .env and export its values."
