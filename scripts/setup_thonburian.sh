#!/usr/bin/env bash
set -euo pipefail

readonly SOURCE_DIR="models/thonburian-tts"
readonly REVISION="032fe7e51674afe066a98e6d3cf47fc96d04b290"

python3.11 -m venv .venv-thonburian
.venv-thonburian/bin/python -m pip install --upgrade pip
.venv-thonburian/bin/python -m pip install \
  'torch==2.11.0+cpu' 'torchaudio==2.11.0+cpu' 'torchcodec==0.15.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
.venv-thonburian/bin/python -m pip install -r requirements-thonburian.txt

mkdir -p models
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone https://github.com/biodatlab/thonburian-tts.git "${SOURCE_DIR}"
  git -C "${SOURCE_DIR}" checkout "${REVISION}"
fi

actual_revision="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${REVISION}" ]]; then
  echo "Unexpected Thonburian source revision: ${actual_revision}" >&2
  echo "Expected ${REVISION}; move ${SOURCE_DIR} aside and rerun setup." >&2
  exit 1
fi

.venv-thonburian/bin/python scripts/thonburian_sidecar.py --preload
echo "ThonburianTTS ready. Start it with:"
echo ".venv-thonburian/bin/python scripts/thonburian_sidecar.py"
