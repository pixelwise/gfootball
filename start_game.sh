#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

PROJECT_PYTHON="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "${PROJECT_PYTHON}" ]]; then
  echo "Python environment not found at ${PROJECT_PYTHON}." >&2
  echo "Run ./build-uv.sh before starting the game." >&2
  exit 1
fi

exec "${PROJECT_PYTHON}" -m gfootball.play_game \
  --config_file gfootball/configs/default.yaml \
  --render=true
