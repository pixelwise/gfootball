#!/usr/bin/env bash

set -euo pipefail

if ! command -v uv &> /dev/null; then
    echo "❌ 'uv' not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# The installer updates future shells; make its default location available to
# this invocation as well.
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv &> /dev/null; then
  echo "❌ uv was installed but is not available on PATH." >&2
  exit 1
fi

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

cleanup_placeholder_extensions() {
  find "${PROJECT_DIR}" -maxdepth 1 -type f \
    -name 'brainball_cpp_engine*.so' -delete
}
trap cleanup_placeholder_extensions EXIT

for required_command in cmake make g++; do
  if ! command -v "${required_command}" &> /dev/null; then
    echo "❌ Required native build tool '${required_command}' was not found." >&2
    echo "Install the native prerequisites documented in gfootball/doc/compile_engine.md." >&2
    exit 1
  fi
done

# The current engine is linked against Boost.Python 3.9 from this environment.
# Override these paths when an equivalent Python 3.9/native toolchain is used.
GFOOTBALL_PYTHON="${GFOOTBALL_PYTHON:-/home/artem/miniconda3/envs/gfootball/bin/python}"
if [[ ! -x "${GFOOTBALL_PYTHON}" ]]; then
  echo "❌ Python interpreter not found: ${GFOOTBALL_PYTHON}" >&2
  echo "Set GFOOTBALL_PYTHON to a Python 3.9 interpreter." >&2
  exit 1
fi

PYTHON_MINOR="$("${GFOOTBALL_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${PYTHON_MINOR}" != "3.9" ]]; then
  echo "❌ Google Research Football requires Python 3.9 for the current native toolchain; found ${PYTHON_MINOR}." >&2
  exit 1
fi

GFOOTBALL_NATIVE_PREFIX="${GFOOTBALL_NATIVE_PREFIX:-$(cd -- "$(dirname -- "${GFOOTBALL_PYTHON}")/.." && pwd)}"
export CMAKE_PREFIX_PATH="${GFOOTBALL_NATIVE_PREFIX}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export LD_LIBRARY_PATH="${GFOOTBALL_NATIVE_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "▶ Synchronizing Python dependencies with uv..."
uv sync --locked --no-install-project --python "${GFOOTBALL_PYTHON}"

echo "▶ Building and installing gfootball..."
export GFOOTBALL_BUILD_PYTHON="${PROJECT_DIR}/.venv/bin/python"
# Native build outputs live in the source tree and can be removed independently
# of the editable package metadata (for example, by `git clean -x`). Force uv to
# rerun the project's build hook so a missing engine is always rebuilt.
uv sync --locked --python "${GFOOTBALL_PYTHON}" --reinstall-package gfootball

echo "▶ Verifying Python and native engine imports..."
uv run --locked python -c 'import gfootball; import gfootball_engine; print("✅ gfootball environment is ready")'

echo "Run the game with:"
echo "  uv run python -m gfootball.play_game --config_file gfootball/configs/default.yaml --render=true"
