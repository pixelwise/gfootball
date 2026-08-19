#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DUMPS_DIR="${PROJECT_DIR}/dumps"
IMAGE_TAG="${GFOOTBALL_DOCKER_IMAGE:-gfootball}"

mkdir -p "${DUMPS_DIR}"

exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -v "${DUMPS_DIR}:/tmp/dumps" \
  "${IMAGE_TAG}" \
  env -u DISPLAY uv run --locked python -m gfootball.play_game \
  --config_file gfootball/configs/default.yaml \
  --render=true
