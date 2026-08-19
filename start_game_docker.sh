#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DUMPS_DIR="${PROJECT_DIR}/dumps"
IMAGE_TAG="${GFOOTBALL_DOCKER_IMAGE:-gfootball}"

if [[ -n "${GFOOTBALL_CONTAINER_RUNTIME:-}" ]]; then
  CONTAINER_RUNTIME="${GFOOTBALL_CONTAINER_RUNTIME}"
elif command -v docker >/dev/null 2>&1; then
  CONTAINER_RUNTIME=docker
elif command -v podman >/dev/null 2>&1; then
  CONTAINER_RUNTIME=podman
else
  echo "ERROR: Docker or Podman is required. Install one, or set GFOOTBALL_CONTAINER_RUNTIME." >&2
  exit 1
fi

if ! command -v "${CONTAINER_RUNTIME}" >/dev/null 2>&1; then
  echo "ERROR: Container runtime not found: ${CONTAINER_RUNTIME}" >&2
  exit 1
fi

mkdir -p "${DUMPS_DIR}"

exec "${CONTAINER_RUNTIME}" run --rm \
  --user "$(id -u):$(id -g)" \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -v "${DUMPS_DIR}:/tmp/dumps" \
  "${IMAGE_TAG}" \
  env -u DISPLAY uv run --locked python -m gfootball.play_game \
  --config_file gfootball/configs/default.yaml \
  --render=true
