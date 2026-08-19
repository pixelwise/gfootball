#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DUMPS_DIR="${GFOOTBALL_DUMPS_DIR:-${PROJECT_DIR}/dumps}"
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

if ! mkdir -p "${DUMPS_DIR}" || [[ ! -w "${DUMPS_DIR}" ]]; then
  echo "ERROR: Dump directory is not writable: ${DUMPS_DIR}" >&2
  echo "Choose a directory owned by your user, for example:" >&2
  echo "  GFOOTBALL_DUMPS_DIR=\"\$HOME/gfootball-dumps\" $0" >&2
  exit 1
fi

VOLUME_SPEC="${DUMPS_DIR}:/tmp/dumps"
# :Z relabels the bind mount for rootless Podman on SELinux hosts. It is
# harmless on systems where SELinux is disabled.
if [[ "${CONTAINER_RUNTIME}" == podman ]]; then
  VOLUME_SPEC="${VOLUME_SPEC}:Z"
  # Rootless Podman maps container root to the invoking host user. Passing
  # --user here instead selects a subordinate ID that cannot write the mount.
  RUN_USER_ARGS=()
else
  RUN_USER_ARGS=(--user "$(id -u):$(id -g)")
fi

exec "${CONTAINER_RUNTIME}" run --rm \
  "${RUN_USER_ARGS[@]}" \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -v "${VOLUME_SPEC}" \
  "${IMAGE_TAG}" \
  env -u DISPLAY uv run --locked python -m gfootball.play_game \
  --config_file gfootball/configs/default.yaml \
  --render=true
