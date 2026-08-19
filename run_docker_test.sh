#!/bin/bash

set -euo pipefail

IMAGE_TAG="gfootball_docker_test"

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

"${CONTAINER_RUNTIME}" build --no-cache -t "${IMAGE_TAG}" .
"${CONTAINER_RUNTIME}" run --rm "${IMAGE_TAG}" bash -lc \
  'set -e; for test_file in gfootball/env/*test.py; do \
     UNITTEST_IN_DOCKER=1 uv run --locked python "$test_file"; \
   done'

# Rendering is tested with a virtual X server, so no host GPU or X11 access is
# required.
"${CONTAINER_RUNTIME}" run --rm "${IMAGE_TAG}" xvfb-run -a \
  uv run --locked python -m unittest \
  gfootball.env.football_env_test.FootballEnvTest.test___render

echo "Container tests successful (${CONTAINER_RUNTIME})."
