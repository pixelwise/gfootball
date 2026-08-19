#!/bin/bash

set -euo pipefail

IMAGE_TAG="gfootball_docker_test"

docker build --no-cache -t "${IMAGE_TAG}" .
docker run --rm "${IMAGE_TAG}" bash -lc \
  'set -e; for test_file in gfootball/env/*test.py; do \
     UNITTEST_IN_DOCKER=1 uv run --locked python "$test_file"; \
   done'

# Rendering is tested with a virtual X server, so no host GPU or X11 access is
# required.
docker run --rm "${IMAGE_TAG}" xvfb-run -a \
  uv run --locked python -m unittest \
  gfootball.env.football_env_test.FootballEnvTest.test___render

echo "Docker tests successful."
