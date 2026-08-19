# Project overview

Google Research Football is a Python reinforcement-learning environment built around a bundled, modified Gameplay Football C++ engine. The Python package exposes Gym-compatible environments, scenarios, wrappers, player implementations, replay utilities, and example training programs; installing the package compiles and packages the native engine.

# Setup

Run commands from the repository root. Docker is the supported development and
runtime environment: it provides Python 3.9, uv, C++14/CMake, SDL2, and
Boost.Python without requiring host-native dependencies.

# Build / run

Build or refresh the development image:

```sh
docker build -t gfootball .
```

The image build runs `uv sync --locked`, which compiles the native engine and
links it as `_gameplayfootball.so`.

Run the checked-in default configuration with Docker off-screen rendering:

```sh
./start_game_docker.sh
```

This starts the `gfootball` Docker image with the default configuration and
writes outputs under `./dumps`. Build the image first with `docker build -t
gfootball .`.

The entry point uses off-screen rendering. For visible X11 rendering or an
interactive shell, see `gfootball/doc/docker.md`.

# Test

Tests are standalone `absltest`/`unittest` scripts under `gfootball/env`; there
is no pytest/tox configuration. Run the Docker validation suite, which builds a
clean image, runs every environment test headlessly, and runs an Xvfb rendering
test:

```sh
./run_docker_test.sh
```

# Code style

No formatter, linter, pre-commit hook, or style CI is configured. Preserve the style of the file being edited instead of applying repository-wide formatting. Python generally uses `snake_case` functions/variables, `PascalCase` classes, module docstrings, and two-space indentation in the older environment code. Scenario files expose `build_scenario(builder)` and intentionally use the engine-facing `SetTeam`, `AddPlayer`, and related APIs. C++ under `third_party/gfootball_engine` is legacy code with mixed conventions; keep changes focused and match adjacent code. Name Python tests `*_test.py` and test methods `test_*`.

# Repo structure

- `gfootball/env/`: Gym environment, core engine bridge, actions, observations, wrappers, players, and unit/E2E tests.
- `gfootball/scenarios/`: built-in scenario definitions; `gfootball/scenarios/tests/` contains deterministic scenarios consumed by environment tests, not a separate test runner.
- `gfootball/configs/default.yaml`: configuration used by `start_game_docker.sh`.
- `gfootball/examples/`: PPO2/RLlib examples and reproduction scripts; these need optional legacy training dependencies not installed by the base package.
- `gfootball/doc/`: API, build, Docker, observation/action, multi-agent, scenario, replay, and imitation-learning documentation.
- `gfootball/eval_server/`: remote evaluation client/server protocol code; `*_pb2.py` and `*_pb2_grpc.py` are generated files.
- `third_party/gfootball_engine/`: bundled C++ engine, CMake build, assets, and Windows vcpkg manifests.
- `pyproject.toml` and `uv.lock`: canonical package metadata, dependency declarations, Python compatibility, and reproducible resolution.
- `setup.py`: custom setuptools bridge for compiling and packaging the native engine.
- `requirements.txt`: compatibility export; do not treat it as the canonical dependency source.
- `Dockerfile`, `Dockerfile_examples`, `run_docker_test.sh`: base environment and legacy TensorFlow 1.15 training images/tests.
- `.github/workflows/`: macOS build checks and Windows wheel publishing/testing.

# Gotchas

- Change dependencies in `pyproject.toml`, refresh `uv.lock`, and keep the compatibility `requirements.txt` aligned when its direct requirements change.
- `gym` is pinned to `0.17.3`; Gym 0.21.0 has legacy metadata that fails with current packaging tools. Training examples additionally expect TensorFlow 1.15, Sonnet 1.x, and OpenAI Baselines. Do not silently modernize these APIs as part of unrelated work.
- Editable install creates a root `gfootball_engine` link (or a copy on Windows). Switching install modes can remove that link and generated native artifacts; rebuild after native-engine changes.
- `GFOOTBALL_USE_PREBUILT_SO=1` makes Unix installation copy the checked-in prebuilt engine rather than compile it. Leave it unset when validating C++ changes.
- Windows builds require `VCPKG_ROOT`; setup derives `PY_VERSION` and the CMake platform from the active interpreter.
- Rendering is single-instance within a process. Headless tests should set `UNITTEST_IN_DOCKER=1`; Docker rendering additionally needs X11 access.
- `start_game_docker.sh` enables full episode dumps and video through `default.yaml`. Each run creates a timestamped directory below `./dumps` on the host unless `tracesdir` is changed.
- Do not hand-edit generated protobuf Python files; update their `.proto` sources and regenerate them with the appropriate protobuf tooling.
