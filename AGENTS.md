# Project overview

Google Research Football is a Python reinforcement-learning environment built around a bundled, modified Gameplay Football C++ engine. The Python package exposes Gym-compatible environments, scenarios, wrappers, player implementations, replay utilities, and example training programs; installing the package compiles and packages the native engine.

# Setup

Run commands from the repository root. The uv project requires Python 3.9-3.10, pins the development interpreter to Python 3.9, the package is version `2.10.3`, and the native engine uses C++14/CMake 3.5+.

Linux prerequisites and environment:

```sh
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip
./build-uv.sh
```

On macOS, install native dependencies first:

```sh
brew install git python3 cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-python3
./build-uv.sh
```

`build-uv.sh` is the single Python setup entry point: it installs uv if necessary, creates/synchronizes `.venv` from `uv.lock`, compiles the engine, and verifies imports. It defaults to `/home/artem/miniconda3/envs/gfootball/bin/python`; override `GFOOTBALL_PYTHON` and `GFOOTBALL_NATIVE_PREFIX` together for another compatible Python 3.9/Boost.Python toolchain. Use Homebrew's Python on macOS because `boost-python3` is built for it. Windows additionally requires Visual Studio C++ tools, CMake, and vcpkg. See `gfootball/doc/compile_engine.md` for platform-specific native prerequisites.

# Build / run

Set up or refresh the development build:

```sh
./build-uv.sh
```

The setuptools `build_ext` hook invokes `gfootball/build_game_engine.sh` on Unix, which clears the engine CMake cache, runs CMake and Make with memory-aware parallelism, and links the resulting library as `_gameplayfootball.so`. For a standalone distribution build, export the same toolchain first: `GFOOTBALL_BUILD_PYTHON="$GFOOTBALL_PYTHON" CMAKE_PREFIX_PATH="$GFOOTBALL_NATIVE_PREFIX" LD_LIBRARY_PATH="$GFOOTBALL_NATIVE_PREFIX/lib" uv build`.

Run the checked-in default configuration with rendering:

```sh
uv run ./start_game.sh
```

This expands to:

```sh
python3 -m gfootball.play_game \
  --config_file gfootball/configs/default.yaml \
  --render=true
```

For the standard built-in configuration instead:

```sh
python3 -m gfootball.play_game --action_set=full
```

Stop play with Ctrl+C. Rendering requires a usable SDL/display environment.

# Test

Tests are standalone `absltest`/`unittest` scripts under `gfootball/env`; there is no pytest/tox configuration. The CI-equivalent suite command on Unix is:

```sh
for test_file in gfootball/env/*test.py; do
UNITTEST_IN_DOCKER=1 uv run python "$test_file" || exit 1
done
```

`UNITTEST_IN_DOCKER=1` skips rendering-only wrapper tests and is useful on headless hosts. Run a single file or test method with:

```sh
UNITTEST_IN_DOCKER=1 uv run python gfootball/env/football_action_set_test.py
UNITTEST_IN_DOCKER=1 uv run python -m unittest \
  gfootball.env.football_action_set_test.FootballActionSetTest.test_action_set_full
```

The repository also provides an expensive Docker smoke test that builds Ubuntu images, runs all environment tests, and starts a short PPO2 training job:

```sh
./run_docker_test.sh
```

# Code style

No formatter, linter, pre-commit hook, or style CI is configured. Preserve the style of the file being edited instead of applying repository-wide formatting. Python generally uses `snake_case` functions/variables, `PascalCase` classes, module docstrings, and two-space indentation in the older environment code. Scenario files expose `build_scenario(builder)` and intentionally use the engine-facing `SetTeam`, `AddPlayer`, and related APIs. C++ under `third_party/gfootball_engine` is legacy code with mixed conventions; keep changes focused and match adjacent code. Name Python tests `*_test.py` and test methods `test_*`.

# Repo structure

- `gfootball/env/`: Gym environment, core engine bridge, actions, observations, wrappers, players, and unit/E2E tests.
- `gfootball/scenarios/`: built-in scenario definitions; `gfootball/scenarios/tests/` contains deterministic scenarios consumed by environment tests, not a separate test runner.
- `gfootball/configs/default.yaml`: configuration used by `start_game.sh`.
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
- `start_game.sh` enables full episode dumps and video through `default.yaml`. Each run creates a timestamped directory below `/tmp/dumps` unless `tracesdir` is changed.
- Do not hand-edit generated protobuf Python files; update their `.proto` sources and regenerate them with the appropriate protobuf tooling.
