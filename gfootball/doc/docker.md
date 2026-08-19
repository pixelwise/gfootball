# Running in Docker #

Docker is the recommended setup when native engine dependencies should not be
installed on the host. The default image includes Python 3.9, uv, SDL2,
Boost.Python, and the C++ build toolchain; the engine is compiled during the
image build.

## Build the image

```shell
git clone https://github.com/google-research/football.git
cd football
docker build -t gfootball .
```

No Python, uv, compiler, SDL, or Boost installation is required on the host.
Docker itself must be installed and usable by the current user.

## Start the default game

The Docker entry point is `start_game_docker.sh`. It runs the same default
configuration as the former local start script, with `render=true` and no
host display. The engine selects off-screen rendering and writes dumps and
videos to `./dumps` beside the script:

```shell
./start_game_docker.sh
```

The script runs the container under your host UID/GID, so generated files are
readable on the host. It sets a temporary uv cache inside the container and
removes the container when the game finishes. Override the image tag when
needed:

```shell
GFOOTBALL_DOCKER_IMAGE=my-gfootball-image ./start_game_docker.sh
```

## Start a container shell

`docker run` creates and starts a container. The following command opens a
shell in a temporary container; `exit` stops and removes it:

```shell
docker run --rm -it --name gfootball-shell gfootball bash
```

Inside the shell, use the image's locked uv environment:

```shell
uv run --locked python -c 'import gfootball; import gfootball_engine; print("ready")'
uv run --locked python gfootball/env/football_env_core_test.py
```

To keep a development container running in the background, start it with a
long-lived command and enter it with `docker exec`:

```shell
docker run -d --name gfootball-dev gfootball sleep infinity
docker exec -it gfootball-dev bash
docker stop gfootball-dev
docker rm gfootball-dev
```

## Run headlessly

Run an environment test without a GPU or display server:

```shell
docker run --rm gfootball \
  uv run --locked python gfootball/env/football_env_core_test.py
```

The default configuration writes episode dumps and video to `/tmp/dumps` in
the container. Mount a local directory to retain them after `--rm` removes the
container. This is the expanded command used by `start_game_docker.sh`:

```shell
mkdir -p dumps
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -v "$(pwd)/dumps:/tmp/dumps" \
  gfootball env -u DISPLAY uv run --locked python -m gfootball.play_game \
  --config_file gfootball/configs/default.yaml --render=true
```

## Run the game with rendering (Linux)

The container can use an X11 display supplied by the host. Allow local Docker
containers running as your current user to access the display, then run the
game with a visible window:

```shell
mkdir -p dumps
xhost +si:localuser:"$(id -un)"
docker run --rm -it \
  --user "$(id -u):$(id -g)" \
  -e DISPLAY="$DISPLAY" \
  -e UV_CACHE_DIR=/tmp/uv-cache \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd)/dumps:/tmp/dumps" \
  gfootball uv run --locked python -m gfootball.play_game \
  --config_file gfootball/configs/default.yaml --render=true
xhost -si:localuser:"$(id -un)"
```

The configuration runs one `11_vs_11_stochastic` episode with the full action
set, enables rendering, and writes its dumps to the host `./dumps` directory.
Stop the game with `Ctrl+C`; run the final `xhost` command afterwards to revoke
display access.

Drop GPU-related Docker flags unless GPU rendering is specifically configured;
software rendering is supported. Host-X11 rendering is Linux-specific. On
macOS and Windows, run headlessly or configure a compatible X server before
using the rendering command.

## Validate the image

Run the complete headless environment suite and an Xvfb rendering smoke test:

```shell
./run_docker_test.sh
```

`Dockerfile_examples` remains a separate legacy TensorFlow 1/OpenAI Baselines
image for the training examples; it is not part of the default Docker workflow.
