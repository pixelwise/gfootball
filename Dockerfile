FROM python:3.9-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get --no-install-recommends install -yq \
    build-essential \
    cmake \
    libboost-all-dev \
    libdirectfb-dev \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libsdl2-dev \
    libsdl2-gfx-dev \
    libsdl2-image-dev \
    libsdl2-ttf-dev \
    libst-dev \
    make \
    pkg-config \
    xauth \
    xvfb \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.30 /uv /uvx /bin/

WORKDIR /gfootball
COPY . .

# This compiles the native engine and creates the locked uv environment.
RUN uv sync --locked
