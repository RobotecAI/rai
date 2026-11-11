# syntax=docker/dockerfile:1
# Copyright (C) 2024 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RAI development image for ROS 2 Jazzy and Lyrical (amd64).
#
# Targets:
#   dev  all dependency groups from uv.lock (CUDA torch)
#   ci   the same with CPU-only torch; published as ghcr.io/robotecai/rai-dev:<distro>
#
#   docker build -f docker/Dockerfile --build-arg ROS_DISTRO=lyrical --target ci -t rai-dev:lyrical .
#
# The image is a regular dev workspace in /rai: src/rai_interfaces, install/ and a
# `uv venv --system-site-packages` in /rai/.venv (see scripts/dev_env.sh).
# Layers are ordered so that a source-only change rebuilds only the last stage.

ARG ROS_DISTRO=lyrical
ARG UV_VERSION=0.13.6

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ROS, system packages and rai_interfaces.
FROM ros:${ROS_DISTRO}-ros-base AS base
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

COPY --from=uv /uv /uvx /usr/local/bin/

# Retry downloads. `apt-get update --error-on=any` fails instead of leaving a partial package list.
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries
# PortAudio backs sounddevice. jackd2 and pulseaudio provide virtual devices for the audio tests
# (docker/virtual-audio.sh).
# libgl1 and libglib2.0 are needed by opencv-python, which some dependencies pull in.
RUN apt-get update --error-on=any && apt-get install -y --no-install-recommends \
        git \
        jackd2 \
        libgl1 \
        libglib2.0-0t64 \
        libportaudio2 \
        portaudio19-dev \
        pulseaudio \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /rai

# rai_interfaces has no Lyrical apt package, so it is built from source (pinned in ros_deps.repos).
COPY ros_deps.repos ./
COPY --parents src/**/package.xml ./
RUN vcs import --input ros_deps.repos . \
    && apt-get update --error-on=any \
    && rosdep update --rosdistro "${ROS_DISTRO}" \
    && rosdep install --from-paths src --ignore-src -y \
    && rm -rf /var/lib/apt/lists/* /root/.ros/rosdep
RUN source "/opt/ros/${ROS_DISTRO}/setup.bash" \
    && colcon build --symlink-install --packages-select rai_interfaces \
    && rm -rf log

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=100 \
    UV_HTTP_RETRIES=10 \
    UV_NO_SYNC=1

# Python dependencies only: workspace metadata, no sources.
COPY pyproject.toml uv.lock ./
COPY --parents src/**/pyproject.toml ./
RUN uv venv --system-site-packages --python /usr/bin/python3 .venv

# Fast DDS over UDP only, so that a container can talk to ROS 2 on the host.
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/rai/docker/fastrtps_config.xml \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp
COPY docker/entrypoint.sh /usr/local/bin/rai-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/rai-entrypoint.sh"]
CMD ["bash"]

# All groups except docs, with the CUDA torch from uv.lock.
FROM base AS deps-cuda
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace --all-groups --no-group docs

# CPU-only torch: the CUDA build does not fit on a GitHub-hosted runner.
# --no-deps keeps the exact versions from uv.lock.
FROM base AS deps-cpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --frozen --all-groups --no-group docs --no-hashes --no-emit-workspace \
        | grep -vE '^(nvidia-|cuda-|triton==)' > /tmp/requirements-cpu.txt \
    && uv pip install --python .venv/bin/python --no-deps --torch-backend cpu -r /tmp/requirements-cpu.txt \
    && rm /tmp/requirements-cpu.txt

FROM deps-cpu AS ci
COPY . /rai
RUN source "/opt/ros/${ROS_DISTRO}/setup.bash" \
    && colcon build --symlink-install && rm -rf log \
    && uv export --frozen --all-groups --no-group docs --no-hashes | grep '^-e ' > /tmp/workspace.txt \
    && uv pip install --python .venv/bin/python --no-deps -r /tmp/workspace.txt \
    && rm /tmp/workspace.txt

# Default target: last, so a plain `docker build` produces it.
FROM deps-cuda AS dev
COPY . /rai
RUN --mount=type=cache,target=/root/.cache/uv \
    source "/opt/ros/${ROS_DISTRO}/setup.bash" \
    && colcon build --symlink-install && rm -rf log \
    && uv sync --frozen --all-groups --no-group docs
