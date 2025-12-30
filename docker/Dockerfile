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

ARG ROS_DISTRO=jazzy
FROM osrf/ros:${ROS_DISTRO}-desktop-full

ARG DEPENDENCIES=all_groups
ENV DEBIAN_FRONTEND=noninteractive
# Increase timeout for Poetry requests.

# Check whether the $DEPENDENCIES ARG has a valid value
RUN /bin/bash -c 'if [ "${DEPENDENCIES}" != "core_only" ] && [ "${DEPENDENCIES}" != "all_groups" ]; then \
  echo Error: invalid DEPENDENCIES value. Valid values "core_only", "all_groups" && exit 1; \
fi'

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

# Install dependencies
RUN apt update && apt install -y \
    python3 \
    python3-pip \
    git \
    wget \
    zip \
    ros-${ROS_DISTRO}-rai-interfaces \
    libglu1-mesa-dev \
    libxcb-xinerama0 \
    libfontconfig1-dev \
    libnvidia-gl-470 \
    libxcb-xkb-dev \
    libxkbcommon-x11-dev \
    libxkbcommon-dev \
    libxcb-xfixes0-dev \
    libxcb-xinput-dev \
    libxcb-xinput0 \
    libpcre2-16-0 \
    libunwind-dev

# Install Poetry
RUN --mount=type=cache,target=/root/.cache/pypoetry/cache curl -sSL https://install.python-poetry.org \
    | python3 - --version 2.1.3

ENV PATH="/root/.local/bin:$PATH"

# Copy the RAI repository
COPY . /rai
WORKDIR /rai

ENV POETRY_REQUESTS_TIMEOUT=100 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

RUN --mount=type=cache,target=/root/.cache/pypoetry/cache \
    --mount=type=cache,target=/root/.cache/pypoetry/artifacts \
    /bin/bash -c 'if [ "${DEPENDENCIES}" = "core_only" ]; then \
  poetry install -v; \
else \
  poetry install --all-groups -v; \
fi'

# Install ROS dependencies
RUN /bin/bash -c '. /opt/ros/${ROS_DISTRO}/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y'

# Build the workspace
RUN /bin/bash -c '. /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --symlink-install'

# Set up the environment to use Fast DDS
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/rai/docker/fastrtps_config.xml
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
