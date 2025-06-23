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

ARG DEPENDENCIES=core_only
ENV DEBIAN_FRONTEND=noninteractive

# Check whether the $DEPENDENCIES ARG has a valid value
RUN /bin/bash -c 'if [ "${DEPENDENCIES}" != "core_only" ] && [ "${DEPENDENCIES}" != "all_groups" ]; then \
  echo Error: invalid DEPENDENCIES value. Valid values "core_only", "all_groups" && exit 1; \
fi'


# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    zip

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.8.4
ENV PATH="/root/.local/bin:$PATH"

# Clone and setup RAI
WORKDIR /rai
RUN git clone https://github.com/RobotecAI/rai.git .

# Set up the environment to use Fast DDS
ENV FASTRTPS_DEFAULT_PROFILES_FILE=/rai/fastrtpsa_config.xml
ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Temporary change to use branch with bug fixes
RUN git pull
RUN git switch dm/feat/docker-image-manipulator-demo

# # Install Python dependencies with Poetry
# RUN poetry install --with nomad,openset


RUN /bin/bash -c 'if [ "${DEPENDENCIES}" = "core_only" ]; then \
  poetry install; \
else \
  poetry install --all-groups; \
fi'

# Install ROS dependencies
RUN /bin/bash -c '. /opt/ros/${ROS_DISTRO}/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y'

# Build the workspace
RUN /bin/bash -c '. /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --symlink-install'
