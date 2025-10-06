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
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.8.4
ENV PATH="/root/.local/bin:$PATH"

# Clone and setup RAI
WORKDIR /rai
RUN git clone https://github.com/RobotecAI/rai.git .

# Install Python dependencies with Poetry
RUN poetry install --with nomad,perception

# Install ROS dependencies
RUN /bin/bash -c '. /opt/ros/${ROS_DISTRO}/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y'

# Build the workspace
RUN /bin/bash -c '. /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --symlink-install'
