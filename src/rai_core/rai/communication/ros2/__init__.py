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

import importlib.util

if importlib.util.find_spec("rclpy") is None:
    raise ImportError(
        "This is a ROS2 feature. Make sure ROS2 is installed and sourced."
    )

from .api import (
    IROS2Message,  # TODO: IROS2Message should not be a part of the public API
)
from .connectors import ROS2Connector, ROS2HRIConnector
from .context import ROS2Context
from .exceptions import ROS2ParameterError, ROS2ServiceError
from .messages import ROS2HRIMessage, ROS2Message
from .parameters import get_param_value
from .waiters import wait_for_ros2_actions, wait_for_ros2_services, wait_for_ros2_topics

__all__ = [
    "IROS2Message",
    "ROS2Connector",
    "ROS2Context",
    "ROS2HRIConnector",
    "ROS2HRIMessage",
    "ROS2Message",
    "ROS2ParameterError",
    "ROS2ServiceError",
    "get_param_value",
    "wait_for_ros2_actions",
    "wait_for_ros2_services",
    "wait_for_ros2_topics",
]
