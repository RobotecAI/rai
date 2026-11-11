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

from . import views
from .connectors import (
    GoalHandle,
    GoalStatus,
    QoS,
    ROS2Connector,
    ROS2HRIConnector,
    Subscription,
)
from .context import ROS2Context
from .egress import MOTION, EgressDenied, EgressHooks, global_egress
from .exceptions import (
    ROS2GoalRejectedError,
    ROS2ParameterError,
    ROS2ServiceError,
    ROS2TimeoutError,
)
from .images import encode_image
from .parameters import get_param_value

__all__ = [
    "MOTION",
    "EgressDenied",
    "EgressHooks",
    "GoalHandle",
    "GoalStatus",
    "QoS",
    "ROS2Connector",
    "ROS2Context",
    "ROS2GoalRejectedError",
    "ROS2HRIConnector",
    "ROS2ParameterError",
    "ROS2ServiceError",
    "ROS2TimeoutError",
    "Subscription",
    "encode_image",
    "get_param_value",
    "global_egress",
    "views",
]
