# Copyright (C) 2025 Robotec.AI
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

"""Helper functions for deprecated agent wrappers."""

from pathlib import Path
from typing import Type

import rclpy
from rai.communication.ros2 import ROS2Connector
from rclpy.parameter import Parameter


def create_service_wrapper(
    service_class: Type,
    ros2_name: str,
    model_name: str,
    service_name: str,
    weights_root_path: str | Path = Path.home() / Path(".cache/rai"),
) -> tuple[ROS2Connector, object]:
    """Create a service instance with ROS2 parameters configured.

    Helper function to reduce duplication in deprecated agent wrapper classes.

    Args:
        service_class: Service class to instantiate
        ros2_name: ROS2 node name
        model_name: Model name parameter value
        service_name: Service name parameter value
        weights_root_path: Path to weights root directory

    Returns:
        Tuple of (ros2_connector, service_instance)
    """
    ros2_connector = ROS2Connector(ros2_name, executor_type="single_threaded")

    ros2_connector.node.set_parameters(
        [
            Parameter(
                "model_name",
                rclpy.parameter.Parameter.Type.STRING,
                model_name,
            ),
            Parameter(
                "service_name",
                rclpy.parameter.Parameter.Type.STRING,
                service_name,
            ),
        ]
    )

    service_instance = service_class(
        weights_root_path, ros2_name, ros2_connector=ros2_connector
    )

    return ros2_connector, service_instance
