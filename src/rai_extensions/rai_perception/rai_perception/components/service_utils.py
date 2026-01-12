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

"""Utility functions for ROS2 service name retrieval and client creation."""

from typing import Type

from rai.communication.ros2 import ROS2ServiceError, get_param_value
from rai.communication.ros2.connectors import ROS2Connector


def get_detection_service_name(connector: ROS2Connector) -> str:
    """Get detection service name from ROS2 parameter or use default.

    Reads from parameter: /detection_tool/service_name
    Default: "/detection" (generic, model-agnostic)

    Args:
        connector: ROS2 connector with node

    Returns:
        Service name string
    """
    default_service = "/detection"
    service_name = get_param_value(
        connector.node, "/detection_tool/service_name", default=default_service
    )
    if isinstance(service_name, str) and service_name:
        return service_name
    connector.node.get_logger().warning(
        f"Parameter /detection_tool/service_name is invalid, using default: {default_service}"
    )
    return default_service


def get_segmentation_service_name(connector: ROS2Connector) -> str:
    """Get segmentation service name from ROS2 parameter or use default.

    Reads from parameter: /segmentation_tool/service_name
    Default: "/segmentation" (generic, model-agnostic)

    Args:
        connector: ROS2 connector with node

    Returns:
        Service name string
    """
    default_service = "/segmentation"
    service_name = get_param_value(
        connector.node, "/segmentation_tool/service_name", default=default_service
    )
    if isinstance(service_name, str) and service_name:
        return service_name
    connector.node.get_logger().warning(
        f"Parameter /segmentation_tool/service_name is invalid, using default: {default_service}"
    )
    return default_service


def check_service_available(
    connector: ROS2Connector, service_name: str, timeout_sec: float = 0.1
) -> bool:
    """Check if a ROS2 service is available without waiting.

    Uses a generic service type to check availability. This is a lightweight check
    that doesn't require knowing the exact service type.

    Args:
        connector: ROS2 connector with node
        service_name: Service name to check
        timeout_sec: Timeout for checking service availability (default: 0.1)

    Returns:
        True if service is available, False otherwise
    """
    # Use std_srvs/Empty as a generic service type for availability checking
    # This works for any service since we only check if the service exists
    from std_srvs.srv import Empty

    try:
        cli = connector.node.create_client(Empty, service_name)
        return cli.wait_for_service(timeout_sec=timeout_sec)
    except Exception as e:
        # If service creation fails, log error and return False
        # This preserves the check function's boolean return while capturing error context
        connector.node.get_logger().debug(
            f"Service client creation failed for {service_name}: {e}"
        )
        return False


def create_service_client(
    connector: ROS2Connector,
    service_type: Type,
    service_name: str,
    timeout_sec: float = 1.0,
    max_wait_time: float = 0.0,
) -> object:
    """Create ROS2 service client and wait for service to be available.

    Args:
        connector: ROS2 connector with node
        service_type: ROS2 service type class
        service_name: Service name to connect to
        timeout_sec: Timeout for each wait attempt (default: 1.0)
        max_wait_time: Maximum total wait time. If 0, wait indefinitely (default: 0.0)

    Returns:
        Service client instance

    Raises:
        ROS2ServiceError: If service is not available within max_wait_time
    """
    import time

    cli = connector.node.create_client(service_type, service_name)
    start_time = time.time()
    while not cli.wait_for_service(timeout_sec=timeout_sec):
        if max_wait_time > 0 and time.time() - start_time >= max_wait_time:
            available_services = [
                s[0] for s in connector.node.get_service_names_and_types()
            ]
            raise ROS2ServiceError(
                service_name=service_name,
                timeout_sec=max_wait_time,
                service_state="unavailable",
                suggestion=(
                    f"Service not available after {max_wait_time}s. "
                    f"Available services: {sorted(available_services)[:10]}. "
                    f"Check if service is running or verify service name."
                ),
            )
        connector.node.get_logger().info(
            f"service {service_name} not available, waiting again..."
        )
    return cli
