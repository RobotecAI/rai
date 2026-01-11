# Copyright (C) 2025 Julia Jia
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

import logging
from typing import Dict, List

from langchain_core.tools import BaseTool
from rai.communication.ros2 import wait_for_ros2_services, wait_for_ros2_topics
from rai.communication.ros2.connectors import ROS2Connector

logger = logging.getLogger(__name__)


def discover_camera_topics(connector: ROS2Connector) -> Dict[str, List[str]]:
    """Discover available camera-related topics in the ROS2 system.

    Searches for topics matching common camera naming patterns and categorizes them.

    Args:
        connector: ROS2 connector to query topics from

    Returns:
        Dictionary with keys:
        - "image_topics": List of image topics (sensor_msgs/Image)
        - "depth_topics": List of depth topics (sensor_msgs/Image)
        - "camera_info_topics": List of camera info topics (sensor_msgs/CameraInfo)
        - "all_topics": List of all available topics
    """
    try:
        all_topics = connector.get_topics_names_and_types()
    except Exception as e:
        logger.warning(f"Failed to query topics: {e}")
        return {
            "image_topics": [],
            "depth_topics": [],
            "camera_info_topics": [],
            "all_topics": [],
        }

    image_topics = []
    depth_topics = []
    camera_info_topics = []
    all_topic_names = []

    for topic_name, topic_types in all_topics:
        all_topic_names.append(topic_name)
        topic_types_str = " ".join(topic_types).lower()

        # Check for image topics
        if (
            "sensor_msgs/msg/image" in topic_types_str
            or "sensor_msgs/Image" in topic_types_str
        ):
            topic_lower = topic_name.lower()
            if "depth" in topic_lower or "depth" in topic_name:
                depth_topics.append(topic_name)
            else:
                image_topics.append(topic_name)

        # Check for camera info topics
        if (
            "sensor_msgs/msg/camerainfo" in topic_types_str
            or "sensor_msgs/CameraInfo" in topic_types_str
        ):
            camera_info_topics.append(topic_name)

    return {
        "image_topics": sorted(image_topics),
        "depth_topics": sorted(depth_topics),
        "camera_info_topics": sorted(camera_info_topics),
        "all_topics": sorted(all_topic_names),
    }


def _validate_topics(connector: ROS2Connector, required_topics: List[str]) -> List[str]:
    """Validate topics exist.

    Args:
        connector: ROS2 connector
        required_topics: List of required topic names

    Returns:
        List of missing topic names
    """
    try:
        available_topics = [
            topic[0] for topic in connector.get_topics_names_and_types()
        ]
    except Exception:
        available_topics = []

    return [t for t in required_topics if t not in available_topics]


def wait_for_perception_dependencies(
    connector: ROS2Connector, tools: List[BaseTool]
) -> None:
    """Wait for ROS2 services and topics required by perception tools.

    Automatically extracts service names and topics from perception tools
    in the tools list and waits for them to be available.

    Args:
        connector: ROS2 connector to use for waiting
        tools: List of tools that may include perception tools

    Raises:
        RuntimeError: If required perception tools are not found in tools list
        TimeoutError: If topics/services don't become available
    """
    # Lazy import to avoid circular dependency
    from rai_perception.tools.gripping_points_tools import GetObjectGrippingPointsTool

    # Extract service names from perception tools
    detection_service = None
    segmentation_service = None

    for tool in tools:
        if isinstance(tool, GetObjectGrippingPointsTool):
            detection_service = tool.detection_service_name
            segmentation_service = tool.segmentation_service_name
            break
        elif hasattr(tool, "service_name"):
            # For tools that only use detection service
            detection_service = tool.service_name
            break

    if detection_service is None or segmentation_service is None:
        raise RuntimeError(
            "Required perception tools not found in tools list. "
            "GetObjectGrippingPointsTool or tools with service_name property required."
        )

    required_services = [detection_service, segmentation_service]

    # Extract topics from perception tools
    required_topics = None
    for tool in tools:
        if isinstance(tool, GetObjectGrippingPointsTool):
            config = tool.get_config()
            required_topics = [
                config["camera_topic"],
                config["depth_topic"],
                config["camera_info_topic"],
            ]
            break

    if required_topics is None:
        raise RuntimeError(
            "GetObjectGrippingPointsTool not found in tools list. "
            "Cannot determine required topics."
        )

    # Wait for services
    try:
        wait_for_ros2_services(connector, required_services)
    except TimeoutError as e:
        available_services = [s[0] for s in connector.get_services_names_and_types()]
        raise TimeoutError(
            f"{str(e)}\n"
            f"Available services: {sorted(available_services)[:10]}\n"
            f"Expected: {required_services}\n"
            f"Tip: Set ROS2 parameters '/detection_tool/service_name' and "
            f"'/segmentation_tool/service_name' to match your service names."
        ) from e

    # Wait for topics
    try:
        wait_for_ros2_topics(connector, required_topics)
    except TimeoutError as e:
        missing_at_timeout = _validate_topics(connector, required_topics)
        discovered = discover_camera_topics(connector)

        raise TimeoutError(
            f"{str(e)}\n"
            f"Missing topics: {missing_at_timeout}\n"
            f"Available image topics: {discovered['image_topics'][:5]}\n"
            f"Available depth topics: {discovered['depth_topics'][:5]}\n"
            f"Available camera_info topics: {discovered['camera_info_topics'][:5]}\n"
            f"Tip: Override topic parameters before tool initialization:\n"
            f"  node.declare_parameter('perception.gripping_points.camera_topic', '/your/topic')"
        ) from e
