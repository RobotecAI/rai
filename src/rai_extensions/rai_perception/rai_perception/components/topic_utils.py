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
from typing import Dict, List, Optional

from langchain_core.tools import BaseTool
from rai.communication.ros2 import ROS2ServiceError
from rai.communication.ros2.connectors import ROS2Connector

from rai_perception.components.service_utils import wait_for

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


def wait_for_perception_dependencies(
    connector: ROS2Connector,
    tools: List[BaseTool],
    timeout: Optional[float] = None,
) -> None:
    """Wait for the services and topics the perception tools in tools need.

    GetObjectGrippingPointsTool (and GetObjectPositionsTool) need the detection
    and segmentation services and the camera, depth and camera_info topics;
    tools with a detection_service_name need that service only. timeout=None
    waits forever.

    Raises:
        RuntimeError: If no perception tool is in tools
        ROS2ServiceError: If services are not available within timeout
        TimeoutError: If topics are not available within timeout
    """
    from rai_perception.tools.gripping_points_tools import GetObjectGrippingPointsTool

    services: list[str] = []
    topics: list[str] = []
    for tool in tools:
        if isinstance(tool, GetObjectGrippingPointsTool):
            services += [tool.detection_service_name, tool.segmentation_service_name]
            topics += [tool.camera_topic, tool.depth_topic, tool.camera_info_topic]
        elif isinstance(getattr(tool, "detection_service_name", None), str):
            services.append(getattr(tool, "detection_service_name"))
    if not services:
        raise RuntimeError(
            "No perception tools in the tools list: GetObjectGrippingPointsTool or "
            "tools with a detection_service_name required."
        )
    services, topics = list(dict.fromkeys(services)), list(dict.fromkeys(topics))

    if absent := wait_for(connector, services, "service", timeout):
        available = sorted(s for s, _ in connector.get_services_names_and_types())
        raise ROS2ServiceError(
            service_name=", ".join(absent),
            timeout_sec=timeout or 0.0,
            service_state="unavailable",
            suggestion=(
                f"Services not available. Available: {available[:10]}. "
                f"Expected: {services}. Set the tools' detection_service_name and "
                "segmentation_service_name to match your service names."
            ),
        )
    if absent := wait_for(connector, topics, "topic", timeout):
        found = discover_camera_topics(connector)
        raise TimeoutError(
            f"Topics {sorted(absent)} not available within {timeout} seconds\n"
            f"Available image topics: {found['image_topics'][:5]}\n"
            f"Available depth topics: {found['depth_topics'][:5]}\n"
            f"Available camera_info topics: {found['camera_info_topics'][:5]}\n"
            "Tip: set the tool's camera_topic, depth_topic and camera_info_topic"
        )
