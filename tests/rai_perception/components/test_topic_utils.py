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

import pytest

try:
    import rclpy  # noqa: F401

    _ = rclpy  # noqa: F841
except ImportError:
    pytest.skip("ROS2 is not installed", allow_module_level=True)

from unittest.mock import Mock, patch

from langchain_core.tools import BaseTool
from rai.communication.ros2.connectors import ROS2Connector
from rai_perception import (
    GetObjectGrippingPointsTool,
    discover_camera_topics,
    wait_for_perception_dependencies,
)
from rai_perception.tools.gripping_points_tools import GRIPPING_POINTS_TOOL_PARAM_PREFIX


def _setup_gripping_points_tool_params(node, **kwargs):
    """Helper to set up parameters for GetObjectGrippingPointsTool."""
    defaults = {
        "target_frame": "test_frame",
        "source_frame": "camera_frame",
        "camera_topic": "/test/camera",
        "depth_topic": "/test/depth",
        "camera_info_topic": "/test/camera_info",
    }
    defaults.update(kwargs)

    for key, value in defaults.items():
        node.declare_parameter(f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.{key}", value)


@pytest.fixture
def ros2_connector():
    """Fixture for ROS2Connector with proper setup/teardown."""
    rclpy.init()
    try:
        connector = ROS2Connector(executor_type="single_threaded")
        yield connector
    finally:
        connector.shutdown()
        rclpy.shutdown()


class TestWaitForPerceptionDependencies:
    """Test cases for wait_for_perception_dependencies utility."""

    def test_with_get_object_gripping_points_tool(self, ros2_connector):
        """Test that utility extracts services and topics from GetObjectGrippingPointsTool."""
        _setup_gripping_points_tool_params(ros2_connector.node)
        tool = GetObjectGrippingPointsTool(connector=ros2_connector)

        with (
            patch(
                "rai_perception.components.topic_utils.wait_for_ros2_services"
            ) as mock_wait_services,
            patch(
                "rai_perception.components.topic_utils.wait_for_ros2_topics"
            ) as mock_wait_topics,
        ):
            wait_for_perception_dependencies(ros2_connector, [tool])

            # Verify services were extracted and waited for
            mock_wait_services.assert_called_once()
            services = mock_wait_services.call_args[0][1]
            assert len(services) == 2
            assert tool.detection_service_name in services
            assert tool.segmentation_service_name in services

            # Verify topics were extracted and waited for
            mock_wait_topics.assert_called_once()
            topics = mock_wait_topics.call_args[0][1]
            assert len(topics) == 3
            config = tool.get_config()
            assert config["camera_topic"] in topics
            assert config["depth_topic"] in topics
            assert config["camera_info_topic"] in topics

    def test_with_tool_having_service_name_property(self):
        """Test that utility works with tools that have service_name property."""
        connector = Mock(spec=ROS2Connector)
        connector.node = Mock()

        # Create a mock tool with service_name property
        mock_tool = Mock(spec=BaseTool)
        mock_tool.service_name = "/test/detection_service"

        tools = [mock_tool]

        # Should raise RuntimeError because segmentation_service is None
        with pytest.raises(RuntimeError, match="Required perception tools not found"):
            wait_for_perception_dependencies(connector, tools)

    def test_without_perception_tools(self):
        """Test that utility raises error when no perception tools are found."""
        connector = Mock(spec=ROS2Connector)
        connector.node = Mock()

        # Create a mock tool without perception capabilities
        mock_tool = Mock(spec=BaseTool)
        tools = [mock_tool]

        with pytest.raises(RuntimeError, match="Required perception tools not found"):
            wait_for_perception_dependencies(connector, tools)

    def test_with_empty_tools_list(self):
        """Test that utility raises error with empty tools list."""
        connector = Mock(spec=ROS2Connector)
        connector.node = Mock()

        tools = []

        with pytest.raises(RuntimeError, match="Required perception tools not found"):
            wait_for_perception_dependencies(connector, tools)

    def test_with_multiple_tools(self, ros2_connector):
        """Test that utility works when multiple tools are present."""
        _setup_gripping_points_tool_params(ros2_connector.node)
        perception_tool = GetObjectGrippingPointsTool(connector=ros2_connector)
        tools = [Mock(spec=BaseTool), perception_tool, Mock(spec=BaseTool)]

        with (
            patch(
                "rai_perception.components.topic_utils.wait_for_ros2_services"
            ) as mock_wait_services,
            patch(
                "rai_perception.components.topic_utils.wait_for_ros2_topics"
            ) as mock_wait_topics,
        ):
            wait_for_perception_dependencies(ros2_connector, tools)

            mock_wait_services.assert_called_once()
            mock_wait_topics.assert_called_once()


class TestDiscoverCameraTopics:
    """Test cases for discover_camera_topics utility."""

    def test_discover_camera_topics(self, ros2_connector):
        """Test camera topic discovery utility."""
        discovered = discover_camera_topics(ros2_connector)

        assert isinstance(discovered, dict)
        assert "image_topics" in discovered
        assert "depth_topics" in discovered
        assert "camera_info_topics" in discovered
        assert "all_topics" in discovered

        # All should be lists
        assert isinstance(discovered["image_topics"], list)
        assert isinstance(discovered["depth_topics"], list)
        assert isinstance(discovered["camera_info_topics"], list)
        assert isinstance(discovered["all_topics"], list)
