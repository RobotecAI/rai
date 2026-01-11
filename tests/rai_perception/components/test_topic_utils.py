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


class TestWaitForPerceptionDependencies:
    """Test cases for wait_for_perception_dependencies utility."""

    def test_with_get_object_gripping_points_tool(self):
        """Test that utility extracts services and topics from GetObjectGrippingPointsTool."""
        rclpy.init()
        try:
            connector = ROS2Connector(executor_type="single_threaded")
            node = connector.node

            # Set up parameters for the tool
            node.declare_parameter(
                f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.target_frame", "test_frame"
            )
            node.declare_parameter(
                f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.source_frame", "camera_frame"
            )
            node.declare_parameter(
                f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_topic", "/test/camera"
            )
            node.declare_parameter(
                f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.depth_topic", "/test/depth"
            )
            node.declare_parameter(
                f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_info_topic",
                "/test/camera_info",
            )

            # Create tool
            tool = GetObjectGrippingPointsTool(connector=connector)
            tools = [tool]

            # Mock the wait functions to verify they're called with correct values
            with (
                patch(
                    "rai_perception.components.topic_utils.wait_for_ros2_services"
                ) as mock_wait_services,
                patch(
                    "rai_perception.components.topic_utils.wait_for_ros2_topics"
                ) as mock_wait_topics,
            ):
                wait_for_perception_dependencies(connector, tools)

                # Verify services were extracted and waited for
                mock_wait_services.assert_called_once()
                call_args = mock_wait_services.call_args
                assert call_args[0][0] == connector
                services = call_args[0][1]
                assert len(services) == 2
                assert tool.detection_service_name in services
                assert tool.segmentation_service_name in services

                # Verify topics were extracted and waited for
                mock_wait_topics.assert_called_once()
                call_args = mock_wait_topics.call_args
                assert call_args[0][0] == connector
                topics = call_args[0][1]
                assert len(topics) == 3
                config = tool.get_config()
                assert config["camera_topic"] in topics
                assert config["depth_topic"] in topics
                assert config["camera_info_topic"] in topics

        finally:
            rclpy.shutdown()

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

    def test_with_multiple_tools(self):
        """Test that utility works when multiple tools are present."""
        rclpy.init()
        try:
            connector = ROS2Connector(executor_type="single_threaded")
            node = connector.node

            # Set up parameters
            node.declare_parameter(
                f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.target_frame", "test_frame"
            )
            node.declare_parameter(
                f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.source_frame", "camera_frame"
            )
            node.declare_parameter(
                f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_topic", "/test/camera"
            )
            node.declare_parameter(
                f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.depth_topic", "/test/depth"
            )
            node.declare_parameter(
                f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_info_topic",
                "/test/camera_info",
            )

            # Create perception tool and other tools
            perception_tool = GetObjectGrippingPointsTool(connector=connector)
            other_tool = Mock(spec=BaseTool)
            tools = [other_tool, perception_tool, Mock(spec=BaseTool)]

            # Should work - finds perception tool in the list
            with (
                patch(
                    "rai_perception.components.topic_utils.wait_for_ros2_services"
                ) as mock_wait_services,
                patch(
                    "rai_perception.components.topic_utils.wait_for_ros2_topics"
                ) as mock_wait_topics,
            ):
                wait_for_perception_dependencies(connector, tools)

                # Verify it was called
                mock_wait_services.assert_called_once()
                mock_wait_topics.assert_called_once()

        finally:
            rclpy.shutdown()


class TestDiscoverCameraTopics:
    """Test cases for discover_camera_topics utility."""

    def test_discover_camera_topics(self):
        """Test camera topic discovery utility."""
        rclpy.init()
        try:
            connector = ROS2Connector(executor_type="single_threaded")

            # Discover topics
            discovered = discover_camera_topics(connector)

            # Verify structure
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

        finally:
            rclpy.shutdown()
