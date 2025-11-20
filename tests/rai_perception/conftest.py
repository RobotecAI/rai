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
# See the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_connector():
    """Mock ROS2Connector for testing perception tools.

    Provides a mock ROS2Connector with attributes and methods used by
    perception tools:
    - connector.node: Mock node with create_client, get_logger, get_parameter
    - connector.receive_message: Mock method for receiving ROS2 messages

    Note: Unlike communication package tests which use real ROS2Connector
    instances with actual ROS2 infrastructure (integration tests), we use
    MagicMock here because:
    - We're testing tool logic, not ROS2 integration
    - Unit tests should be fast and not require ROS2 infrastructure
    - We can control mock behavior for specific test scenarios
    """
    connector = MagicMock()

    # Mock the node with all required methods
    mock_node = MagicMock()
    mock_node.create_client = MagicMock()
    mock_node.get_logger = MagicMock(return_value=MagicMock())
    mock_node.get_parameter = MagicMock()

    connector.node = mock_node
    connector._node = mock_node  # Some code accesses _node directly
    connector.receive_message = MagicMock()

    return connector


@contextmanager
def patch_ros2_for_agent_tests(mock_connector):
    """Context manager to patch ROS2Connector and rclpy.ok for agent tests.

    This patches:
    - ROS2Connector at both the source and usage locations to return the provided mock_connector
    - rclpy.ok to return False (prevents cleanup_agent from calling rclpy.shutdown)

    Use this in agent tests where BaseVisionAgent creates a real ROS2Connector
    which would otherwise require ROS2 to be initialized.
    """
    with (
        patch("rai.communication.ros2.ROS2Connector", return_value=mock_connector),
        patch(
            "rai_perception.agents.base_vision_agent.ROS2Connector",
            return_value=mock_connector,
        ),
        patch("rclpy.ok", return_value=False),
    ):
        yield
