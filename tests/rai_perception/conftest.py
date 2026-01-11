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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def pytest_configure(config):
    """Configure pytest to suppress deprecation warnings for deprecated agent classes."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore:GroundedSamAgent is deprecated:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:GroundingDinoAgent is deprecated:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:BaseVisionAgent is deprecated:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Importing from timm.models.layers is deprecated:FutureWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:rai_perception.vision_markup.boxer.GDBoxer is deprecated:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:rai_perception.vision_markup.segmenter.GDSegmenter is deprecated:DeprecationWarning",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:rai_perception.vision_markup is deprecated:DeprecationWarning",
    )


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

    # Track parameters that are set
    _parameters = {}

    def set_parameters(params):
        """Store parameters for later retrieval."""
        for param in params:
            _parameters[param.name] = param
        return []  # Return empty list (success)

    def has_parameter(name):
        """Check if parameter exists."""
        return name in _parameters

    def get_parameter(name):
        """Get parameter by name."""
        if name not in _parameters:
            from rclpy.exceptions import ParameterNotDeclaredException

            raise ParameterNotDeclaredException(f"Parameter '{name}' not found")
        return _parameters[name]

    # Mock the node with all required methods
    mock_node = MagicMock()
    mock_node.create_client = MagicMock()
    mock_node.get_logger = MagicMock(return_value=MagicMock())
    mock_node.get_parameter = get_parameter
    mock_node.has_parameter = has_parameter
    mock_node.declare_parameter = MagicMock()
    mock_node.set_parameters = set_parameters
    mock_node.get_clock = MagicMock()
    mock_node.create_service = MagicMock()

    connector.node = mock_node
    connector._node = mock_node  # Some code accesses _node directly
    connector.receive_message = MagicMock()
    connector.create_service = MagicMock()
    connector.shutdown = MagicMock()

    return connector


@contextmanager
def patch_ros2_for_agent_tests(mock_connector):
    """Context manager to patch ROS2Connector and rclpy.ok for agent tests.

    This patches:
    - ROS2Connector at both the source and usage locations to return the provided mock_connector
    - rclpy.ok to return False (prevents cleanup_agent from calling rclpy.shutdown)

    Use this in agent tests where agents create ROS2Connectors and delegate to services.
    Since agents now use services internally, this patches both agent and service locations.
    """
    with (
        patch("rai.communication.ros2.ROS2Connector", return_value=mock_connector),
        patch(
            "rai_perception.agents._helpers.ROS2Connector",
            return_value=mock_connector,
        ),
        patch(
            "rai_perception.services.base_vision_service.ROS2Connector",
            return_value=mock_connector,
        ),
        patch("rclpy.ok", return_value=False),
    ):
        yield


@contextmanager
def patch_ros2_for_service_tests(mock_connector):
    """Context manager to patch ROS2Connector for service tests.

    This patches:
    - ROS2Connector at service locations to return the provided mock_connector

    Use this in service tests where services create ROS2Connectors.
    """
    with (
        patch("rai.communication.ros2.ROS2Connector", return_value=mock_connector),
        patch(
            "rai_perception.services.base_vision_service.ROS2Connector",
            return_value=mock_connector,
        ),
        patch("rclpy.ok", return_value=False),
    ):
        yield


def create_valid_weights_file(weights_path: Path, size_mb: int = 2) -> None:
    """Helper to create a valid weights file for testing.

    Args:
        weights_path: Path where the weights file should be created
        size_mb: Size of the file in megabytes (default: 2MB)
    """
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"0" * (size_mb * 1024 * 1024))


def get_weights_path(tmp_path: Path) -> Path:
    """Helper to get the standard weights path for testing.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Path to the weights file
    """
    return tmp_path / "vision" / "weights" / "test_weights.pth"


def setup_mock_clock(instance):
    """Setup mock clock for tests.

    The code calls clock().now().to_msg() to get ts, then passes ts to
    to_detection_msg which expects rclpy.time.Time and calls ts.to_msg() again.
    However, ts is also assigned to response.detections.header.stamp which expects
    builtin_interfaces.msg.Time.

    ROS2 Humble vs Jazzy difference:
    - Humble: Strict type checking in __debug__ mode requires actual BuiltinTime
      instances, not MagicMock objects. Using MagicMock causes AssertionError.
    - Jazzy: More lenient with MagicMock, but BuiltinTime instances don't allow
      dynamically adding methods (AttributeError when accessing to_msg).

    Solution: Create a wrapper class that inherits from BuiltinTime and adds to_msg().

    Args:
        instance: Test instance with ros2_connector.node or ros2_connector._node attribute
    """
    from builtin_interfaces.msg import Time as BuiltinTime

    class TimeWithToMsg(BuiltinTime):
        """BuiltinTime wrapper that adds to_msg() method for compatibility."""

        def to_msg(self):
            return self

    mock_clock = MagicMock()
    mock_time = MagicMock()
    mock_ts = TimeWithToMsg()
    mock_time.to_msg.return_value = mock_ts
    mock_clock.now.return_value = mock_time

    # Support both .node and ._node access patterns
    if hasattr(instance, "ros2_connector"):
        if hasattr(instance.ros2_connector, "_node"):
            instance.ros2_connector._node.get_clock = MagicMock(return_value=mock_clock)
        if hasattr(instance.ros2_connector, "node"):
            instance.ros2_connector.node.get_clock = MagicMock(return_value=mock_clock)
