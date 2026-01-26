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

import pytest

try:
    import rclpy  # noqa: F401

    _ = rclpy  # noqa: F841
except ImportError:
    pytest.skip("ROS2 is not installed", allow_module_level=True)


from unittest.mock import MagicMock, patch

from geometry_msgs.msg import TransformStamped
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.navigation.nav2_blocking import (
    GetCurrentPoseTool,
    NavigateToPoseBlockingTool,
    _get_error_code_string,
    _get_error_message,
)

from tests.communication.ros2.helpers import (
    ros_setup,
)

_ = ros_setup  # Explicitly use the fixture to prevent pytest warnings


def test_get_current_pose_tool(ros_setup: None, request: pytest.FixtureRequest) -> None:
    connector = MagicMock(spec=ROS2Connector)
    tool = GetCurrentPoseTool(connector=connector)

    # Mock get_transform return value
    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.child_frame_id = "base_link"
    transform.transform.translation.x = 1.0
    transform.transform.translation.y = 2.0
    transform.transform.translation.z = 0.0
    transform.transform.rotation.w = 1.0
    connector.get_transform.return_value = transform

    result = tool._run()

    connector.get_transform.assert_called_once_with("map", "base_link")
    assert result == str(transform)


def test_get_error_code_string() -> None:
    """Test error code to string conversion."""
    assert _get_error_code_string(0) == "NONE"
    assert _get_error_code_string(2) == "TIMEOUT"
    assert _get_error_code_string(6) == "PLANNER_FAILED"
    assert _get_error_code_string(99) == "UNKNOWN_ERROR_CODE_99"


def test_get_error_message() -> None:
    """Test error message extraction with error code and optional message."""
    # Test with error code only
    result = MagicMock()
    result.error_code = 2
    error_msg = _get_error_message(result)
    assert "(TIMEOUT)" in error_msg

    # Test with error message field
    result_with_msg = MagicMock()
    result_with_msg.error_code = 6
    result_with_msg.error_message = "Planner failed to find path"
    error_msg = _get_error_message(result_with_msg)
    assert "(PLANNER_FAILED)" in error_msg
    assert "Error message: Planner failed to find path" in error_msg


def _create_mock_connector() -> MagicMock:
    """Helper to create a mock connector with node and clock."""
    connector = MagicMock(spec=ROS2Connector)
    mock_node = MagicMock()
    mock_clock = MagicMock()
    mock_time = MagicMock()
    mock_clock.now.return_value.to_msg.return_value = mock_time
    connector.node = mock_node
    mock_node.get_clock.return_value = mock_clock
    return connector


def test_navigate_to_pose_blocking_tool(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    """Test navigation tool success, error, and None result scenarios."""
    connector = _create_mock_connector()
    tool = NavigateToPoseBlockingTool(connector=connector)

    with patch(
        "rai.tools.ros2.navigation.nav2_blocking.ActionClient"
    ) as mock_action_client_class:
        mock_action_client = MagicMock()
        mock_action_client_class.return_value = mock_action_client

        # Test success
        success_result = MagicMock()
        success_result.error_code = 0
        mock_action_client.send_goal.return_value = success_result
        result = tool._run(x=1.0, y=2.0, z=0.0, yaw=0.5)
        assert result == "Navigate to pose successful."

        # Test error with error code
        error_result = MagicMock()
        error_result.error_code = 2
        mock_action_client.send_goal.return_value = error_result
        result = tool._run(x=1.0, y=2.0, z=0.0, yaw=0.5)
        assert "Navigate to pose action failed" in result
        assert "Error code: 2" in result
        assert "(TIMEOUT)" in result

        # Test error with message
        error_result_with_msg = MagicMock()
        error_result_with_msg.error_code = 6
        error_result_with_msg.error_message = "Planner failed"
        mock_action_client.send_goal.return_value = error_result_with_msg
        result = tool._run(x=1.0, y=2.0, z=0.0, yaw=0.5)
        assert "Error code: 6" in result
        assert "(PLANNER_FAILED)" in result
        assert "Error message: Planner failed" in result

        # Test None result
        mock_action_client.send_goal.return_value = None
        result = tool._run(x=1.0, y=2.0, z=0.0, yaw=0.5)
        assert result == "Navigate to pose action failed. Please try again."

        assert mock_action_client.send_goal.call_count == 4
