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
    _get_error_message,
    _get_status_string,
)

from tests.communication.ros2.helpers import (
    create_mock_connector_with_clock,
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


def test_get_status_string() -> None:
    """Test GoalStatus to string conversion."""
    from action_msgs.msg import GoalStatus

    assert _get_status_string(GoalStatus.STATUS_UNKNOWN) == "UNKNOWN"
    assert _get_status_string(GoalStatus.STATUS_SUCCEEDED) == "SUCCEEDED"
    assert _get_status_string(GoalStatus.STATUS_ABORTED) == "ABORTED"
    assert _get_status_string(GoalStatus.STATUS_CANCELED) == "CANCELED"
    assert _get_status_string(999) == "UNKNOWN_STATUS_999"


def test_get_error_message() -> None:
    """Test error message extraction with status and error_msg."""
    from action_msgs.msg import GoalStatus

    # Test with ABORTED status and no error message
    result_response = MagicMock()
    result_response.status = GoalStatus.STATUS_ABORTED
    result_response.result = MagicMock()
    result_response.result.error_msg = ""
    error_msg = _get_error_message(result_response)
    assert error_msg == "Status: ABORTED"

    # Test with ABORTED status and error message
    result_with_msg = MagicMock()
    result_with_msg.status = GoalStatus.STATUS_ABORTED
    result_with_msg.result = MagicMock()
    result_with_msg.result.error_msg = "Planner failed to find path"
    error_msg = _get_error_message(result_with_msg)
    assert error_msg == "Status: ABORTED. Planner failed to find path"

    # Test with CANCELED status
    canceled_response = MagicMock()
    canceled_response.status = GoalStatus.STATUS_CANCELED
    canceled_response.result = MagicMock()
    canceled_response.result.error_msg = ""
    error_msg = _get_error_message(canceled_response)
    assert error_msg == "Status: CANCELED"


def test_navigate_to_pose_blocking_tool(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    """Test navigation tool success, error, and None result scenarios."""
    from action_msgs.msg import GoalStatus

    connector = create_mock_connector_with_clock()
    tool = NavigateToPoseBlockingTool(connector=connector)

    with patch(
        "rai.tools.ros2.navigation.nav2_blocking.ActionClient"
    ) as mock_action_client_class:
        mock_action_client = MagicMock()
        mock_action_client_class.return_value = mock_action_client

        # Test success
        success_result = MagicMock()
        success_result.status = GoalStatus.STATUS_SUCCEEDED
        success_result.result = MagicMock()
        success_result.result.error_msg = ""
        mock_action_client.send_goal.return_value = success_result
        result = tool._run(x=1.0, y=2.0, z=0.0, yaw=0.5)
        assert result == "Navigate to pose successful."

        # Test error without message
        error_result = MagicMock()
        error_result.status = GoalStatus.STATUS_ABORTED
        error_result.result = MagicMock()
        error_result.result.error_msg = ""
        mock_action_client.send_goal.return_value = error_result
        result = tool._run(x=1.0, y=2.0, z=0.0, yaw=0.5)
        assert result == "Navigate to pose action failed. Status: ABORTED"

        # Test error with message
        error_result_with_msg = MagicMock()
        error_result_with_msg.status = GoalStatus.STATUS_ABORTED
        error_result_with_msg.result = MagicMock()
        error_result_with_msg.result.error_msg = "Planner failed"
        mock_action_client.send_goal.return_value = error_result_with_msg
        result = tool._run(x=1.0, y=2.0, z=0.0, yaw=0.5)
        assert (
            result == "Navigate to pose action failed. Status: ABORTED. Planner failed"
        )

        # Test None result
        mock_action_client.send_goal.return_value = None
        result = tool._run(x=1.0, y=2.0, z=0.0, yaw=0.5)
        assert result == "Navigate to pose action failed. Please try again."

        assert mock_action_client.send_goal.call_count == 4
