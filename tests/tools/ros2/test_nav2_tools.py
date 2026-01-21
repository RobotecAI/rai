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


from unittest.mock import MagicMock

from geometry_msgs.msg import TransformStamped
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.navigation.nav2_blocking import (
    GetCurrentPoseTool,
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
