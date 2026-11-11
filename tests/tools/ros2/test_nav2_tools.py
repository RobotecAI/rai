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
    import rclpy

    _ = rclpy
except ImportError:
    pytest.skip("ROS2 is not installed", allow_module_level=True)


import time
from unittest.mock import AsyncMock, MagicMock

from geometry_msgs.msg import TransformStamped
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.navigation.nav2 import Nav2Toolkit
from rai.tools.ros2.navigation.nav2_blocking import (
    GetCurrentPoseTool,
    NavigateToPoseBlockingTool,
)

from tests.communication.ros2.helpers import (
    TestActionServer,
    multi_threaded_spinner,
    ros_setup,
    shutdown_executors_and_threads,
)

_ = ros_setup  # Explicitly use the fixture to prevent pytest warnings


@pytest.fixture
def nav2_server(ros_setup: None, request: pytest.FixtureRequest):
    """A NavigateToPose server that sends 10 feedbacks over 0.1 s, then succeeds."""
    action_name = f"/{request.node.originalname}_navigate"  # type: ignore
    server = TestActionServer(action_name=action_name)
    executors, threads = multi_threaded_spinner([server])
    connector = ROS2Connector()
    try:
        yield connector, server, action_name
    finally:
        connector.close()
        shutdown_executors_and_threads(executors, threads)


def test_get_current_pose_tool() -> None:
    connector = MagicMock(spec=ROS2Connector)
    transform = TransformStamped()
    transform.header.frame_id = "map"
    transform.transform.translation.x = 1.0
    connector.transform = AsyncMock(return_value=transform)
    tool = GetCurrentPoseTool(connector=connector)

    assert tool._run() == str(transform)
    connector.transform.assert_awaited_once_with("map", "base_link")


def test_navigate_to_pose_blocking_tool_succeeds(nav2_server) -> None:
    connector, _, action_name = nav2_server
    tool = NavigateToPoseBlockingTool(connector=connector, action_name=action_name)
    assert tool._run(x=1.0, y=2.0, z=0.0, yaw=0.5) == "Navigate to pose successful."


def test_navigate_to_pose_blocking_tool_cancels_on_timeout(nav2_server) -> None:
    connector, server, action_name = nav2_server
    tool = NavigateToPoseBlockingTool(
        connector=connector, action_name=action_name, timeout_sec=0.02
    )
    result = tool._run(x=1.0, y=2.0, z=0.0, yaw=0.5)
    assert result.startswith("Navigate to pose timed out after 0.02s")
    deadline = time.monotonic() + 5
    while not server.cancelled and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.cancelled


def test_nav2_toolkit_tracks_its_newest_goal(nav2_server) -> None:
    connector, _, action_name = nav2_server
    tools = {
        t.name: t
        for t in Nav2Toolkit(connector=connector, action_name=action_name).get_tools()
    }
    assert tools["get_navigate_to_pose_result"]._run() == "No navigation goal"
    assert tools["cancel_navigate_to_pose"]._run() == "No action to cancel"

    assert tools["navigate_to_pose"]._run(x=0.0, y=0.0, z=0.0, yaw=0.0) == (
        "Navigating to pose"
    )
    deadline = time.monotonic() + 10
    while (result := tools["get_navigate_to_pose_result"]._run()) == (
        "Action is not done yet"
    ):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    assert "'status': 'SUCCEEDED'" in result
    assert "distance_remaining" in tools["get_navigate_to_pose_feedback"]._run()

    # A second toolkit has its own goals
    other = Nav2Toolkit(connector=connector, action_name=action_name).get_tools()
    assert other[3]._run() == "No navigation goal"


def test_nav2_lazy_import() -> None:
    """rai.tools.ros2 imports without nav2_msgs; Nav2 tools import it on use."""
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['nav2_msgs'] = None\n"
        "import rai.tools.ros2, rai.tools.ros2.navigation\n"
        "assert 'nav2_msgs.action' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
