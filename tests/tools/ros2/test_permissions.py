# Copyright (C) 2024 Robotec.AI
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
"""Permission lists must match fully resolved names, so a relative name
like "cmd_vel" cannot bypass forbidden=["/cmd_vel"]."""

import pytest

try:
    import rclpy  # noqa: F401
except ImportError:
    pytest.skip("ROS2 is not installed", allow_module_level=True)

from langchain_core.tools import ToolException
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2 import (
    CallROS2ServiceTool,
    PublishROS2MessageTool,
    ReceiveROS2MessageTool,
    StartROS2ActionTool,
)

from tests.communication.ros2.helpers import ros_setup

_ = ros_setup  # Explicitly use the fixture to prevent pytest warnings


@pytest.fixture
def connector(ros_setup: None):
    connector = ROS2Connector()
    yield connector
    connector.close()


@pytest.mark.parametrize(
    "configured,checked", [("/cmd_vel", "cmd_vel"), ("cmd_vel", "/cmd_vel")]
)
def test_forbidden_matches_resolved_name(connector, configured, checked) -> None:
    tool = PublishROS2MessageTool(connector=connector, forbidden=[configured])
    assert not tool.is_writable(checked)
    assert not tool.is_readable(checked)
    with pytest.raises(ToolException, match="not writable"):
        tool._run(topic=checked, message={}, message_type="std_msgs/msg/String")
    receive = ReceiveROS2MessageTool(connector=connector, forbidden=[configured])
    with pytest.raises(ToolException, match="not readable"):
        receive._run(topic=checked)


def test_whitelists_match_resolved_name(connector) -> None:
    tool = PublishROS2MessageTool(
        connector=connector, readable=["/scan"], writable=["cmd_vel"]
    )
    assert tool.is_readable("scan")
    assert tool.is_writable("/cmd_vel")
    assert not tool.is_writable("other")


def test_forbidden_service_and_action_match_resolved_name(connector) -> None:
    service_tool = CallROS2ServiceTool(connector=connector, forbidden=["/reset"])
    with pytest.raises(ToolException, match="not writable"):
        service_tool._run(service_name="reset", service_type="std_srvs/srv/Empty")
    action_tool = StartROS2ActionTool(connector=connector, forbidden=["/navigate"])
    with pytest.raises(ToolException, match="not writable"):
        action_tool._run(
            action_name="navigate",
            action_type="nav2_msgs/action/NavigateToPose",
            action_args={},
        )
