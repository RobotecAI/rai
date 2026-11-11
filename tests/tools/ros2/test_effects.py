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

"""Every ROS 2 tool declares its effect and resolved targets, and hooks see them."""

import time

import pytest

try:
    import rclpy.executors
except ImportError:
    pytest.skip("ROS2 is not installed", allow_module_level=True)

from langchain_core.tools import BaseTool, ToolException
from rai.communication.ros2 import ROS2Connector
from rai.tools import Allow, Deny, Effect, ToolCallContext, global_tool_hooks
from rai.tools.ros2 import (
    ROS2CLIToolkit,
    ROS2Toolkit,
    cli,
)
from rai.tools.ros2.manipulation import (
    GetObjectPositionsTool,
    MoveObjectFromToTool,
    MoveToPointTool,
    ResetArmTool,
)
from rai.tools.ros2.navigation import (
    GetOccupancyGridTool,
    Nav2Toolkit,
    NavigateToPoseBlockingTool,
)
from rai.tools.ros2.navigation.nav2_blocking import GetCurrentPoseTool
from rai.tools.ros2.simple import (
    GetROS2ImageConfiguredTool,
    GetROS2TransformConfiguredTool,
)

from tests.communication.ros2.helpers import MessageSubscriber, ros_setup

_ = ros_setup  # Explicitly use the fixture to prevent pytest warnings

R, W, A = Effect.READ, Effect.WRITE, Effect.ACTUATE
MANIPULATOR = ("/manipulator_move_to",)

# tool name -> (effect, static targets, call kwargs, targets the hooks see)
EXPECTED = {
    "publish_ros2_message": (A, (), {"topic": "cmd_vel"}, ("/cmd_vel",)),
    "receive_ros2_message": (R, (), {"topic": "scan"}, ("/scan",)),
    "get_ros2_image": (R, (), {"topic": "camera"}, ("/camera",)),
    "get_ros2_transform": (R, (), {}, ()),
    "get_ros2_topics_names_and_types": (R, (), {}, ()),
    "get_ros2_message_interface": (R, (), {}, ()),
    "call_ros2_service": (W, (), {"service_name": "reset"}, ("/reset",)),
    "get_ros2_services_names_and_types": (R, (), {}, ()),
    "start_ros2_action": (A, (), {"action_name": "nav"}, ("/nav",)),
    "cancel_ros2_action": (W, (), {"action_id": "x"}, ()),
    "get_ros2_action_feedback": (R, (), {"action_id": "x"}, ()),
    "get_ros2_action_result": (R, (), {"action_id": "x"}, ()),
    "get_ros2_action_ids": (R, (), {}, ()),
    "get_ros2_actions_names_and_types": (R, (), {}, ()),
    "navigate_to_pose": (A, ("/navigate_to_pose",), {}, ("/navigate_to_pose",)),
    "cancel_navigate_to_pose": (W, (), {}, ()),
    "get_navigate_to_pose_feedback": (R, (), {}, ()),
    "get_navigate_to_pose_result": (R, (), {}, ()),
    "navigate_to_pose_blocking": (
        A,
        ("/navigate_to_pose",),
        {},
        ("/navigate_to_pose",),
    ),
    "get_current_pose": (R, (), {}, ()),
    "GetOccupancyGridTool": (R, ("/map",), {}, ("/map",)),
    "move_to_point": (A, MANIPULATOR, {}, MANIPULATOR),
    "move_object_from_to": (A, MANIPULATOR, {}, MANIPULATOR),
    "reset_arm": (A, MANIPULATOR, {}, MANIPULATOR),
    "get_object_positions": (R, (), {}, ()),
    "get_ros2_camera_image": (R, ("/camera",), {}, ("/camera",)),
    "get_ros2_robot_position": (R, (), {}, ()),
    **{t.name: (R, (), {}, ()) for t in ROS2CLIToolkit().get_tools()},
}


@pytest.fixture(scope="module")
def tools(ros_setup: None):
    connector = ROS2Connector()
    frame = {"connector": connector, "manipulator_frame": "base_link"}
    all_tools: list[BaseTool] = [
        *ROS2Toolkit(connector=connector).get_tools(),
        *Nav2Toolkit(connector=connector).get_tools(),
        NavigateToPoseBlockingTool(connector=connector),
        GetCurrentPoseTool(connector=connector),
        GetOccupancyGridTool(connector=connector),
        MoveToPointTool(**frame),
        MoveObjectFromToTool(**frame),
        ResetArmTool(**frame),
        GetObjectPositionsTool(
            connector=connector,
            target_frame="a",
            source_frame="b",
            camera_topic="c",
            depth_topic="d",
            camera_info_topic="e",
            get_grabbing_point_tool=cli.ros2_node,
        ),
        GetROS2ImageConfiguredTool(connector=connector, topic="camera"),
        GetROS2TransformConfiguredTool(
            connector=connector, source_frame="a", target_frame="b"
        ),
        *ROS2CLIToolkit().get_tools(),
    ]
    yield {tool.name: tool for tool in all_tools}
    connector.close()


def test_every_tool_is_in_the_table(tools) -> None:
    assert sorted(tools) == sorted(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_effect_and_targets(tools, name) -> None:
    effect, targets, kwargs, call_targets = EXPECTED[name]
    tool = tools[name]
    assert (tool.effect, tool.targets) == (effect, targets)
    assert tool.call_targets(kwargs) == call_targets


@pytest.fixture
def deny_actuation():
    seen: list[ToolCallContext] = []

    async def no_actuation(call: ToolCallContext) -> Allow | Deny:
        seen.append(call)
        return Deny("robot is parked") if call.effect is Effect.ACTUATE else Allow()

    global_tool_hooks.add(no_actuation)
    yield seen
    global_tool_hooks.remove(no_actuation)


def test_hook_denies_actuation_before_publishing(
    ros_setup: None, request: pytest.FixtureRequest, deny_actuation
) -> None:
    topic = f"{request.node.originalname}_cmd"  # type: ignore
    receiver = MessageSubscriber(topic=topic)
    connector = ROS2Connector()
    tools = {t.name: t for t in ROS2Toolkit(connector=connector).get_tools()}
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(receiver)
    try:
        with pytest.raises(ToolException, match="robot is parked"):
            tools["publish_ros2_message"]._run(
                topic=topic, message={"data": "go"}, message_type="std_msgs/msg/String"
            )
        # The hook saw the resolved target; READ tools still run.
        assert deny_actuation[-1].targets == (f"/{topic}",)
        assert tools["get_ros2_action_ids"]._run() == "[]"
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)
        assert receiver.received_messages == []
    finally:
        executor.shutdown()
        receiver.destroy_node()
        connector.close()
