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

import pytest

try:
    import rclpy  # noqa: F401
except ImportError:
    pytest.skip("ROS2 is not installed", allow_module_level=True)


import time

from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2 import (
    CancelROS2ActionTool,
    GetROS2ActionsNamesAndTypesTool,
    StartROS2ActionTool,
)

from tests.communication.ros2.helpers import (
    TestActionServer,
    multi_threaded_spinner,
    ros_setup,
    shutdown_executors_and_threads,
)

_ = ros_setup  # Explicitly use the fixture to prevent pytest warnings


def test_action_call_tool(ros_setup: None, request: pytest.FixtureRequest) -> None:
    action_name = f"{request.node.originalname}_action"  # type: ignore
    connector = ROS2Connector()
    server = TestActionServer(action_name=action_name)
    executors, threads = multi_threaded_spinner([server])
    tool = StartROS2ActionTool(connector=connector)
    try:
        response = tool._run(  # type: ignore
            action_name=action_name,
            action_type="nav2_msgs/action/NavigateToPose",
            action_args={},
        )
        assert "Action started with ID:" in response

    finally:
        shutdown_executors_and_threads(executors, threads)


def test_action_call_tool_with_forbidden_action(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    action_name = f"{request.node.originalname}_action"  # type: ignore
    connector = ROS2Connector()
    tool = StartROS2ActionTool(connector=connector, forbidden=[action_name])
    with pytest.raises(ValueError):
        tool._run(  # type: ignore
            action_name=action_name,
            action_type="nav2_msgs/action/NavigateToPose",
            action_args={},
        )


def test_action_call_tool_with_writable_action(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    action_name = f"{request.node.originalname}_action"  # type: ignore
    connector = ROS2Connector()
    server = TestActionServer(action_name=action_name)
    executors, threads = multi_threaded_spinner([server])
    tool = StartROS2ActionTool(connector=connector, writable=[action_name])
    try:
        response = tool._run(  # type: ignore
            action_name=action_name,
            action_type="nav2_msgs/action/NavigateToPose",
            action_args={},
        )
        assert "Action started with ID:" in response

    finally:
        shutdown_executors_and_threads(executors, threads)


def test_cancel_action_tool(ros_setup: None, request: pytest.FixtureRequest) -> None:
    action_name = f"{request.node.originalname}_action"  # type: ignore
    connector = ROS2Connector()
    server = TestActionServer(action_name=action_name)
    executors, threads = multi_threaded_spinner([server])
    start_tool = StartROS2ActionTool(connector=connector)
    cancel_tool = CancelROS2ActionTool(connector=connector)
    try:
        response = start_tool._run(  # type: ignore
            action_name=action_name,
            action_type="nav2_msgs/action/NavigateToPose",
            action_args={},
        )
        action_id = response.split("Action started with ID:")[1].strip()

        response = cancel_tool._run(  # type: ignore
            action_id=action_id,
        )
        assert server.cancelled

    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_actions_names_and_types_tool_no_restrictions(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    action_name = f"{request.node.originalname}_action"  # type: ignore
    connector = ROS2Connector()
    server = TestActionServer(action_name=action_name)
    executors, threads = multi_threaded_spinner([server])
    try:
        time.sleep(0.2)
        tool = GetROS2ActionsNamesAndTypesTool(connector=connector)
        response = tool._run()  # type: ignore
        assert response != ""
        assert action_name in response
        assert "action:" in response
        assert "type:" in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_actions_names_and_types_tool_with_writable(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    writable_action = f"/{request.node.originalname}_writable_action"  # type: ignore
    non_writable_action = f"/{request.node.originalname}_non_writable_action"  # type: ignore
    connector = ROS2Connector()
    server1 = TestActionServer(action_name=writable_action)
    server2 = TestActionServer(action_name=non_writable_action)
    executors, threads = multi_threaded_spinner([server1, server2])
    try:
        time.sleep(0.2)
        tool = GetROS2ActionsNamesAndTypesTool(
            connector=connector, writable=[writable_action]
        )
        response = tool._run()  # type: ignore
        assert response != ""
        assert writable_action in response
        assert non_writable_action not in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_actions_names_and_types_tool_with_forbidden(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    action_name = f"/{request.node.originalname}_action"  # type: ignore
    forbidden_action = f"/{request.node.originalname}_forbidden_action"  # type: ignore
    connector = ROS2Connector()
    server1 = TestActionServer(action_name=action_name)
    server2 = TestActionServer(action_name=forbidden_action)
    executors, threads = multi_threaded_spinner([server1, server2])
    try:
        time.sleep(0.2)
        tool = GetROS2ActionsNamesAndTypesTool(
            connector=connector, forbidden=[forbidden_action]
        )
        response = tool._run()  # type: ignore
        assert response != ""
        assert action_name in response
        assert forbidden_action not in response
    finally:
        shutdown_executors_and_threads(executors, threads)


def test_get_actions_names_and_types_tool_with_writable_and_forbidden(
    ros_setup: None, request: pytest.FixtureRequest
) -> None:
    writable_action = f"/{request.node.originalname}_writable_action"  # type: ignore
    forbidden_action = f"/{request.node.originalname}_forbidden_action"  # type: ignore
    connector = ROS2Connector()
    server1 = TestActionServer(action_name=writable_action)
    server2 = TestActionServer(action_name=forbidden_action)
    executors, threads = multi_threaded_spinner([server1, server2])
    try:
        time.sleep(0.2)
        tool = GetROS2ActionsNamesAndTypesTool(
            connector=connector,
            writable=[writable_action],
            forbidden=[forbidden_action],
        )
        response = tool._run()  # type: ignore
        assert response != ""
        assert writable_action in response
        assert forbidden_action not in response
    finally:
        shutdown_executors_and_threads(executors, threads)
