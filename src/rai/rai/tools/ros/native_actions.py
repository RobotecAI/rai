# Copyright (C) 2024 Robotec.AI
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


from typing import Annotated, Any, Dict, Type

import rclpy.node
import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities
from action_msgs.msg import GoalStatus
from langchain.tools import tool
from langchain_core.tools import InjectedToolArg
from rclpy.action.client import ActionClient

from rai.node import RaiBaseNode


@tool
def ros2_get_action_names_and_types(
    node: Annotated[RaiBaseNode, InjectedToolArg]
) -> Dict[str, str]:
    """A tool for getting all ros2 actions names and types"""
    return node.ros_discovery_info.actions_and_types


def ros2_run_action_sync(
    node: Annotated[rclpy.node.Node, InjectedToolArg],
    action_name: str,
    action_type: str,
    action_goal_args: Dict[str, Any],
) -> str:
    """
    A tool for running a ros2 action. Make sure you know the action interface first!!! Actions might take some time to execute and are blocking - you will not be able to check their feedback, only will be informed about the result

    msgs args can have two formats:
    { "goal" : {arg 1 : xyz, ... } or {arg 1 : xyz, ... }
    """

    try:
        msg_cls: Type = rosidl_runtime_py.utilities.get_interface(action_type)
        goal_msg = msg_cls.Goal()

        if "goal" in action_goal_args:
            action_goal_args = action_goal_args["goal"]
        rosidl_runtime_py.set_message.set_message_fields(goal_msg, action_goal_args)

        if action_name[0] != "/":
            action_name = "/" + action_name
            node.get_logger().info(f"Action name corrected to: {action_name}")
    except Exception as e:
        return f"Failed to build message: {e}"

    client = ActionClient(node, msg_cls, action_name)

    retries = 0
    while not client.wait_for_server(timeout_sec=1.0):
        retries += 1
        if retries > 5:
            raise Exception(
                f"Action server '{action_name}' is not available. Make sure `action_name` is correct..."
            )
        node.get_logger().info(
            f"'{action_name}' action server not available, waiting..."
        )

    node.get_logger().info(f"Sending action message: {goal_msg}")
    result = client.send_goal(goal_msg)
    node.get_logger().info("Action finished and result received!")

    if result is not None:
        status = result.status
    else:
        status = GoalStatus.STATUS_UNKNOWN

    if status == GoalStatus.STATUS_SUCCEEDED:
        res = f"Action succeeded, {result.result}"
    elif status == GoalStatus.STATUS_ABORTED:
        res = f"Action aborted, {result.result}"
    elif status == GoalStatus.STATUS_CANCELED:
        res = f"Action canceled: {result.result}"
    else:
        res = "Action failed"

    node.get_logger().info(res)
    return res
