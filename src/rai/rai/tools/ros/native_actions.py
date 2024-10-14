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
#

from typing import Any, Dict, Optional, Tuple, Type

import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities
from action_msgs.msg import GoalStatus
from pydantic import BaseModel, Field
from rclpy.action.client import ActionClient

from .native import Ros2BaseInput, Ros2BaseTool


# --------------------- Inputs ---------------------
class Ros2ActionRunnerInput(BaseModel):
    action_name: str = Field(..., description="Name of the action")
    action_type: str = Field(..., description="Type of the action")
    action_goal_args: Dict[str, Any] = Field(
        ..., description="Dictionary with arguments for the action goal message"
    )


class ActionUidInput(BaseModel):
    uid: str = Field(..., description="Action uid.")


class OptionalActionUidInput(BaseModel):
    uid: Optional[str] = Field(
        None,
        description="Optional action uid. If None - results from all submitted actions will be returned.",
    )


# --------------------- Tools ---------------------
class Ros2GetActionNamesAndTypesTool(Ros2BaseTool):
    name: str = "Ros2GetActionNamesAndTypes"
    description: str = "A tool for getting all ros2 actions names and types"

    def _run(self):
        return self.node.ros_discovery_info.actions_and_types


class Ros2BaseActionTool(Ros2BaseTool):
    pass


class Ros2RunActionSync(Ros2BaseTool):
    name: str = "Ros2RunAction"
    description: str = (
        "A tool for running a ros2 action. Make sure you know the action interface first!!! Actions might take some time to execute and are blocking - you will not be able to check their feedback, only will be informed about the result"
    )

    args_schema: Type[Ros2ActionRunnerInput] = Ros2ActionRunnerInput

    def _build_msg(
        self, msg_type: str, msg_args: Dict[str, Any]
    ) -> Tuple[object, Type]:
        """
        Import message and create it. Return both ready message and message class.

        msgs args can have two formats:
        { "goal" : {arg 1 : xyz, ... } or {arg 1 : xyz, ... }
        """

        msg_cls: Type = rosidl_runtime_py.utilities.get_interface(msg_type)
        msg = msg_cls.Goal()

        if "goal" in msg_args:
            msg_args = msg_args["goal"]
        rosidl_runtime_py.set_message.set_message_fields(msg, msg_args)
        return msg, msg_cls

    def _run(
        self, action_name: str, action_type: str, action_goal_args: Dict[str, Any]
    ):
        if action_name[0] != "/":
            action_name = "/" + action_name
            self.node.get_logger().info(f"Action name corrected to: {action_name}")

        try:
            goal_msg, msg_cls = self._build_msg(action_type, action_goal_args)
        except Exception as e:
            return f"Failed to build message: {e}"

        client = ActionClient(self.node, msg_cls, action_name)

        retries = 0
        while not client.wait_for_server(timeout_sec=1.0):
            retries += 1
            if retries > 5:
                raise Exception(
                    f"Action server '{action_name}' is not available. Make sure `action_name` is correct..."
                )
            self.node.get_logger().info(
                f"'{action_name}' action server not available, waiting..."
            )

        self.node.get_logger().info(f"Sending action message: {goal_msg}")
        result = client.send_goal(goal_msg)
        self.node.get_logger().info("Action finished and result received!")

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

        self.node.get_logger().info(res)
        return res


class Ros2RunActionAsync(Ros2BaseActionTool):
    name: str = "Ros2RunAction"
    description: str = """A tool for running a ros2 action.
        <rule>Always check action interface before setting action_goal_args.</rule>"""

    args_schema: Type[Ros2ActionRunnerInput] = Ros2ActionRunnerInput

    def _run(
        self, action_name: str, action_type: str, action_goal_args: Dict[str, Any]
    ):
        return self.node._run_action(action_name, action_type, action_goal_args)


class Ros2IsActionComplete(Ros2BaseActionTool):
    name: str = "Ros2IsActionComplete"
    description: str = "A tool for checking if submitted ros2 actions is complete"

    args_schema: Type[Ros2BaseInput] = Ros2BaseInput

    def _run(self) -> bool:
        return self.node._is_task_complete()


class Ros2GetActionResult(Ros2BaseActionTool):
    name: str = "Ros2GetActionResult"
    description: str = "A tool for checking the result of submitted ros2 action"

    args_schema: Type[Ros2BaseInput] = Ros2BaseInput

    def _run(self) -> bool:
        return self.node._get_task_result()


class Ros2CancelAction(Ros2BaseActionTool):
    name: str = "Ros2CancelAction"
    description: str = "Cancel submitted action"

    args_schema: Type[Ros2BaseInput] = Ros2BaseInput

    def _run(self) -> bool:
        return self.node._cancel_task()


class Ros2GetLastActionFeedback(Ros2BaseActionTool):
    name: str = "Ros2GetLastActionFeedback"
    description: str = "Get last action feedback."

    args_schema: Type[Ros2BaseInput] = Ros2BaseInput

    def _run(self) -> str:
        return str(self.node.action_feedback)
