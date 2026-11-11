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

"""Generic action tools. Goals live in the toolkit's GoalRegistry, so tools of
one toolkit share them and separate toolkits (agents, robots) do not."""

from typing import Any, Dict, List, Mapping, Type

from langchain_core.tools import BaseTool
from langchain_core.utils import stringify_dict
from pydantic import BaseModel, Field

from rai.communication.ros2.runtime import message_class
from rai.tools.base import RaiTool
from rai.tools.hooks import Effect
from rai.tools.ros2.base import (
    BaseROS2Tool,
    BaseROS2Toolkit,
    GoalRegistry,
    from_dict,
    to_dict,
)


class ROS2ActionToolkit(BaseROS2Toolkit):
    name: str = "ros2_action"
    description: str = "A toolkit for ROS2 actions"

    goals: GoalRegistry = Field(default_factory=GoalRegistry, exclude=True)

    def get_tools(self) -> List[BaseTool]:
        return [
            *self._build(StartROS2ActionTool, goals=self.goals),
            CancelROS2ActionTool(goals=self.goals),
            GetROS2ActionFeedbackTool(goals=self.goals),
            GetROS2ActionResultTool(goals=self.goals),
            GetROS2ActionIDsTool(goals=self.goals),
            *self._build(GetROS2ActionsNamesAndTypesTool),
        ]


class GetROS2ActionsNamesAndTypesTool(BaseROS2Tool):
    """Lists the actions the tool may start (all but forbidden, or the writable ones)."""

    default_effect = Effect.READ
    name: str = "get_ros2_actions_names_and_types"
    description: str = "Get the names and types of all ROS2 actions"

    async def _acall(self) -> str:
        return "\n".join(
            stringify_dict({"action": action, "type": types})
            for action, types in self.connector.get_actions_names_and_types()
            if self.is_writable(action)
        )


class StartROS2ActionToolInput(BaseModel):
    action_name: str = Field(
        ..., min_length=1, description="The name of the action to start"
    )
    action_type: str = Field(..., min_length=1, description="The type of the action")
    action_args: Dict[str, Any] = Field(
        ..., description="The arguments to pass to the action"
    )


class StartROS2ActionTool(BaseROS2Tool):
    default_effect = Effect.ACTUATE
    target_args = {"action_name": "action"}
    name: str = "start_ros2_action"
    description: str = "Start a ROS2 action"
    args_schema: Type[StartROS2ActionToolInput] = StartROS2ActionToolInput

    goals: GoalRegistry = Field(default_factory=GoalRegistry, exclude=True)

    async def _acall(
        self, action_name: str, action_type: str, action_args: Dict[str, Any]
    ) -> str:
        self.check_writable(action_name, "action")
        action_cls = message_class(action_type)
        handle = await self.connector.send_goal(
            action_name, from_dict(action_cls.Goal, action_args), action_type=action_cls
        )
        return "Action started with ID: " + self.goals.add(handle)


class _GoalTool(RaiTool):
    """A tool on the goals of a GoalRegistry; its target is the goal's action."""

    goals: GoalRegistry = Field(default_factory=GoalRegistry, exclude=True)

    def call_targets(self, kwargs: Mapping[str, Any]) -> tuple[str, ...]:
        try:
            return (self.goals.get(kwargs["action_id"]).action,)
        except (KeyError, ValueError):
            return ()


class ActionIDInput(BaseModel):
    action_id: str = Field(..., description="The ID of the action")


class GetROS2ActionFeedbackTool(_GoalTool):
    default_effect = Effect.READ
    name: str = "get_ros2_action_feedback"
    description: str = "Get the feedback of a ROS2 action received since the last call, by its action ID"
    args_schema: Type[ActionIDInput] = ActionIDInput

    async def _acall(self, action_id: str) -> str:
        return str([to_dict(f) for f in self.goals.get(action_id).feedback()])


class GetROS2ActionResultTool(_GoalTool):
    default_effect = Effect.READ
    name: str = "get_ros2_action_result"
    description: str = "Get the result of a ROS2 action by its id"
    args_schema: Type[ActionIDInput] = ActionIDInput

    async def _acall(self, action_id: str) -> str:
        handle = self.goals.get(action_id)
        if not handle.done:
            return f"Action {action_id} is not done yet."
        result = await handle.result(0)
        return str({"status": handle.status.name, "result": to_dict(result)})


class CancelROS2ActionTool(_GoalTool):
    """Cancelling stops motion, so it is a WRITE, never denied as ACTUATE."""

    default_effect = Effect.WRITE
    name: str = "cancel_ros2_action"
    description: str = "Cancel a ROS2 action"
    args_schema: Type[ActionIDInput] = ActionIDInput

    async def _acall(self, action_id: str) -> str:
        handle = self.goals.get(action_id)
        if await handle.cancel():
            return f"Action {action_id} cancelled."
        return (
            f"Action {action_id} could not be cancelled (status: {handle.status.name})."
        )


class GetROS2ActionIDsTool(_GoalTool):
    default_effect = Effect.READ
    name: str = "get_ros2_action_ids"
    description: str = "Get the IDs of all ROS2 actions"

    async def _acall(self) -> str:
        return str(self.goals.ids())
