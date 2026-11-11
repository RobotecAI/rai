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

import asyncio
import contextlib
from typing import Type

from pydantic import Field

from rai.communication.ros2 import GoalStatus
from rai.tools.hooks import Effect
from rai.tools.ros2.base import BaseROS2Tool
from rai.tools.ros2.navigation.bounds import WorkspaceBounds
from rai.tools.ros2.navigation.nav2 import (
    NAVIGATE_TO_POSE,
    NavigateToPoseToolInput,
    navigate_to_pose_goal,
)


class GetCurrentPoseTool(BaseROS2Tool):
    default_effect = Effect.READ
    name: str = "get_current_pose"
    description: str = "Get the current pose of the robot"
    frame_id: str = Field(
        default="map", description="The frame id of the Nav2 stack (map, odom, etc.)"
    )
    robot_frame_id: str = Field(
        default="base_link",
        description="The frame id of the robot's base frame (base_link, base_footprint, etc.)",
    )

    async def _acall(self) -> str:
        return str(await self.connector.transform(self.frame_id, self.robot_frame_id))


NavigateToPoseBlockingToolInput = NavigateToPoseToolInput


class NavigateToPoseBlockingTool(WorkspaceBounds, BaseROS2Tool):
    """Navigate and wait for the result. The goal is cancelled if it does not
    finish within timeout_sec or the call is cancelled."""

    default_effect = Effect.ACTUATE
    target_fields = {"action_name": "action"}
    name: str = "navigate_to_pose_blocking"
    description: str = "Navigate to a specific pose"
    frame_id: str = Field(
        default="map", description="The frame id of the Nav2 stack (map, odom, etc.)"
    )
    action_name: str = Field(
        default="navigate_to_pose", description="The name of the Nav2 action"
    )
    timeout_sec: float = Field(
        default=300.0, gt=0, description="Cancel the goal after this many seconds"
    )
    args_schema: Type[NavigateToPoseBlockingToolInput] = NavigateToPoseBlockingToolInput

    async def _acall(self, x: float, y: float, z: float, yaw: float) -> str:
        self.reject_out_of_bounds(x, y, z)
        goal = navigate_to_pose_goal(self.connector, self.frame_id, x, y, z, yaw)
        handle = await self.connector.send_goal(
            self.action_name, goal, action_type=NAVIGATE_TO_POSE
        )
        try:
            result = await handle.result(self.timeout_sec)
        except BaseException as e:
            if not handle.done:
                with contextlib.suppress(Exception):
                    await asyncio.shield(handle.cancel())
            if isinstance(e, TimeoutError):
                return f"Navigate to pose timed out after {self.timeout_sec}s; the goal was cancelled."
            raise
        if handle.status == GoalStatus.SUCCEEDED:
            return "Navigate to pose successful."
        error = f". {result.error_msg}" if getattr(result, "error_msg", "") else ""
        return f"Navigate to pose action failed. Status: {handle.status.name}{error}"
