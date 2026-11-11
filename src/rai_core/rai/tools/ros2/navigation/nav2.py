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

"""Nav2 tools. nav2_msgs is imported on first use, so these tools import
without Nav2 installed."""

import time
from typing import Any, Dict, List, Tuple, cast

import cv2
import numpy as np
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from langchain_core.tools import BaseTool
from nav_msgs.msg import OccupancyGrid
from pydantic import BaseModel, Field

from rai.communication.ros2 import ROS2Connector, encode_image
from rai.tools.base import RaiTool
from rai.tools.hooks import Effect
from rai.tools.ros2.base import BaseROS2Tool, BaseROS2Toolkit, GoalRegistry, to_dict
from rai.tools.ros2.navigation.bounds import WorkspaceBounds
from rai.tools.ros2.navigation.geometry import quaternion_to_yaw, yaw_to_quaternion

NAVIGATE_TO_POSE = "nav2_msgs/action/NavigateToPose"


def navigate_to_pose_goal(
    connector: ROS2Connector, frame_id: str, x: float, y: float, z: float, yaw: float
) -> Any:
    """A NavigateToPose goal for (x, y, z, yaw) in frame_id, stamped now."""
    from nav2_msgs.action import NavigateToPose

    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = frame_id
    goal.pose.header.stamp = connector.now().to_msg()
    goal.pose.pose.position = Point(x=x, y=y, z=z)
    goal.pose.pose.orientation = yaw_to_quaternion(yaw)
    return goal


class Nav2Toolkit(WorkspaceBounds, BaseROS2Toolkit):
    """Non-blocking navigation: start a goal, then poll, or cancel it. The tools
    share the toolkit's goals; feedback, result and cancel act on the newest."""

    frame_id: str = Field(
        default="map", description="The frame id of the Nav2 stack (map, odom, etc.)"
    )
    action_name: str = Field(
        default="navigate_to_pose", description="The name of the NavigateToPose action"
    )
    goals: GoalRegistry = Field(default_factory=GoalRegistry, exclude=True)

    def get_tools(self) -> List[BaseTool]:
        return [
            NavigateToPoseTool(
                connector=self.connector,
                frame_id=self.frame_id,
                action_name=self.action_name,
                workspace_bounds_min=self.workspace_bounds_min,
                workspace_bounds_max=self.workspace_bounds_max,
                goals=self.goals,
            ),
            CancelNavigateToPoseTool(goals=self.goals),
            GetNavigateToPoseFeedbackTool(goals=self.goals),
            GetNavigateToPoseResultTool(goals=self.goals),
        ]


class NavigateToPoseToolInput(BaseModel):
    x: float = Field(
        ..., allow_inf_nan=False, description="The x coordinate of the pose"
    )
    y: float = Field(
        ..., allow_inf_nan=False, description="The y coordinate of the pose"
    )
    z: float = Field(
        ..., allow_inf_nan=False, description="The z coordinate of the pose"
    )
    yaw: float = Field(
        ..., allow_inf_nan=False, description="The yaw angle of the pose"
    )


class NavigateToPoseTool(WorkspaceBounds, BaseROS2Tool):
    default_effect = Effect.ACTUATE
    target_fields = {"action_name": "action"}
    name: str = "navigate_to_pose"
    description: str = "Navigate to a specific pose"

    args_schema: type[NavigateToPoseToolInput] = NavigateToPoseToolInput

    frame_id: str = Field(
        default="map", description="The frame id of the Nav2 stack (map, odom, etc.)"
    )
    action_name: str = Field(
        default="navigate_to_pose", description="The name of the NavigateToPose action"
    )
    goals: GoalRegistry = Field(default_factory=GoalRegistry, exclude=True)

    async def _acall(self, x: float, y: float, z: float, yaw: float) -> str:
        self.reject_out_of_bounds(x, y, z)
        goal = navigate_to_pose_goal(self.connector, self.frame_id, x, y, z, yaw)
        self.goals.add(
            await self.connector.send_goal(
                self.action_name, goal, action_type=NAVIGATE_TO_POSE
            )
        )
        return "Navigating to pose"


class _CurrentGoalTool(RaiTool):
    goals: GoalRegistry = Field(default_factory=GoalRegistry, exclude=True)

    def call_targets(self, kwargs: Any) -> Tuple[str, ...]:
        handle = self.goals.latest()
        return (handle.action,) if handle else ()


class GetNavigateToPoseFeedbackTool(_CurrentGoalTool):
    default_effect = Effect.READ
    name: str = "get_navigate_to_pose_feedback"
    description: str = "Get the latest feedback of the navigate to pose action"

    async def _acall(self) -> str:
        handle = self.goals.latest()
        if handle is None:
            return "No navigation goal"
        feedback = handle.feedback()
        return str(to_dict(feedback[-1])) if feedback else "No new feedback"


class GetNavigateToPoseResultTool(_CurrentGoalTool):
    default_effect = Effect.READ
    name: str = "get_navigate_to_pose_result"
    description: str = "Get the result of the navigate to pose action"

    async def _acall(self) -> str:
        handle = self.goals.latest()
        if handle is None:
            return "No navigation goal"
        if not handle.done:
            return "Action is not done yet"
        result = await handle.result(0)
        return str({"status": handle.status.name, "result": to_dict(result)})


class CancelNavigateToPoseTool(_CurrentGoalTool):
    default_effect = Effect.WRITE
    name: str = "cancel_navigate_to_pose"
    description: str = "Cancel the navigate to pose action"

    async def _acall(self) -> str:
        handle = self.goals.latest()
        if handle is None or handle.done:
            return "No action to cancel"
        if await handle.cancel():
            return "Action cancelled"
        return f"Action could not be cancelled (status: {handle.status.name})"


class GetOccupancyGridTool(BaseROS2Tool):
    """Get the current map as an image with the robot's position marked on it (red arrow)."""

    default_effect = Effect.READ
    target_fields = {"map_topic": "topic"}
    name: str = "GetOccupancyGridTool"
    description: str = "A tool for getting the current map as an image with the robot's position marked on it."

    response_format: str = "content_and_artifact"
    map_topic: str = "/map"
    map_frame: str = "map"
    robot_frame: str = "base_link"
    image_width: int = 1500
    debug: bool = False

    def _postprocess_msg(self, map_msg: OccupancyGrid, transform: TransformStamped):
        width = cast(int, map_msg.info.width)
        height = cast(int, map_msg.info.height)
        resolution = cast(float, map_msg.info.resolution)
        origin_position = cast(Point, map_msg.info.origin.position)
        origin_orientation = cast(Quaternion, map_msg.info.origin.orientation)

        data = np.array(map_msg.data).reshape((height, width))

        # Convert the OccupancyGrid values to grayscale image (0-255)
        # the final image shape is (self.image_width, self.image_width), scale to fit
        scale = self.image_width / max(width, height)
        width = int(width * scale)
        height = int(height * scale)
        data = cv2.resize(data, (width, height), interpolation=cv2.INTER_NEAREST)
        resolution = resolution / scale
        image = np.zeros_like(data, dtype=np.uint8)
        image[data == -1] = 127  # Unknown space
        image[data == 0] = 255  # Free space
        image[data > 0] = 0  # Occupied space

        # Calculate robot's position in the image
        robot_x = cast(
            float, (transform.transform.translation.x - origin_position.x) / resolution
        )

        robot_y = cast(
            float, (transform.transform.translation.y - origin_position.y) / resolution
        )

        yaw = quaternion_to_yaw(origin_orientation)
        # Rotate the robot's position based on the yaw angle
        rotated_x = robot_x * np.cos(yaw) - robot_y * np.sin(yaw)
        rotated_y = robot_x * np.sin(yaw) + robot_y * np.cos(yaw)
        robot_x = int(rotated_x)
        robot_y = int(rotated_y)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Draw the robot's position as an arrow
        if 0 <= robot_x < width and 0 <= robot_y < height:
            yaw = quaternion_to_yaw(transform.transform.rotation)
            arrow_length = 100
            arrow_end_x = int(robot_x + arrow_length * np.cos(yaw))
            arrow_end_y = int(robot_y + arrow_length * np.sin(yaw))
            cv2.arrowedLine(
                image, (robot_x, robot_y), (arrow_end_x, arrow_end_y), (255, 0, 0), 5
            )

        image = cv2.flip(image, 1)

        step_size_m: float = 2.0  # Step size for grid lines in meters, adjust as needed
        step_size_pixels = int(step_size_m / resolution)
        # print(step_size_pixels, scale)
        for x in range(0, width, step_size_pixels):
            cv2.line(
                img=image,
                pt1=(x, 40),
                pt2=(x, height),
                color=(200, 200, 200),
                thickness=1,
            )
            cv2.putText(
                img=image,
                text=f"{x * resolution + origin_position.x:.1f}",
                org=(x, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5,
                color=(255, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        for y in range(0, height, step_size_pixels):
            cv2.line(
                img=image,
                pt1=(0, y),
                pt2=(width, y),
                color=(200, 200, 200),
                thickness=1,
            )
            cv2.putText(
                img=image,
                text=f"{y * resolution + origin_position.y:.1f}",
                org=(15, y + 35),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5,
                color=(255, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        if self.debug:
            cv2.imwrite(f"map{time.time()}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return encode_image(image, format="png", max_side=self.image_width)

    async def _acall(self) -> Tuple[str, Dict[str, List[str]]]:
        """Gets the current map from the map topic."""
        map_msg = await self.connector.receive(self.map_topic, timeout=10)
        transform = await self.connector.transform(
            self.map_frame, self.robot_frame, timeout=10
        )
        image = self._postprocess_msg(map_msg, transform)
        return "Map grabbed successfully", {"images": [image]}
