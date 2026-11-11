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

"""Manipulation tools on the rai_interfaces ManipulatorMoveTo service."""

import asyncio
import contextlib
from typing import Any, Literal, Type

from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from tf2_geometry_msgs import do_transform_pose
from typing_extensions import deprecated

from rai.tools.hooks import Effect
from rai.tools.ros2.base import BaseROS2Tool

GRIPPER_DOWN = Quaternion(x=0.9238795325112867, y=-0.3826834323650898, z=0.0, w=0.0)


class ManipulatorTool(BaseROS2Tool):
    """Moves the end effector with the ManipulatorMoveTo service."""

    default_effect = Effect.ACTUATE
    target_fields = {"service": "service"}

    manipulator_frame: str = Field(..., description="Manipulator frame")
    service: str = Field(
        default="/manipulator_move_to", description="The ManipulatorMoveTo service"
    )
    timeout_sec: float = Field(
        default=20.0, description="Timeout for service calls in seconds"
    )
    quaternion: Quaternion = Field(
        default=GRIPPER_DOWN, description="Constant end effector orientation"
    )

    async def move(self, x: float, y: float, z: float, grab: bool) -> bool:
        """Move to (x, y, z). grab: open the gripper before and close it after
        the move; otherwise close before and open after. True on success."""
        from rai_interfaces.srv import ManipulatorMoveTo

        request = ManipulatorMoveTo.Request()
        request.target_pose = PoseStamped()
        request.target_pose.header.frame_id = self.manipulator_frame
        request.target_pose.pose = Pose(
            position=Point(x=float(x), y=float(y), z=float(z)),
            orientation=self.quaternion,
        )
        request.initial_gripper_state = grab  # True: open
        request.final_gripper_state = not grab
        response = await self.connector.call(
            self.service, request, timeout=self.timeout_sec
        )
        return response.success


class CalibratedManipulatorTool(ManipulatorTool):
    min_z: float = Field(default=0.135, description="Minimum z coordinate [m]")
    calibration_x: float = Field(default=0.0, description="Calibration x [m]")
    calibration_y: float = Field(default=0.0, description="Calibration y [m]")
    calibration_z: float = Field(default=0.0, description="Calibration z [m]")
    additional_height: float = Field(
        default=0.05, description="Additional height for the place task [m]"
    )

    async def move_calibrated(
        self, x: float, y: float, z: float, grab: bool, lift: float = 0.0
    ) -> bool:
        return await self.move(
            x + self.calibration_x,
            y + self.calibration_y,
            max(z + lift + self.calibration_z, self.min_z),
            grab,
        )


class MoveToPointToolInput(BaseModel):
    x: float = Field(description="The x coordinate of the point to move to")
    y: float = Field(description="The y coordinate of the point to move to")
    z: float = Field(description="The z coordinate of the point to move to")
    task: Literal["grab", "drop"] = Field(
        description="Specify the intended action: use 'grab' to pick up an object, or 'drop' to release it. "
        "This determines the gripper's behavior during the movement."
    )


class MoveToPointTool(CalibratedManipulatorTool):
    name: str = "move_to_point"
    description: str = (
        "Guide the robot's end effector to a specific point within the manipulator's operational space. "
        "This tool ensures precise movement to the desired location. "
        "While it confirms successful positioning, please note that it doesn't provide feedback on the "
        "success of grabbing or releasing objects. Use additional sensors or tools for that information."
    )
    args_schema: Type[MoveToPointToolInput] = MoveToPointToolInput

    async def _acall(
        self, x: float, y: float, z: float, task: Literal["grab", "drop"]
    ) -> str:
        lift = self.additional_height if task == "drop" else 0.0
        if await self.move_calibrated(x, y, z, grab=task == "grab", lift=lift):
            return f"End effector successfully positioned at coordinates ({x:.2f}, {y:.2f}, {z:.2f}). Note: The status of object interaction (grab/drop) is not confirmed by this movement."
        return f"Failed to position end effector at coordinates ({x:.2f}, {y:.2f}, {z:.2f})."


class MoveObjectFromToToolInput(BaseModel):
    x: float = Field(
        description="The x coordinate of the source point (pick-up location) in the manipulator frame"
    )
    y: float = Field(
        description="The y coordinate of the source point (pick-up location) in the manipulator frame"
    )
    z: float = Field(
        description="The z coordinate of the source point (pick-up location) in the manipulator frame"
    )
    x1: float = Field(
        description="The x coordinate of the destination point (drop-off location) in the manipulator frame"
    )
    y1: float = Field(
        description="The y coordinate of the destination point (drop-off location) in the manipulator frame"
    )
    z1: float = Field(
        description="The z coordinate of the destination point (drop-off location) in the manipulator frame"
    )


class MoveObjectFromToTool(CalibratedManipulatorTool):
    """Pick at (x, y, z), place at (x1, y1, z1), then reset the arm, also when
    the place move fails."""

    name: str = "move_object_from_to"
    description: str = (
        "Move an object from one point to another. "
        "The tool will grab the object from the first point and then release it at the second point. "
        "The tool will not confirm the success of grabbing or releasing objects. Use additional sensors (e.g. camera) or tools for that information."
    )
    calibration_z: float = Field(default=0.1, description="Calibration z [m]")
    args_schema: Type[MoveObjectFromToToolInput] = MoveObjectFromToToolInput

    async def _acall(
        self, x: float, y: float, z: float, x1: float, y1: float, z1: float
    ) -> str:
        if not await self.move_calibrated(x, y, z, grab=True):
            return f"Failed to position end effector at coordinates ({x:.2f}, {y:.2f}, {z:.2f})."
        try:
            placed = await self.move_calibrated(x1, y1, z1, grab=False)
        finally:
            with contextlib.suppress(Exception):
                await asyncio.shield(self.reset())
        if placed:
            return f"End effector successfully positioned at coordinates ({x1:.2f}, {y1:.2f}, {z1:.2f}). Note: The status of object interaction (grab/drop) is not confirmed by this movement."
        return f"Failed to position end effector at coordinates ({x1:.2f}, {y1:.2f}, {z1:.2f})."

    async def reset(self) -> bool:
        return await self.move(*RESET_POSITION, grab=True)


RESET_POSITION = (0.31, 0.0, 0.59)


class ResetArmTool(ManipulatorTool):
    name: str = "reset_arm"
    description: str = "Reset the arm to the initial position. Use when the arm is stuck or when arm obstructs the objects."
    timeout_sec: float = Field(
        default=5.0, description="Timeout for service calls in seconds"
    )

    async def _acall(self) -> str:
        if await self.move(*RESET_POSITION, grab=True):
            return "Arm successfully reset."
        return "Failed to reset the arm."


class GetObjectPositionsToolInput(BaseModel):
    object_name: str = Field(
        ..., description="The name of the object to get the positions of"
    )


@deprecated("Use GetObjectGrippingPointsTool from rai_perception instead")
class GetObjectPositionsTool(BaseROS2Tool):
    default_effect = Effect.READ
    name: str = "get_object_positions"
    description: str = (
        "Retrieve the positions of all objects of a specified type in the target frame. "
        "This tool provides accurate positional data but does not distinguish between different colors of the same object type. "
        "While position detection is reliable, please note that object classification may occasionally be inaccurate."
    )

    target_frame: str
    source_frame: str
    camera_topic: str  # rgb camera topic
    depth_topic: str
    camera_info_topic: str  # rgb camera info topic
    get_grabbing_point_tool: BaseTool  # rai_perception.tools.GetGrabbingPointTool

    args_schema: Type[GetObjectPositionsToolInput] = GetObjectPositionsToolInput

    @staticmethod
    def format_pose(pose: Pose) -> str:
        return f"Centroid(x={pose.position.x:.2f}, y={pose.position.y:.2f}, z={pose.position.z:.2f})"

    async def _acall(self, object_name: str) -> str:
        transform = await self.connector.transform(self.target_frame, self.source_frame)
        results: Any = await asyncio.to_thread(
            self.get_grabbing_point_tool._run,
            camera_topic=self.camera_topic,
            depth_topic=self.depth_topic,
            camera_info_topic=self.camera_info_topic,
            object_name=object_name,
        )
        poses = [
            do_transform_pose(
                Pose(position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))),
                transform,
            )
            for p, *_ in results
        ]
        if not poses:
            return f"No {object_name}s detected."
        # Sort by rounded position for a stable, deterministic ordering across runs.
        poses.sort(
            key=lambda p: tuple(
                round(v, 2) for v in (p.position.x, p.position.y, p.position.z)
            )
        )
        return f"Centroids of detected {object_name}s in {self.target_frame} frame: [{', '.join(map(self.format_pose, poses))}]. Sizes of the detected objects are unknown."
