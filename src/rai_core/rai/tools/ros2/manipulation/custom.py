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

import logging
from typing import Literal, Type

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from pydantic import BaseModel, Field
from tf2_geometry_msgs import do_transform_pose

from rai.communication.ros2.ros_async import get_future_result
from rai.tools.ros2.base import BaseROS2Tool

try:
    from rai_interfaces.srv import ManipulatorMoveTo
except ImportError:
    logging.warning(
        "rai_interfaces is not installed, ManipulatorMoveTo tool will not work."
    )

try:
    from rai_perception.tools import GetGrabbingPointTool
except ImportError:
    logging.warning(
        "rai-perception is not installed, GetGrabbingPointTool will not work"
    )


class MoveToPointToolInput(BaseModel):
    x: float = Field(description="The x coordinate of the point to move to")
    y: float = Field(description="The y coordinate of the point to move to")
    z: float = Field(description="The z coordinate of the point to move to")
    task: Literal["grab", "drop"] = Field(
        description="Specify the intended action: use 'grab' to pick up an object, or 'drop' to release it. "
        "This determines the gripper's behavior during the movement."
    )


class MoveToPointTool(BaseROS2Tool):
    name: str = "move_to_point"
    description: str = (
        "Guide the robot's end effector to a specific point within the manipulator's operational space. "
        "This tool ensures precise movement to the desired location. "
        "While it confirms successful positioning, please note that it doesn't provide feedback on the "
        "success of grabbing or releasing objects. Use additional sensors or tools for that information."
    )

    manipulator_frame: str = Field(..., description="Manipulator frame")
    min_z: float = Field(default=0.135, description="Minimum z coordinate [m]")
    calibration_x: float = Field(default=0.0, description="Calibration x [m]")
    calibration_y: float = Field(default=0.0, description="Calibration y [m]")
    calibration_z: float = Field(default=0.0, description="Calibration z [m]")
    additional_height: float = Field(
        default=0.05, description="Additional height for the place task [m]"
    )

    # constant quaternion
    quaternion: Quaternion = Field(
        default=Quaternion(x=0.9238795325112867, y=-0.3826834323650898, z=0.0, w=0.0),
        description="Constant quaternion",
    )

    args_schema: Type[MoveToPointToolInput] = MoveToPointToolInput

    def _run(
        self,
        x: float,
        y: float,
        z: float,
        task: Literal["grab", "drop"],
    ) -> str:
        client = self.connector.node.create_client(
            ManipulatorMoveTo,
            "/manipulator_move_to",
        )
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.manipulator_frame
        pose_stamped.pose = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=self.quaternion,
        )

        if task == "drop":
            pose_stamped.pose.position.z += self.additional_height

        pose_stamped.pose.position.x += self.calibration_x
        pose_stamped.pose.position.y += self.calibration_y
        pose_stamped.pose.position.z += self.calibration_z

        pose_stamped.pose.position.z = np.max(
            [pose_stamped.pose.position.z, self.min_z]
        )

        request = ManipulatorMoveTo.Request()
        request.target_pose = pose_stamped

        if task == "grab":
            request.initial_gripper_state = True  # open
            request.final_gripper_state = False  # closed
        else:
            request.initial_gripper_state = False  # closed
            request.final_gripper_state = True  # open

        future = client.call_async(request)
        self.connector.node.get_logger().debug(
            f"Calling ManipulatorMoveTo service with request: x={request.target_pose.pose.position.x:.2f}, y={request.target_pose.pose.position.y:.2f}, z={request.target_pose.pose.position.z:.2f}"
        )
        response = get_future_result(future, timeout_sec=20.0)
        if response is None:
            return f"Service call failed for point ({x:.2f}, {y:.2f}, {z:.2f})."

        if response.success:
            return f"End effector successfully positioned at coordinates ({x:.2f}, {y:.2f}, {z:.2f}). Note: The status of object interaction (grab/drop) is not confirmed by this movement."
        else:
            return f"Failed to position end effector at coordinates ({x:.2f}, {y:.2f}, {z:.2f})."


class MoveObjectFromToToolInput(BaseModel):
    x: float = Field(description="The x coordinate of the point to move to")
    y: float = Field(description="The y coordinate of the point to move to")
    z: float = Field(description="The z coordinate of the point to move to")
    x1: float = Field(description="The x coordinate of the point to move to")
    y1: float = Field(description="The y coordinate of the point to move to")
    z1: float = Field(description="The z coordinate of the point to move to")


class MoveObjectFromToTool(BaseROS2Tool):
    name: str = "move_object_from_to"
    description: str = (
        "Move an object from one point to another. "
        "The tool will grab the object from the first point and then release it at the second point. "
        "The tool will not confirm the success of grabbing or releasing objects. Use additional sensors (e.g. camera) or tools for that information."
    )

    manipulator_frame: str = Field(..., description="Manipulator frame")
    min_z: float = Field(default=0.135, description="Minimum z coordinate [m]")
    calibration_x: float = Field(default=0.0, description="Calibration x [m]")
    calibration_y: float = Field(default=0.0, description="Calibration y [m]")
    calibration_z: float = Field(default=0.1, description="Calibration z [m]")
    additional_height: float = Field(
        default=0.05, description="Additional height for the place task [m]"
    )

    # constant quaternion
    quaternion: Quaternion = Field(
        default=Quaternion(x=0.9238795325112867, y=-0.3826834323650898, z=0.0, w=0.0),
        description="Constant quaternion",
    )

    args_schema: Type[MoveObjectFromToToolInput] = MoveObjectFromToToolInput

    def _run(
        self,
        x: float,
        y: float,
        z: float,
        x1: float,
        y1: float,
        z1: float,
    ) -> str:
        # NOTE: create_client could be refactored into self.connector.service_call
        self.connector.service_call
        client = self.connector.node.create_client(
            ManipulatorMoveTo,
            "/manipulator_move_to",
        )
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.manipulator_frame
        pose_stamped.pose = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=self.quaternion,
        )

        pose_stamped1 = PoseStamped()
        pose_stamped1.header.frame_id = self.manipulator_frame
        pose_stamped1.pose = Pose(
            position=Point(x=x1, y=y1, z=z1),
            orientation=self.quaternion,
        )

        pose_stamped.pose.position.x += self.calibration_x
        pose_stamped.pose.position.y += self.calibration_y
        pose_stamped.pose.position.z += self.calibration_z

        pose_stamped.pose.position.z = np.max(
            [pose_stamped.pose.position.z, self.min_z]
        )

        pose_stamped1.pose.position.x += self.calibration_x
        pose_stamped1.pose.position.y += self.calibration_y
        pose_stamped1.pose.position.z += self.calibration_z

        pose_stamped1.pose.position.z = np.max(
            [pose_stamped1.pose.position.z, self.min_z]
        )

        request = ManipulatorMoveTo.Request()
        request.target_pose = pose_stamped

        request.initial_gripper_state = True  # open
        request.final_gripper_state = False  # closed

        future = client.call_async(request)
        self.connector.node.get_logger().debug(
            f"Calling ManipulatorMoveTo service with request: x={request.target_pose.pose.position.x:.2f}, y={request.target_pose.pose.position.y:.2f}, z={request.target_pose.pose.position.z:.2f}"
        )
        response = get_future_result(future, timeout_sec=5.0)

        if response is None:
            return f"Service call failed for point ({x:.2f}, {y:.2f}, {z:.2f})."

        if response.success:
            self.connector.logger.info(
                f"End effector successfully positioned at coordinates ({x:.2f}, {y:.2f}, {z:.2f})."
            )
        else:
            self.connector.logger.error(
                f"Failed to position end effector at coordinates ({x:.2f}, {y:.2f}, {z:.2f})."
            )
            return "Failed to position end effector at coordinates ({x:.2f}, {y:.2f}, {z:.2f})."

        request = ManipulatorMoveTo.Request()
        request.target_pose = pose_stamped1

        request.initial_gripper_state = False  # closed
        request.final_gripper_state = True  # open

        future = client.call_async(request)
        self.connector.node.get_logger().debug(
            f"Calling ManipulatorMoveTo service with request: x={request.target_pose.pose.position.x:.2f}, y={request.target_pose.pose.position.y:.2f}, z={request.target_pose.pose.position.z:.2f}"
        )

        response = get_future_result(future, timeout_sec=20.0)

        if response is None:
            return f"Service call failed for point ({x1:.2f}, {y1:.2f}, {z1:.2f})."

        if response.success:
            return f"End effector successfully positioned at coordinates ({x1:.2f}, {y1:.2f}, {z1:.2f}). Note: The status of object interaction (grab/drop) is not confirmed by this movement."
        else:
            return f"Failed to position end effector at coordinates ({x1:.2f}, {y1:.2f}, {z1:.2f})."


class GetObjectPositionsToolInput(BaseModel):
    object_name: str = Field(
        ..., description="The name of the object to get the positions of"
    )


class GetObjectPositionsTool(BaseROS2Tool):
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
    get_grabbing_point_tool: "GetGrabbingPointTool"

    args_schema: Type[GetObjectPositionsToolInput] = GetObjectPositionsToolInput

    @staticmethod
    def format_pose(pose: Pose):
        return f"Centroid(x={pose.position.x:.2f}, y={pose.position.y:2f}, z={pose.position.z:2f})"

    def _run(self, object_name: str):
        transform = self.connector.get_transform(
            target_frame=self.target_frame, source_frame=self.source_frame
        )

        results = self.get_grabbing_point_tool._run(
            camera_topic=self.camera_topic,
            depth_topic=self.depth_topic,
            camera_info_topic=self.camera_info_topic,
            object_name=object_name,
        )

        poses = []
        for result in results:
            cam_pose = result[0]
            poses.append(
                Pose(position=Point(x=cam_pose[0], y=cam_pose[1], z=cam_pose[2]))
            )

        mani_frame_poses = []
        for pose in poses:
            mani_frame_pose = do_transform_pose(pose, transform)
            mani_frame_poses.append(mani_frame_pose)

        if len(mani_frame_poses) == 0:
            return f"No {object_name}s detected."
        else:
            return f"Centroids of detected {object_name}s in {self.target_frame} frame: [{', '.join(map(self.format_pose, mani_frame_poses))}]. Sizes of the detected objects are unknown."


class ResetArmToolInput(BaseModel):
    pass


class ResetArmTool(BaseROS2Tool):
    name: str = "reset_arm"
    description: str = "Reset the arm to the initial position. Use when the arm is stuck or when arm obstructs the objects."

    args_schema: Type[ResetArmToolInput] = ResetArmToolInput

    # constant quaternion
    quaternion: Quaternion = Field(
        default=Quaternion(x=0.9238795325112867, y=-0.3826834323650898, z=0.0, w=0.0),
        description="Constant quaternion",
    )
    manipulator_frame: str = Field(..., description="Manipulator frame")

    def _run(self):
        client = self.connector.node.create_client(
            ManipulatorMoveTo,
            "/manipulator_move_to",
        )

        x = 0.31
        y = 0.0
        z = 0.59

        request = ManipulatorMoveTo.Request()
        request.target_pose = PoseStamped()
        request.target_pose.header.frame_id = self.manipulator_frame
        request.target_pose.pose = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=self.quaternion,
        )

        request.initial_gripper_state = True  # open
        request.final_gripper_state = False  # closed

        future = client.call_async(request)
        self.connector.node.get_logger().debug(
            f"Calling ManipulatorMoveTo service with request: x={request.target_pose.pose.position.x:.2f}, y={request.target_pose.pose.position.y:.2f}, z={request.target_pose.pose.position.z:.2f}"
        )
        response = get_future_result(future, timeout_sec=5.0)

        if response is None:
            return "Failed to reset the arm."

        if response.success:
            return "Arm successfully reset."
        else:
            return "Failed to reset the arm."
