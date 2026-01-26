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

from typing import Type

from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from pydantic import BaseModel, Field
from rclpy.action import ActionClient
from tf_transformations import quaternion_from_euler

from rai.tools.ros2.base import BaseROS2Tool


def _get_error_code_string(error_code: int) -> str:
    """Convert NavigateToPose error code to human-readable string."""
    error_code_map = {
        0: "NONE",
        1: "UNKNOWN",
        2: "TIMEOUT",
        3: "CANCELED",
        4: "FAILED",
        5: "INVALID_POSE",
        6: "PLANNER_FAILED",
        7: "CONTROLLER_FAILED",
        8: "RECOVERY_FAILED",
    }
    return error_code_map.get(error_code, f"UNKNOWN_ERROR_CODE_{error_code}")


def _get_error_message(result) -> str:
    """Extract error message from NavigateToPose result."""
    error_parts = []

    # Get error code string for context
    error_code_str = _get_error_code_string(result.error_code)
    error_parts.append(f"({error_code_str})")

    # Check for additional error message fields (if they exist)
    # These may vary between ROS2 versions, so we check safely
    if hasattr(result, "error_message") and result.error_message:
        error_parts.append(f"Error message: {result.error_message}")
    elif hasattr(result, "error_msg") and result.error_msg:
        error_parts.append(f"Error message: {result.error_msg}")
    elif hasattr(result, "message") and result.message:
        error_parts.append(f"Message: {result.message}")

    # Include full result string representation for debugging
    result_str = str(result)
    if result_str and result_str != str(result.error_code):
        # Only add if it provides additional info beyond just the error code
        if len(result_str) > 50:  # Only if it's substantial
            error_parts.append(f"Full result: {result_str}")

    return ". ".join(error_parts)


class GetCurrentPoseToolInput(BaseModel):
    pass


class GetCurrentPoseTool(BaseROS2Tool):
    name: str = "get_current_pose"
    description: str = "Get the current pose of the robot"
    frame_id: str = Field(
        default="map", description="The frame id of the Nav2 stack (map, odom, etc.)"
    )
    robot_frame_id: str = Field(
        default="base_link",
        description="The frame id of the robot's base frame (base_link, base_footprint, etc.)",
    )
    args_schema: Type[GetCurrentPoseToolInput] = GetCurrentPoseToolInput

    def _run(self) -> str:
        transform_stamped = self.connector.get_transform(
            self.frame_id, self.robot_frame_id
        )
        return str(transform_stamped)


class NavigateToPoseBlockingToolInput(BaseModel):
    x: float = Field(..., description="The x coordinate of the pose")
    y: float = Field(..., description="The y coordinate of the pose")
    z: float = Field(..., description="The z coordinate of the pose")
    yaw: float = Field(..., description="The yaw angle of the pose")


class NavigateToPoseBlockingTool(BaseROS2Tool):
    name: str = "navigate_to_pose_blocking"
    description: str = "Navigate to a specific pose"
    frame_id: str = Field(
        default="map", description="The frame id of the Nav2 stack (map, odom, etc.)"
    )
    action_name: str = Field(
        default="navigate_to_pose", description="The name of the Nav2 action"
    )
    args_schema: Type[NavigateToPoseBlockingToolInput] = NavigateToPoseBlockingToolInput

    def _run(self, x: float, y: float, z: float, yaw: float) -> str:
        action_client = ActionClient(
            self.connector.node, NavigateToPose, self.action_name
        )

        pose = PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.header.stamp = self.connector.node.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        quat = quaternion_from_euler(0, 0, yaw)
        pose.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])

        goal = NavigateToPose.Goal()
        goal.pose = pose

        result = action_client.send_goal(goal)

        if result is None:
            return "Navigate to pose action failed. Please try again."

        if result.error_code != 0:
            error_msg = _get_error_message(result)
            return f"Navigate to pose action failed. Error code: {result.error_code}. {error_msg}"

        return "Navigate to pose successful."
