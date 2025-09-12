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

        if result.result.error_code != 0:
            return f"Navigate to pose action failed. Error code: {result.result.error_code}"

        return "Navigate to pose successful."
