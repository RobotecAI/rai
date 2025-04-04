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

from typing import List, Optional, Type

from geometry_msgs.msg import PoseStamped, Quaternion
from langchain_core.tools import BaseTool
from nav2_msgs.action import NavigateToPose
from pydantic import BaseModel, Field
from rclpy.action import ActionClient
from tf_transformations import quaternion_from_euler

from rai.communication.ros2 import ROS2ARIMessage
from rai.communication.ros2.connectors import ROS2ARIConnector
from rai.tools.ros2.base import BaseROS2Tool, BaseROS2Toolkit

action_client: Optional[ActionClient] = None
current_action_id: Optional[str] = None
current_feedback: Optional[NavigateToPose.Feedback] = None
current_result: Optional[NavigateToPose.Result] = None


class Nav2Toolkit(BaseROS2Toolkit):
    connector: ROS2ARIConnector

    def get_tools(self) -> List[BaseTool]:
        return [
            NavigateToPoseTool(connector=self.connector),
            CancelNavigateToPoseTool(connector=self.connector),
            GetNavigateToPoseFeedbackTool(connector=self.connector),
            GetNavigateToPoseResultTool(connector=self.connector),
        ]


class NavigateToPoseToolInput(BaseModel):
    x: float = Field(..., description="The x coordinate of the pose")
    y: float = Field(..., description="The y coordinate of the pose")
    z: float = Field(..., description="The z coordinate of the pose")
    yaw: float = Field(..., description="The yaw angle of the pose")


class NavigateToPoseTool(BaseROS2Tool):
    name: str = "navigate_to_pose"
    description: str = "Navigate to a specific pose"

    args_schema: Type[NavigateToPoseToolInput] = NavigateToPoseToolInput

    frame_id: str = "base_link"
    action_name: str = "navigate_to_pose"

    def on_feedback(self, feedback: NavigateToPose.Feedback) -> None:
        global current_feedback
        current_feedback = feedback

    def on_done(self, result: NavigateToPose.Result) -> None:
        global current_result
        current_result = result

    def _run(self, x: float, y: float, z: float, yaw: float) -> str:
        global action_client
        if action_client is None:
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

        goal = {
            "pose": {
                "header": {
                    "frame_id": self.frame_id,
                    "stamp": self.connector.node.get_clock().now().to_msg(),
                },
                "pose": {
                    "position": {"x": x, "y": y, "z": z},
                    "orientation": {
                        "x": quat[0],
                        "y": quat[1],
                        "z": quat[2],
                        "w": quat[3],
                    },
                },
            }
        }

        msg = ROS2ARIMessage(payload=goal)
        action_id = self.connector.start_action(
            action_data=msg,
            target=self.action_name,
            msg_type="nav2_msgs/action/NavigateToPose",
            on_feedback=self.on_feedback,
            on_done=self.on_done,
        )
        global current_action_id
        current_action_id = action_id

        return "Navigating to pose"


class GetNavigateToPoseFeedbackTool(BaseROS2Tool):
    name: str = "get_navigate_to_pose_feedback"
    description: str = "Get the feedback of the navigate to pose action"

    def _run(self) -> str:
        global current_feedback
        return str(current_feedback)


class GetNavigateToPoseResultTool(BaseROS2Tool):
    name: str = "get_navigate_to_pose_result"
    description: str = "Get the result of the navigate to pose action"

    def _run(self) -> str:
        global current_result
        if current_result is None:
            return "Action is not done yet"
        return str(current_result.result().result)


class CancelNavigateToPoseTool(BaseROS2Tool):
    name: str = "cancel_navigate_to_pose"
    description: str = "Cancel the navigate to pose action"

    def _run(self) -> str:
        global current_action_id
        if current_action_id is None:
            return "No action to cancel"
        self.connector.terminate_action(current_action_id)
        return "Action cancelled"
