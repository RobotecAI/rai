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

import base64
import time
from typing import List, Optional, Type, cast

import cv2
import numpy as np
from geometry_msgs.msg import Point, PoseStamped, Quaternion, TransformStamped
from langchain_core.tools import BaseTool
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from pydantic import BaseModel, Field
from rclpy.action import ActionClient
from tf_transformations import euler_from_quaternion, quaternion_from_euler

from rai.communication.ros2 import ROS2Message
from rai.communication.ros2.connectors import ROS2Connector
from rai.messages import MultimodalArtifact
from rai.tools.ros2.base import BaseROS2Tool, BaseROS2Toolkit

action_client: Optional[ActionClient] = None
current_action_id: Optional[str] = None
current_feedback: Optional[NavigateToPose.Feedback] = None
current_result: Optional[NavigateToPose.Result] = None


class Nav2Toolkit(BaseROS2Toolkit):
    connector: ROS2Connector
    frame_id: str = Field(
        default="map", description="The frame id of the Nav2 stack (map, odom, etc.)"
    )
    action_name: str = Field(
        default="navigate_to_pose", description="The name of the NavigateToPose action"
    )

    def get_tools(self) -> List[BaseTool]:
        return [
            NavigateToPoseTool(
                connector=self.connector,
                frame_id=self.frame_id,
                action_name=self.action_name,
            ),
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

    frame_id: str = Field(
        default="map", description="The frame id of the Nav2 stack (map, odom, etc.)"
    )
    action_name: str = Field(
        default="navigate_to_pose", description="The name of the NavigateToPose action"
    )

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

        msg = ROS2Message(payload=goal)
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


class GetOccupancyGridTool(BaseROS2Tool):
    """Get the current map as an image with the robot's position marked on it (red dot)."""

    name: str = "GetOccupancyGridTool"
    description: str = "A tool for getting the current map as an image with the robot's position marked on it."

    response_format: str = "content_and_artifact"
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

        _, _, yaw = euler_from_quaternion(
            [
                origin_orientation.x,  # type: ignore
                origin_orientation.y,  # type: ignore
                origin_orientation.z,  # type: ignore
                origin_orientation.w,  # type: ignore
            ]
        )
        # Rotate the robot's position based on the yaw angle
        rotated_x = robot_x * np.cos(yaw) - robot_y * np.sin(yaw)
        rotated_y = robot_x * np.sin(yaw) + robot_y * np.cos(yaw)
        robot_x = int(rotated_x)
        robot_y = int(rotated_y)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Draw the robot's position as an arrow
        if 0 <= robot_x < width and 0 <= robot_y < height:
            _, _, yaw = euler_from_quaternion(
                [
                    transform.transform.rotation.x,  # type: ignore
                    transform.transform.rotation.y,  # type: ignore
                    transform.transform.rotation.z,  # type: ignore
                    transform.transform.rotation.w,  # type: ignore
                ]
            )
            arrow_length = 100
            arrow_end_x = int(robot_x + arrow_length * np.cos(yaw))
            arrow_end_y = int(robot_y + arrow_length * np.sin(yaw))
            cv2.arrowedLine(
                image, (robot_x, robot_y), (arrow_end_x, arrow_end_y), (0, 0, 255), 5
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
                color=(0, 0, 255),
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
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        # Encode into PNG base64
        _, buffer = cv2.imencode(".png", image)

        if self.debug:
            cv2.imwrite(f"map{time.time()}.png", image)
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def _run(self):
        """Gets the current map from the specified topic."""
        map_msg = self.connector.receive_message("/map", timeout_sec=10).payload
        transform = self.connector.get_transform(
            target_frame="map", source_frame="base_link", timeout_sec=10
        )

        if map_msg is None or transform is None:
            return {"content": "Failed to get the map, wrong topic?"}

        base64_image = self._postprocess_msg(map_msg, transform)
        return "Map grabbed successfully", MultimodalArtifact(
            images=[base64_image], audios=[]
        )
