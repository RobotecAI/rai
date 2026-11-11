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

"""
This module provides streamlined ROS2 tools designed for enhanced usability.
Unlike ROS2Toolkit, these tools are purpose-built for specific use cases
rather than offering generic functionality across topics, services, and actions.
For generic ROS2 functionality, use ROS2Toolkit.
"""

from typing import Any, Dict, List, Literal, Tuple

from pydantic import Field

from rai.tools.hooks import Effect
from rai.tools.ros2.base import BaseROS2Tool
from rai.tools.ros2.generic.topics import describe_transform, receive_image


class GetROS2ImageConfiguredTool(BaseROS2Tool):
    default_effect = Effect.READ
    target_fields = {"topic": "topic"}
    name: str = "get_ros2_camera_image"
    description: str = "Get the current image from the camera"
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    topic: str = Field(..., min_length=1, description="The topic to get the image from")
    timeout_sec: float = Field(default=5.0, gt=0, description="The timeout in seconds")

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if not self.is_readable(self.topic):
            raise ValueError(f"Bad configuration: topic {self.topic} is not readable")

    async def _acall(self) -> Tuple[str, Dict[str, List[str]]]:
        return await receive_image(self, self.topic, self.timeout_sec)


class GetROS2TransformConfiguredTool(BaseROS2Tool):
    default_effect = Effect.READ
    name: str = "get_ros2_robot_position"
    description: str = "Get the robot's position"

    source_frame: str = Field(..., min_length=1, description="The source frame")
    target_frame: str = Field(..., min_length=1, description="The target frame")
    timeout_sec: float = Field(default=10.0, gt=0, description="The timeout in seconds")

    async def _acall(self) -> str:
        return await describe_transform(
            self, self.target_frame, self.source_frame, self.timeout_sec
        )
