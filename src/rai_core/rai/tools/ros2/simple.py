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

from typing import Any, Literal

from pydantic import Field

from rai.tools.ros2.base import BaseROS2Tool
from rai.tools.ros2.generic.topics import (
    GetROS2ImageTool,
    GetROS2TransformTool,
)


class GetROS2ImageConfiguredTool(BaseROS2Tool):
    name: str = "get_ros2_camera_image"
    description: str = "Get the current image from the camera"
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    topic: str = Field(..., description="The topic to get the image from")

    def model_post_init(self, __context: Any) -> None:
        if not self.is_readable(topic=self.topic):
            raise ValueError(f"Bad configuration: topic {self.topic} is not readable")

    def _run(self) -> Any:
        tool = GetROS2ImageTool(
            connector=self.connector,
        )
        return tool._run(topic=self.topic)


class GetROS2TransformConfiguredTool(BaseROS2Tool):
    name: str = "get_ros2_robot_position"
    description: str = "Get the robot's position"

    source_frame: str = Field(..., description="The source frame")
    target_frame: str = Field(..., description="The target frame")
    timeout_sec: float = Field(default=5.0, description="The timeout in seconds")

    def _run(self) -> Any:
        tool = GetROS2TransformTool(
            connector=self.connector,
        )
        return tool._run(
            source_frame=self.source_frame,
            target_frame=self.target_frame,
            timeout_sec=self.timeout_sec,
        )
