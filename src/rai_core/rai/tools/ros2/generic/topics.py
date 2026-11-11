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

import json
from typing import Any, Dict, List, Literal, Tuple, Type

import rosidl_runtime_py.utilities
from langchain_core.tools import BaseTool
from langchain_core.utils import stringify_dict
from pydantic import BaseModel, Field
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Image

from rai.communication.ros2 import encode_image
from rai.tools.hooks import Effect
from rai.tools.ros2.base import (
    BaseROS2Tool,
    BaseROS2Toolkit,
    from_dict,
    to_dict,
    type_name,
)
from rai.tools.ros2.generic.interface_parser import render_interface_string


class ROS2TopicsToolkit(BaseROS2Toolkit):
    name: str = "ROS2TopicsToolkit"
    description: str = "A toolkit for ROS2 topics"

    def get_tools(self) -> List[BaseTool]:
        return self._build(
            PublishROS2MessageTool,
            ReceiveROS2MessageTool,
            GetROS2ImageTool,
            GetROS2TransformTool,
            GetROS2TopicsNamesAndTypesTool,
            GetROS2MessageInterfaceTool,
        )


class PublishROS2MessageToolInput(BaseModel):
    topic: str = Field(
        ..., min_length=1, description="The topic to publish the message to"
    )
    message: Dict[str, Any] = Field(..., description="The message to publish")
    message_type: str = Field(..., min_length=1, description="The type of the message")


class PublishROS2MessageTool(BaseROS2Tool):
    """Publishing to an arbitrary topic (e.g. /cmd_vel) can move the robot."""

    default_effect = Effect.ACTUATE
    target_args = {"topic": "topic"}
    name: str = "publish_ros2_message"
    description: str = "Publish a message to a ROS2 topic"
    args_schema: Type[PublishROS2MessageToolInput] = PublishROS2MessageToolInput

    async def _acall(
        self, topic: str, message: Dict[str, Any], message_type: str
    ) -> str:
        self.check_writable(topic)
        await self.connector.publish(topic, from_dict(message_type, message))
        return "Message published successfully"


class ReceiveROS2MessageToolInput(BaseModel):
    topic: str = Field(
        ..., min_length=1, description="The topic to receive the message from"
    )
    timeout_sec: float = Field(1.0, gt=0, description="The timeout in seconds")


class ReceiveROS2MessageTool(BaseROS2Tool):
    default_effect = Effect.READ
    target_args = {"topic": "topic"}
    name: str = "receive_ros2_message"
    description: str = "Receive a message from a ROS2 topic"
    args_schema: Type[ReceiveROS2MessageToolInput] = ReceiveROS2MessageToolInput

    async def _acall(self, topic: str, timeout_sec: float = 1.0) -> str:
        self.check_readable(topic)
        msg = await self.connector.receive(topic, timeout=timeout_sec)
        return str({"payload": to_dict(msg), "msg_type": type_name(msg)})


class GetROS2ImageToolInput(BaseModel):
    topic: str = Field(
        ..., min_length=1, description="The topic to receive the image from"
    )
    timeout_sec: float = Field(1.0, gt=0, description="The timeout in seconds")


async def receive_image(
    tool: BaseROS2Tool, topic: str, timeout_sec: float
) -> Tuple[str, Dict[str, List[str]]]:
    tool.check_readable(topic)
    msg = await tool.connector.receive(topic, timeout=timeout_sec)
    if not isinstance(msg, (Image, CompressedImage)):
        raise ValueError(f"Unsupported message type: {type_name(msg)}")
    return "Image received successfully", {"images": [encode_image(msg)]}


class GetROS2ImageTool(BaseROS2Tool):
    default_effect = Effect.READ
    target_args = {"topic": "topic"}
    name: str = "get_ros2_image"
    description: str = "Get an image from a ROS2 topic"
    args_schema: Type[GetROS2ImageToolInput] = GetROS2ImageToolInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    async def _acall(
        self, topic: str, timeout_sec: float = 1.0
    ) -> Tuple[str, Dict[str, List[str]]]:
        return await receive_image(self, topic, timeout_sec)


class GetROS2TopicsNamesAndTypesTool(BaseROS2Tool):
    default_effect = Effect.READ
    name: str = "get_ros2_topics_names_and_types"
    description: str = "Get the names and types of all ROS2 topics"

    async def _acall(self) -> str:
        groups: Dict[str, List[str]] = {"both": [], "read": [], "write": []}
        unrestricted = self.readable is None and self.writable is None
        for topic, types in self.connector.get_topics_names_and_types():
            if self._forbidden(topic):
                continue
            readable, writable = self.is_readable(topic), self.is_writable(topic)
            if not (
                unrestricted
                or (self.readable is not None and readable)
                or (self.writable is not None and writable)
            ):
                continue
            group = "both" if readable and writable else "read" if readable else "write"
            groups[group].append(stringify_dict({"topic": topic, "type": types}))
        lines = groups["both"]
        if groups["read"]:
            lines += ["Readable topics:", *groups["read"]]
        if groups["write"]:
            lines += ["Writable topics:", *groups["write"]]
        return "\n".join(lines)


class GetROS2MessageInterfaceToolInput(BaseModel):
    msg_type: str = Field(
        ..., description="The type of the message e.g. std_msgs/msg/String"
    )


class GetROS2MessageInterfaceTool(BaseROS2Tool):
    default_effect = Effect.READ
    name: str = "get_ros2_message_interface"
    description: str = "Get the interface of a ROS2 message"
    args_schema: Type[GetROS2MessageInterfaceToolInput] = (
        GetROS2MessageInterfaceToolInput
    )

    async def _acall(self, msg_type: str) -> str:
        """Show ros2 message interface in json format."""
        msg_cls: Any = rosidl_runtime_py.utilities.get_interface(msg_type)
        try:
            return render_interface_string(msg_type)
        except (ValueError, LookupError, NotImplementedError):
            # For action classes that can't be instantiated
            return json.dumps(
                {
                    "goal": to_dict(msg_cls.Goal()),
                    "result": to_dict(msg_cls.Result()),
                    "feedback": to_dict(msg_cls.Feedback()),
                }
            )


class GetROS2TransformToolInput(BaseModel):
    target_frame: str = Field(..., min_length=1, description="The target frame")
    source_frame: str = Field(..., min_length=1, description="The source frame")
    timeout_sec: float = Field(default=10.0, gt=0, description="The timeout in seconds")


async def describe_transform(
    tool: BaseROS2Tool, target_frame: str, source_frame: str, timeout_sec: float
) -> str:
    """The transform as text, with a warning if it is more than 1 s old."""
    transform = await tool.connector.transform(
        target_frame, source_frame, timeout=timeout_sec
    )
    age = (tool.connector.now() - Time.from_msg(transform.header.stamp)).nanoseconds
    result = stringify_dict(to_dict(transform))
    if abs(age / 1e9) > 1.0:
        result = (
            f"WARNING: Robot position data is {age / 1e9:.1f} seconds old"
            " (potentially stale). The robot may have moved since then.\n\n" + result
        )
    return result


class GetROS2TransformTool(BaseROS2Tool):
    default_effect = Effect.READ
    name: str = "get_ros2_transform"
    description: str = "Get the transform between two frames"
    args_schema: Type[GetROS2TransformToolInput] = GetROS2TransformToolInput

    async def _acall(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> str:
        return await describe_transform(self, target_frame, source_frame, timeout_sec)
