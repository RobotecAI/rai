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

try:
    pass
except ImportError:
    raise ImportError(
        "This is a ROS2 feature. Make sure ROS2 is installed and sourced."
    )

import json
from typing import Any, Dict, Literal, OrderedDict, Tuple, Type

import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities
from cv_bridge import CvBridge
from langchain.tools import BaseTool
from langchain_core.utils import stringify_dict
from pydantic import BaseModel, Field
from sensor_msgs.msg import CompressedImage, Image

from rai.communication.ros2.connectors import ROS2ARIConnector, ROS2ARIMessage
from rai.messages.multimodal import MultimodalArtifact
from rai.messages.utils import preprocess_image
from rai.tools.utils import wrap_tool_input  # type: ignore


class PublishROS2MessageToolInput(BaseModel):
    topic: str = Field(..., description="The topic to publish the message to")
    message: Dict[str, Any] = Field(..., description="The message to publish")
    message_type: str = Field(..., description="The type of the message")


class PublishROS2MessageTool(BaseTool):
    connector: ROS2ARIConnector
    name: str = "publish_ros2_message"
    description: str = "Publish a message to a ROS2 topic"
    args_schema: Type[PublishROS2MessageToolInput] = PublishROS2MessageToolInput

    @wrap_tool_input
    def _run(self, tool_input: PublishROS2MessageToolInput) -> str:
        ros_message = ROS2ARIMessage(
            payload=tool_input.message,
            metadata={"topic": tool_input.topic, "msg_type": tool_input.message_type},
        )
        self.connector.send_message(ros_message, target=tool_input.topic)
        return "Message published successfully"


class ReceiveROS2MessageToolInput(BaseModel):
    topic: str = Field(..., description="The topic to receive the message from")


class ReceiveROS2MessageTool(BaseTool):
    connector: ROS2ARIConnector
    name: str = "receive_ros2_message"
    description: str = "Receive a message from a ROS2 topic"
    args_schema: Type[ReceiveROS2MessageToolInput] = ReceiveROS2MessageToolInput

    @wrap_tool_input
    def _run(self, tool_input: ReceiveROS2MessageToolInput) -> str:
        message = self.connector.receive_message(tool_input.topic)
        return str({"payload": message.payload, "metadata": message.metadata})


class GetROS2ImageToolInput(BaseModel):
    topic: str = Field(..., description="The topic to receive the image from")


class GetROS2ImageTool(BaseTool):
    connector: ROS2ARIConnector
    name: str = "get_ros2_image"
    description: str = "Get an image from a ROS2 topic"
    args_schema: Type[GetROS2ImageToolInput] = GetROS2ImageToolInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    @wrap_tool_input
    def _run(self, tool_input: GetROS2ImageToolInput) -> Tuple[str, MultimodalArtifact]:
        message = self.connector.receive_message(tool_input.topic)
        msg_type = type(message.payload)
        if msg_type == Image:
            image = CvBridge().imgmsg_to_cv2(  # type: ignore
                message.payload, desired_encoding="rgb8"
            )
        elif msg_type == CompressedImage:
            image = CvBridge().compressed_imgmsg_to_cv2(  # type: ignore
                message.payload, desired_encoding="rgb8"
            )
        else:
            raise ValueError(
                f"Unsupported message type: {message.metadata['msg_type']}"
            )
        return "Image received successfully", MultimodalArtifact(images=[preprocess_image(image)])  # type: ignore


class GetROS2TopicsNamesAndTypesToolInput(BaseModel):
    pass


class GetROS2TopicsNamesAndTypesTool(BaseTool):
    connector: ROS2ARIConnector
    name: str = "get_ros2_topics_names_and_types"
    description: str = "Get the names and types of all ROS2 topics"
    args_schema: Type[GetROS2TopicsNamesAndTypesToolInput] = (
        GetROS2TopicsNamesAndTypesToolInput
    )

    @wrap_tool_input
    def _run(self, tool_input: GetROS2TopicsNamesAndTypesToolInput) -> str:
        topics_and_types = self.connector.get_topics_names_and_types()
        response = [
            stringify_dict({"topic": topic, "type": type})
            for topic, type in topics_and_types
        ]
        return "\n".join(response)


class GetROS2MessageInterfaceToolInput(BaseModel):
    msg_type: str = Field(
        ..., description="The type of the message e.g. std_msgs/msg/String"
    )


class GetROS2MessageInterfaceTool(BaseTool):
    connector: ROS2ARIConnector
    name: str = "get_ros2_message_interface"
    description: str = "Get the interface of a ROS2 message"
    args_schema: Type[GetROS2MessageInterfaceToolInput] = (
        GetROS2MessageInterfaceToolInput
    )

    @wrap_tool_input
    def _run(self, tool_input: GetROS2MessageInterfaceToolInput) -> str:
        """Show ros2 message interface in json format."""
        msg_cls: Type[object] = rosidl_runtime_py.utilities.get_interface(
            tool_input.msg_type
        )
        try:
            msg_dict: OrderedDict[str, Any] = (
                rosidl_runtime_py.convert.message_to_ordereddict(msg_cls())  # type: ignore
            )
            return json.dumps(msg_dict)
        except NotImplementedError:
            # For action classes that can't be instantiated
            goal_dict: OrderedDict[str, Any] = (
                rosidl_runtime_py.convert.message_to_ordereddict(msg_cls.Goal())  # type: ignore
            )

            result_dict: OrderedDict[str, Any] = (
                rosidl_runtime_py.convert.message_to_ordereddict(msg_cls.Result())  # type: ignore
            )

            feedback_dict: OrderedDict[str, Any] = (
                rosidl_runtime_py.convert.message_to_ordereddict(msg_cls.Feedback())  # type: ignore
            )
            return json.dumps(
                {"goal": goal_dict, "result": result_dict, "feedback": feedback_dict}
            )


class GetROS2TransformToolInput(BaseModel):
    target_frame: str = Field(..., description="The target frame")
    source_frame: str = Field(..., description="The source frame")
    timeout_sec: float = Field(default=5.0, description="The timeout in seconds")


class GetROS2TransformTool(BaseTool):
    connector: ROS2ARIConnector
    name: str = "get_ros2_transform"
    description: str = "Get the transform between two frames"
    args_schema: Type[GetROS2TransformToolInput] = GetROS2TransformToolInput

    @wrap_tool_input
    def _run(self, tool_input: GetROS2TransformToolInput) -> str:
        transform = self.connector.get_transform(
            target_frame=tool_input.target_frame,
            source_frame=tool_input.source_frame,
            timeout_sec=tool_input.timeout_sec,
        )
        return str(transform)
