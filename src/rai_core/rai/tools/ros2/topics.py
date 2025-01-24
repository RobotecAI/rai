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

from typing import Any, Dict, Literal, Tuple, Type

from cv_bridge import CvBridge
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rai.communication.ros2.connectors import ROS2ARIConnector, ROS2ARIMessage
from rai.messages.multimodal import MultimodalArtifact
from rai.messages.utils import preprocess_image
from rai.tools.utils import wrap_tool_input  # type: ignore
from sensor_msgs.msg import CompressedImage, Image


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


class GetImageToolInput(BaseModel):
    topic: str = Field(..., description="The topic to receive the image from")


class GetImageTool(BaseTool):
    connector: ROS2ARIConnector
    name: str = "get_ros2_image"
    description: str = "Get an image from a ROS2 topic"
    args_schema: Type[GetImageToolInput] = GetImageToolInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    @wrap_tool_input
    def _run(self, tool_input: GetImageToolInput) -> Tuple[str, MultimodalArtifact]:
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
