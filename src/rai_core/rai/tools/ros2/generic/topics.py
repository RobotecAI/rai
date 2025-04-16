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

import json
from typing import Any, Dict, List, Literal, Tuple, Type

import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities
from cv_bridge import CvBridge
from langchain.tools import BaseTool
from langchain_core.utils import stringify_dict
from pydantic import BaseModel, Field
from sensor_msgs.msg import CompressedImage, Image

from rai.communication.ros2 import ROS2Connector, ROS2Message
from rai.communication.ros2.api.conversion import ros2_message_to_dict
from rai.messages import MultimodalArtifact, preprocess_image
from rai.tools.ros2.base import BaseROS2Tool, BaseROS2Toolkit
from rai.tools.ros2.generic.interface_parser import render_interface_string


class ROS2TopicsToolkit(BaseROS2Toolkit):
    name: str = "ROS2TopicsToolkit"
    description: str = "A toolkit for ROS2 topics"

    def get_tools(self) -> List[BaseTool]:
        return [
            PublishROS2MessageTool(
                connector=self.connector,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ),
            ReceiveROS2MessageTool(
                connector=self.connector,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ),
            GetROS2ImageTool(
                connector=self.connector,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ),
            GetROS2TransformTool(
                connector=self.connector,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ),
            GetROS2TopicsNamesAndTypesTool(
                connector=self.connector,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ),
            GetROS2MessageInterfaceTool(
                connector=self.connector,
                readable=self.readable,
                writable=self.writable,
                forbidden=self.forbidden,
            ),
        ]


class PublishROS2MessageToolInput(BaseModel):
    topic: str = Field(..., description="The topic to publish the message to")
    message: Dict[str, Any] = Field(..., description="The message to publish")
    message_type: str = Field(..., description="The type of the message")


class PublishROS2MessageTool(BaseROS2Tool):
    name: str = "publish_ros2_message"
    description: str = "Publish a message to a ROS2 topic"
    args_schema: Type[PublishROS2MessageToolInput] = PublishROS2MessageToolInput

    def _run(self, topic: str, message: Dict[str, Any], message_type: str) -> str:
        if not self.is_writable(topic):
            raise ValueError(f"Topic {topic} is not writable")
        ros_message = ROS2Message(
            payload=message,
            metadata={"topic": topic},
        )
        self.connector.send_message(ros_message, target=topic, msg_type=message_type)
        return "Message published successfully"


class ReceiveROS2MessageToolInput(BaseModel):
    topic: str = Field(..., description="The topic to receive the message from")
    timeout_sec: float = Field(1.0, description="The timeout in seconds")


class ReceiveROS2MessageTool(BaseROS2Tool):
    connector: ROS2Connector
    name: str = "receive_ros2_message"
    description: str = "Receive a message from a ROS2 topic"
    args_schema: Type[ReceiveROS2MessageToolInput] = ReceiveROS2MessageToolInput

    def _run(self, topic: str, timeout_sec: float = 1.0) -> str:
        if not self.is_readable(topic):
            raise ValueError(f"Topic {topic} is not readable")
        message = self.connector.receive_message(topic, timeout_sec=timeout_sec)
        return str({"payload": message.payload, "metadata": message.metadata})


class GetROS2ImageToolInput(BaseModel):
    topic: str = Field(..., description="The topic to receive the image from")
    timeout_sec: float = Field(1.0, description="The timeout in seconds")


class GetROS2ImageTool(BaseROS2Tool):
    connector: ROS2Connector
    name: str = "get_ros2_image"
    description: str = "Get an image from a ROS2 topic"
    args_schema: Type[GetROS2ImageToolInput] = GetROS2ImageToolInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    def _run(
        self, topic: str, timeout_sec: float = 1.0
    ) -> Tuple[str, MultimodalArtifact]:
        if not self.is_readable(topic):
            raise ValueError(f"Topic {topic} is not readable")
        message = self.connector.receive_message(topic, timeout_sec=timeout_sec)
        msg_type = type(message.payload)
        if msg_type == Image:
            image = CvBridge().imgmsg_to_cv2(  # type: ignore
                message.payload, desired_encoding="bgr8"
            )
        elif msg_type == CompressedImage:
            image = CvBridge().compressed_imgmsg_to_cv2(  # type: ignore
                message.payload, desired_encoding="bgr8"
            )
        else:
            raise ValueError(
                f"Unsupported message type: {message.metadata['msg_type']}"
            )
        return "Image received successfully", MultimodalArtifact(
            images=[preprocess_image(image)]
        )  # type: ignore


class GetROS2TopicsNamesAndTypesTool(BaseROS2Tool):
    connector: ROS2Connector
    name: str = "get_ros2_topics_names_and_types"
    description: str = "Get the names and types of all ROS2 topics"

    def _run(self) -> str:
        topics_and_types = self.connector.get_topics_names_and_types()
        if all([self.readable is None, self.writable is None, self.forbidden is None]):
            response = [
                {"topic": topic, "type": type} for topic, type in topics_and_types
            ]
            return "\n".join([stringify_dict(topic) for topic in response])
        else:
            readable_and_writable_topics: List[Dict[str, Any]] = []
            readable_topics: List[Dict[str, Any]] = []
            writable_topics: List[Dict[str, Any]] = []

            for topic, type in topics_and_types:
                if self.is_readable(topic) and self.is_writable(topic):
                    readable_and_writable_topics.append({"topic": topic, "type": type})
                    continue
                if self.is_readable(topic):
                    readable_topics.append({"topic": topic, "type": type})
                if self.is_writable(topic):
                    writable_topics.append({"topic": topic, "type": type})

            text_response = "\n".join(
                [
                    stringify_dict(topic_description)
                    for topic_description in readable_and_writable_topics
                ]
            )
            if readable_topics:
                text_response += "\nReadable topics:" + "\n".join(
                    [
                        stringify_dict(topic_description)
                        for topic_description in readable_topics
                    ]
                )
            if writable_topics:
                text_response += "\nWritable topics:" + "\n".join(
                    [
                        stringify_dict(topic_description)
                        for topic_description in writable_topics
                    ]
                )
            return text_response


class GetROS2MessageInterfaceToolInput(BaseModel):
    msg_type: str = Field(
        ..., description="The type of the message e.g. std_msgs/msg/String"
    )


class GetROS2MessageInterfaceTool(BaseROS2Tool):
    connector: ROS2Connector
    name: str = "get_ros2_message_interface"
    description: str = "Get the interface of a ROS2 message"
    args_schema: Type[GetROS2MessageInterfaceToolInput] = (
        GetROS2MessageInterfaceToolInput
    )

    def _run(self, msg_type: str) -> str:
        """Show ros2 message interface in json format."""
        msg_cls: Type[object] = rosidl_runtime_py.utilities.get_interface(msg_type)
        try:
            return render_interface_string(msg_type)
        except (ValueError, LookupError, NotImplementedError):
            # For action classes that can't be instantiated
            goal_dict = ros2_message_to_dict(msg_cls.Goal())  # type: ignore

            result_dict = ros2_message_to_dict(msg_cls.Result())  # type: ignore

            feedback_dict = ros2_message_to_dict(msg_cls.Feedback())  # type: ignore
            return json.dumps(
                {"goal": goal_dict, "result": result_dict, "feedback": feedback_dict}
            )


class GetROS2TransformToolInput(BaseModel):
    target_frame: str = Field(..., description="The target frame")
    source_frame: str = Field(..., description="The source frame")
    timeout_sec: float = Field(default=5.0, description="The timeout in seconds")


class GetROS2TransformTool(BaseROS2Tool):
    connector: ROS2Connector
    name: str = "get_ros2_transform"
    description: str = "Get the transform between two frames"
    args_schema: Type[GetROS2TransformToolInput] = GetROS2TransformToolInput

    def _run(self, target_frame: str, source_frame: str, timeout_sec: float) -> str:
        transform = self.connector.get_transform(
            target_frame=target_frame,
            source_frame=source_frame,
            timeout_sec=timeout_sec,
        )
        return stringify_dict(ros2_message_to_dict(transform))
