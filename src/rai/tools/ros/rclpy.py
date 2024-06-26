from typing import Any, Dict, Tuple, Type

import rclpy
from langchain.tools import BaseTool, tool
from langchain_core.pydantic_v1 import BaseModel, Field
from rclpy.node import Node
from rosidl_runtime_py.set_message import set_message_fields

from rai.communication.ros_communication import SingleMessageGrabber

from .utils import import_message_from_str


@tool
def get_topics_names_and_types_tool():
    """Call rclpy.node.Node().get_topics_names_and_types()."""

    node = Node(node_name="rai_get_topics_names_and_types_tool")
    rclpy.spin_once(node, timeout_sec=2)
    try:
        return [
            (topic_name, topic_type)
            for topic_name, topic_type in node.get_topic_names_and_types()
            if len(topic_name.split("/")) <= 2
        ]
    finally:
        node.destroy_node()


class Ros2GetOneMsgFromTopicInput(BaseModel):
    """Input for the get_current_position tool."""

    topic: str = Field(..., description="Ros2 topic")
    msg_type: str = Field(
        ..., description="Type of ros2 message in typical ros2 format."
    )


class Ros2GetOneMsgFromTopicTool(BaseTool):
    """Get one message from a specific ros2 topic"""

    name: str = "Ros2GetOneMsgFromTopic"
    description: str = "A tool for getting one message from a ros2 topic"

    args_schema: Type[Ros2GetOneMsgFromTopicInput] = Ros2GetOneMsgFromTopicInput

    def _run(self, topic: str, msg_type: str):
        """Gets the current position from the specified topic."""
        msg_cls: Type = import_message_from_str(msg_type)

        grabber = SingleMessageGrabber(topic, msg_cls, timeout_sec=10)
        msg = grabber.get_data()

        if msg is None:
            return {"content": "No message received."}

        return {
            "content": str(msg),
        }


class PubRos2MessageToolInput(BaseModel):
    topic_name: str = Field(..., description="Ros2 topic to publish the message")
    msg_type: str = Field(
        ..., description="Type of ros2 message in typical ros2 format."
    )
    msg_args: Dict[str, Any] = Field(
        ..., description="The arguments of the service call."
    )


class Ros2PubMessageTool(BaseTool):
    name: str = "PubRos2MessageTool"
    description: str = """A tool for publishing a message to a ros2 topic
    Example usage:

    ```python
    tool = Ros2PubMessageTool()
    tool.run(
        {
            "topic_name": "/some_topic",
            "msg_type": "geometry_msgs/Point",
            "msg_args": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
    )

    ```
    """

    args_schema: Type[PubRos2MessageToolInput] = PubRos2MessageToolInput

    def _build_msg(
        self, msg_type: str, msg_args: Dict[str, Any]
    ) -> Tuple[object, Type]:
        msg_cls: Type = import_message_from_str(msg_type)
        msg = msg_cls()
        set_message_fields(msg, msg_args)
        return msg, msg_cls

    def _run(self, topic_name: str, msg_type: str, msg_args: Dict[str, Any]):
        """Publishes a message to the specified topic."""
        msg, msg_cls = self._build_msg(msg_type, msg_args)

        node = Node(node_name=f"rai_{self,__class__.__name__}")
        publisher = node.create_publisher(
            msg_cls, topic_name, 10
        )  # TODO(boczekbartek): infer qos profile from topic info
        try:
            msg.header.stamp = node.get_clock().now().to_msg()

            rclpy.spin_once(node)
            publisher.publish(msg)
        finally:
            node.destroy_publisher(publisher)
            node.destroy_node()
