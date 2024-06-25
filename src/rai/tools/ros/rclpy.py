from typing import Any, Dict, Tuple, Type

import rclpy
from langchain.tools import BaseTool, tool
from langchain_core.pydantic_v1 import BaseModel, Field
from rclpy.node import Node
from rosidl_parser.definition import NamespacedType
from rosidl_runtime_py.import_message import import_message_from_namespaced_type
from rosidl_runtime_py.set_message import set_message_fields
from rosidl_runtime_py.utilities import get_namespaced_type

from rai.communication.ros_communication import SingleMessageGrabber

from .utils import import_message_from_str


@tool
def get_topics_names_and_types():
    """Call rclpy.node.Node().get_topics_names_and_types(). Return in a csv format. topic_name, serice_type"""

    node = Node(node_name="rai_tool_node")
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

    name = "Ros2GetOneMsgFromTopic"
    description: str = "A tool for getting one message from a ros2 topic"

    args_schema: Type[Ros2GetOneMsgFromTopicInput] = Ros2GetOneMsgFromTopicInput

    def _run(self, topic: str, msg_type: str):
        """Gets the current position from the specified topic."""
        msg_cls: Type = import_message_from_str(msg_type)

        grabber = SingleMessageGrabber(topic, msg_cls, timeout_sec=10)
        msg = grabber.get_data()

        if msg is None:
            return {"content": "Failed to get the position, wrong topic?"}

        return {
            "content": str(msg),
        }


class PubRos2MessageToolInput(BaseModel):
    """Input for the set_goal_pose tool."""

    topic_name: str = Field(..., description="Ros2 topic to publish the goal pose to")
    msg_type: str = Field(
        ..., description="Type of ros2 message in typical ros2 format."
    )
    msg_args: Dict[str, Any] = Field(
        ..., description="The arguments of the service call."
    )


class Ros2PubMessageTool(BaseTool):
    """Set the goal pose for the robot"""

    name = "PubRos2MessageTool"
    description: str = "A tool for setting the goal pose for the robot."

    args_schema: Type[PubRos2MessageToolInput] = PubRos2MessageToolInput

    def _build_msg(
        self, msg_type: str, msg_args: Dict[str, Any]
    ) -> Tuple[object, object]:
        msg_namespaced_type: NamespacedType = get_namespaced_type(msg_type)
        msg_cls = import_message_from_namespaced_type(msg_namespaced_type)
        msg = msg_cls()
        set_message_fields(msg, msg_args)
        return msg, msg_cls

    def _run(self, topic_name: str, msg_type: str, msg_args: Dict[str, Any]):
        """Sets the goal pose for the robot."""

        msg, msg_cls = self._build_msg(msg_type, msg_args)

        node = Node(node_name="RAI_PubRos2MessageTool")
        publisher = node.create_publisher(
            msg_cls, topic_name, 10
        )  # TODO(boczekbartek): infer qos profile from topic info
        try:
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = "map"

            rclpy.spin_once(node)
            publisher.publish(msg)
        finally:
            node.destroy_publisher(publisher)
            node.destroy_node()
