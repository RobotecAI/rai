import json
from typing import Any, Dict, List, OrderedDict, Tuple, Type

import rclpy
import rosidl_runtime_py.utilities
from langchain.tools import BaseTool
from langchain_core.pydantic_v1 import BaseModel, Field
from rclpy.action import get_action_names_and_types
from rclpy.impl.rcutils_logger import RcutilsLogger
from ros2cli.node.strategy import NodeStrategy
from rosidl_runtime_py.set_message import set_message_fields

from rai.communication.ros_communication import wait_for_message

from .utils import import_message_from_str


class Ros2BaseInput(BaseModel):
    """Empty input for ros2 tool"""


class Ros2BaseTool(BaseTool):
    node: NodeStrategy = Field(..., exclude=True, include=False, required=True)

    args_schema: Type[Ros2BaseInput] = Ros2BaseInput

    @property
    def logger(self) -> RcutilsLogger:
        return self.node.get_logger()


class Ros2GetTopicsNamesAndTypesTool(Ros2BaseTool):
    name: str = "Ros2GetTopicsNamesAndTypes"
    description: str = "A tool for getting all ros2 topics names and types"

    def _run(self):
        return [
            (topic_name, topic_type)
            for topic_name, topic_type in self.node.get_topic_names_and_types()
            if len(topic_name.split("/")) <= 2
        ]


class Ros2GetServicesNamesAndTypesTool(Ros2BaseTool):
    name: str = "Ros2GetServicesNamesAndTypes"
    description: str = "A tool for getting all ros2 services names and types"

    def _run(self):
        with NodeStrategy(dict()) as node:
            return [
                (topic_name, topic_type)
                for topic_name, topic_type in node.get_service_names_and_types()
            ]


class Ros2GetActionNamesAndTypesTool(Ros2BaseTool):
    name: str = "Ros2GetActionNamesAndTypes"
    description: str = "A tool for getting all ros2 actions names and types"

    def _run(self):
        # with NodeStrategy(dict()) as node:
        return [
            (topic_name, topic_type)
            for topic_name, topic_type in get_action_names_and_types(self.node)  # type: ignore
        ]


class Ros2GetInterfacesTool(Ros2BaseTool):
    name: str = "Ros2GetInterfaces"
    description: str = "A tool for getting all ros2 interfaces names and types"

    def _run(self) -> Dict[str, List[Tuple[str, str]]]:
        topics = [
            (topic_name, topic_type)
            for topic_name, topic_type in self.node.get_topic_names_and_types()
            if len(topic_name.split("/")) <= 2
        ]
        return {
            "actions": get_action_names_and_types(self.node),  # type: ignore
            "services": self.node.get_service_names_and_types(),
            "topics": topics,
        }


class ShowRos2MsgInterfaceInput(BaseModel):
    """Input for the show_ros2_msg_interface tool."""

    msg_name: str = Field(..., description="Ros2 message name in typical ros2 format.")


class Ros2ShowRos2MsgInterfaceTool(BaseTool):
    name: str = "ShowRos2MsgInterface"
    description: str = """A tool for showing ros2 message interface in json format.
    usage:
    ```python
    ShowRos2MsgInterface.run({"msg_name": "geometry_msgs/msg/PoseStamped"})
    ```
    """

    args_schema: Type[ShowRos2MsgInterfaceInput] = ShowRos2MsgInterfaceInput

    def _run(self, msg_name: str):
        """Show ros2 message interface in json format."""
        msg_cls: Type = rosidl_runtime_py.utilities.get_interface(msg_name)
        msg_dict: OrderedDict = rosidl_runtime_py.convert.message_to_ordereddict(
            msg_cls()
        )
        return json.dumps(msg_dict)


class Ros2GetOneMsgFromTopicInput(BaseModel):
    """Input for the get_current_position tool."""

    topic: str = Field(..., description="Ros2 topic")
    msg_type: str = Field(
        ..., description="Type of ros2 message in typical ros2 format."
    )
    timeout_sec: int = Field(
        10, description="The time in seconds to wait for a message to be received."
    )


class Ros2GetOneMsgFromTopicTool(Ros2BaseTool):
    """Get one message from a specific ros2 topic"""

    name: str = "Ros2GetOneMsgFromTopic"
    description: str = "A tool for getting one message from a ros2 topic"

    args_schema: Type[Ros2GetOneMsgFromTopicInput] = Ros2GetOneMsgFromTopicInput

    def _run(self, topic: str, msg_type: str, timeout_sec: int):
        """Gets the current position from the specified topic."""
        msg_cls: Type = import_message_from_str(msg_type)

        qos_profile = (
            rclpy.qos.qos_profile_sensor_data
        )  # TODO(@boczekbartek): infer QoS from topic

        success, msg = wait_for_message(
            msg_cls,
            self.node,
            topic,
            qos_profile=qos_profile,
            time_to_wait=timeout_sec,
        )
        msg = msg.get_data()

        if success:
            self.logger.info(f"Received message of type {msg_type} from topic {topic}")
        else:
            self.logger.error(
                f"Failed to receive message of type {msg_type} from topic {topic}"
            )

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


class Ros2PubMessageTool(Ros2BaseTool):
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
        if "/msg/" not in msg_type:
            raise ValueError("msg_name must contain 'msg' in the name.")
        msg, msg_cls = self._build_msg(msg_type, msg_args)

        publisher = self.node.create_publisher(
            msg_cls, topic_name, 10
        )  # TODO(boczekbartek): infer qos profile from topic info

        msg.header.stamp = self.node.get_clock().now().to_msg()
        publisher.publish(msg)


class Ros2ActionRunnerInput(BaseModel):
    action_name: str = Field(..., description="Name of the action")


class Ros2ActionRunner(Ros2BaseTool):
    name: str = "Ros2ActionRunner"
    description: str = "A tool for running a ros2 action"

    args_schema: Type[Ros2ActionRunnerInput] = Ros2ActionRunnerInput

    def _build_msg(
        self, msg_type: str, msg_args: Dict[str, Any]
    ) -> Tuple[object, Type]:
        msg_cls: Type = import_message_from_str(msg_type)
        msg = msg_cls.Goal()
        set_message_fields(msg, msg_args)
        return msg, msg_cls

    def _run(
        self, action_name: str, action_type: str, action_goal_args: Dict[str, Any]
    ):
        msg = _build_msg(action_type, action_goal_args)
        action_client = self.node.create_action_client(action_name, action_type)
        action_client.wait_for_server()
        action_client.send_goal(msg)
        action_client.wait_for_result()
        return action_client.get_result()
