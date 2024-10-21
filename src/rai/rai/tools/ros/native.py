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
#

import importlib
import json
import time
from typing import Any, Dict, OrderedDict, Tuple, Type

import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.qos
import rclpy.subscription
import rclpy.task
import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities
import sensor_msgs.msg
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from rclpy.impl.rcutils_logger import RcutilsLogger
from rosidl_runtime_py.utilities import get_namespaced_type

from .utils import convert_ros_img_to_base64, import_message_from_str


# --------------------- Inputs ---------------------
class Ros2BaseInput(BaseModel):
    """Empty input for ros2 tool"""


class Ros2MsgInterfaceInput(BaseModel):
    """Input for the show_ros2_msg_interface tool."""

    msg_name: str = Field(..., description="Ros2 message name in typical ros2 format.")


class Ros2GetOneMsgFromTopicInput(BaseModel):
    """Input for the get_current_position tool."""

    topic: str = Field(..., description="Ros2 topic")
    msg_type: str = Field(
        ..., description="Type of ros2 message in typical ros2 format."
    )
    timeout_sec: int = Field(
        10, description="The time in seconds to wait for a message to be received."
    )


class PubRos2MessageToolInput(BaseModel):
    topic_name: str = Field(..., description="Ros2 topic to publish the message")
    msg_type: str = Field(
        ..., description="Type of ros2 message in typical ros2 format."
    )
    msg_args: Dict[str, Any] = Field(
        ..., description="The arguments of the service call."
    )
    rate: int = Field(10, description="The rate at which to publish the message.")
    timeout_seconds: int = Field(1, description="The timeout in seconds.")


# --------------------- Tools ---------------------
class Ros2BaseTool(BaseTool):
    # TODO: Make the decision between rclpy.node.Node and RaiNode
    node: rclpy.node.Node = Field(..., exclude=True, required=True)

    args_schema: Type[Ros2BaseInput] = Ros2BaseInput

    @property
    def logger(self) -> RcutilsLogger:
        return self.node.get_logger()


class Ros2GetTopicsNamesAndTypesTool(Ros2BaseTool):
    name: str = "Ros2GetTopicsNamesAndTypes"
    description: str = "A tool for getting all ros2 topics names and types"

    def _run(self):
        return self.node.get_topic_names_and_types()


class Ros2GetRobotInterfaces(Ros2BaseTool):
    name: str = "ros2_robot_interfaces"
    description: str = (
        "A tool for getting all ros2 robot interfaces: topics, services and actions"
    )

    def _run(self):
        return self.node.ros_discovery_info.dict()


class Ros2ShowMsgInterfaceTool(BaseTool):
    name: str = "Ros2ShowMsgInterface"
    description: str = """A tool for showing ros2 message interface in json format.
    usage:
    ```python
    ShowRos2MsgInterface.run({"msg_name": "geometry_msgs/msg/PoseStamped"})
    ```
    """

    args_schema: Type[Ros2MsgInterfaceInput] = Ros2MsgInterfaceInput

    def _run(self, msg_name: str):
        """Show ros2 message interface in json format."""
        msg_cls: Type = rosidl_runtime_py.utilities.get_interface(msg_name)
        try:
            msg_dict: OrderedDict = rosidl_runtime_py.convert.message_to_ordereddict(
                msg_cls()
            )
            return json.dumps(msg_dict)
        except NotImplementedError:
            # For action classes that can't be instantiated
            goal_dict: OrderedDict = rosidl_runtime_py.convert.message_to_ordereddict(
                msg_cls.Goal()
            )

            result_dict: OrderedDict = rosidl_runtime_py.convert.message_to_ordereddict(
                msg_cls.Result()
            )

            feedback_dict: OrderedDict = (
                rosidl_runtime_py.convert.message_to_ordereddict(msg_cls.Feedback())
            )
            return json.dumps(
                {"goal": goal_dict, "result": result_dict, "feedback": feedback_dict}
            )


class Ros2PubMessageTool(Ros2BaseTool):
    name: str = "PubRos2MessageTool"
    description: str = """A tool for publishing a message to a ros2 topic

    By default 10 messages are published for 1 second. If you want to publish multiple messages, you can specify 'rate' and 'timeout_sec'.
    Example usage:

    ```python
    tool = Ros2PubMessageTool()
    tool.run(
        {
            "topic_name": "/some_topic",
            "msg_type": "geometry_msgs/Point",
            "msg_args": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rate" : 10,
            "timeout_sec" : 1
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
        rosidl_runtime_py.set_message.set_message_fields(msg, msg_args)
        return msg, msg_cls

    def _run(
        self,
        topic_name: str,
        msg_type: str,
        msg_args: Dict[str, Any],
        rate: int = 10,
        timeout_seconds: int = 1,
    ):
        """Publishes a message to the specified topic."""
        if "/msg/" not in msg_type:
            raise ValueError("msg_name must contain 'msg' in the name.")
        msg, msg_cls = self._build_msg(msg_type, msg_args)

        publisher = self.node.create_publisher(
            msg_cls, topic_name, 10
        )  # TODO(boczekbartek): infer qos profile from topic info

        def callback():
            publisher.publish(msg)
            self.logger.info(f"Published message '{msg}' to topic '{topic_name}'")

        ts = time.perf_counter()
        timer = self.node.create_timer(1.0 / rate, callback)

        while time.perf_counter() - ts < timeout_seconds:
            time.sleep(0.1)

        timer.cancel()
        timer.destroy()

        self.logger.info(
            f"Published messages for {timeout_seconds}s to topic '{topic_name}' with rate {rate}"
        )
        return


class TopicInput(Ros2BaseInput):
    topic_name: str = Field(..., description="Ros2 topic name")


class GetMsgFromTopic(Ros2BaseTool):
    name: str = "get_msg_from_topic"
    description: str = "Get message from topic"
    args_schema: Type[TopicInput] = TopicInput
    response_format: str = "content_and_artifact"

    def _run(self, topic_name: str):
        msg = self.node.get_raw_message_from_topic(topic_name)
        if type(msg) is sensor_msgs.msg.Image:
            img = convert_ros_img_to_base64(msg)
            return "Got image", {"images": [img]}
        else:
            return str(msg), {}


class Ros2GenericServiceCallerInput(BaseModel):
    service_name: str = Field(..., description="Name of the ROS2 service to call")
    service_type: str = Field(
        ..., description="Type of the ROS2 service in typical ros2 format"
    )
    request_args: Dict[str, Any] = Field(
        ..., description="Arguments for the service request"
    )


class Ros2GenericServiceCaller(Ros2BaseTool):
    name: str = "Ros2GenericServiceCaller"
    description: str = "A tool for calling any ROS2 service dynamically."

    args_schema: Type[Ros2GenericServiceCallerInput] = Ros2GenericServiceCallerInput

    def _build_request(self, service_type: str, request_args: Dict[str, Any]) -> Any:
        srv_module, _, srv_name = service_type.split("/")
        srv_class = getattr(importlib.import_module(f"{srv_module}.srv"), srv_name)
        request = srv_class.Request()
        rosidl_runtime_py.set_message.set_message_fields(request, request_args)
        return request

    def _run(self, service_name: str, service_type: str, request_args: Dict[str, Any]):
        if not service_name.startswith("/"):
            service_name = f"/{service_name}"

        try:
            request = self._build_request(service_type, request_args)
        except Exception as e:
            return f"Failed to build service request: {e}"
        namespaced_type = get_namespaced_type(service_type)
        client = self.node.create_client(
            rosidl_runtime_py.import_message.import_message_from_namespaced_type(
                namespaced_type
            ),
            service_name,
        )

        if not client.wait_for_service(timeout_sec=1.0):
            return f"Service '{service_name}' is not available"

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result() is not None:
            return str(future.result())
        else:
            return f"Service call to '{service_name}' failed"


class GetCameraImage(Ros2BaseTool):
    name: str = "get_camera_image"
    description: str = "get image from robots camera"
    response_format: str = "content_and_artifact"
    args_schema: Type[TopicInput] = TopicInput

    def _run(self, topic_name: str):
        msg = self.node.get_raw_message_from_topic(topic_name)
        img = convert_ros_img_to_base64(msg)
        return "Got image", {"images": [img]}
