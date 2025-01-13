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

import importlib
import logging
from typing import Any, Callable, Dict, List, Sequence
from uuid import uuid4

import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.qos
import rclpy.subscription
import rclpy.task
import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities
from langchain_core.messages import BaseMessage as LangchainBaseMessage
from rclpy.action.client import ActionClient as ROS2ActionClient
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import qos_profile_default
from rosidl_runtime_py.utilities import get_namespaced_type
from std_msgs.msg import String

from rai.communication.hri_connector import HRIConnector, HRIMessage
from rai.communication.rri_connector import ROS2RRIMessage, RRIConnector, RRIMessage
from rai.messages.multimodal import MultimodalMessage as RAIMultimodalMessage
from rai.tools.ros.utils import import_message_from_str, wait_for_message


class ServiceCaller:
    node: Node

    def _service_call(
        self, message: ROS2RRIMessage, target: str, timeout_sec: float = 1.0
    ) -> ROS2RRIMessage:
        response = self.__service_call(
            target, message.ros_message_type, message.payload
        )
        if isinstance(response, str):
            return ROS2RRIMessage(
                payload=response, ros_message_type="str", python_message_class=str
            )
        else:
            return ROS2RRIMessage(
                payload=response,
                ros_message_type=message.ros_message_type,
                python_message_class=type(response),
            )

    def _build_request(self, service_type: str, request_args: Dict[str, Any]) -> Any:
        srv_module, _, srv_name = service_type.split("/")
        srv_class = getattr(importlib.import_module(f"{srv_module}.srv"), srv_name)
        request = srv_class.Request()
        rosidl_runtime_py.set_message.set_message_fields(request, request_args)
        return request

    def __service_call(  # TODO: refactor into single _ method
        self, service_name: str, service_type: str, request_args: Dict[str, Any]
    ) -> str | object:
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
            return future.result()
        else:
            return f"Service call to '{service_name}' failed"


class TopicSubscriber:
    node: Node

    def _receive_message(
        self, source: str, timeout_sec: float = 1.0
    ) -> str | object:  # TODO: ROS 2 msg type
        publishers_info = self.node.get_publishers_info_by_topic(topic_name=source)
        if len(publishers_info) == 0:
            return f"No publisher found for topic {source}."

        msg_type_str = publishers_info[0].topic_type
        if len({publisher_info.topic_type for publisher_info in publishers_info}) > 1:
            logging.warning(
                f"Multiple publishers on topic {source} with different message types. Will use the first one."
            )

        msg_type = import_message_from_str(msg_type_str)
        status, ros2_msg = wait_for_message(
            msg_type=msg_type,
            node=self.node,
            topic=source,
            time_to_wait=int(timeout_sec),
        )
        if status:
            return ros2_msg
        else:
            return (
                f"Message could not be received from {source}. Try increasing timeout."
            )


class TopicPublisher:
    node: Node
    publishers: Dict[str, Publisher] = {}

    def _publish(
        self,
        message: ROS2RRIMessage,
        topic: str,
    ):
        msg_python_type = import_message_from_str(message.ros_message_type)
        if topic not in self.publishers:
            # TODO: use qos matching when available
            self.publishers[topic] = self.node.create_publisher(
                msg_python_type, topic, qos_profile_default
            )
        ros2_msg = msg_python_type()
        rosidl_runtime_py.set_message.set_message_fields(ros2_msg, message.payload)
        self.publishers[topic].publish(ros2_msg)


class ActionClient:
    node: Node
    action_clients: Dict[str, ROS2ActionClient] = {}

    def _generate_handle(self) -> str:
        return str(uuid4())

    def _start_action(
        self,
        action: ROS2RRIMessage,
        target: str,
        on_feedback: Callable[[Any], None],
        on_done: Callable[[Any], None],
        timeout_sec: float = 1.0,
    ) -> str:
        raise NotImplementedError("Action client not implemented")

    def _terminate_action(self, action_handle: str) -> ROS2RRIMessage:
        raise NotImplementedError("Action termination not implemented")


class ROS2RRIConnector(
    RRIConnector,
    TopicPublisher,
    TopicSubscriber,
    ServiceCaller,
    ActionClient,
):
    def __init__(self, node: Node):
        self.node = node

    def send_message(
        self,
        message: ROS2RRIMessage,
        target: str,
    ):
        self._publish(message, target)

    def receive_message(self, source: str, timeout_sec: float = 1.0) -> ROS2RRIMessage:
        ros2_msg = self._receive_message(source, timeout_sec)
        if isinstance(ros2_msg, str):
            return ROS2RRIMessage(
                payload=ros2_msg, ros_message_type="str", python_message_class=str
            )
        else:
            return ROS2RRIMessage(
                payload=ros2_msg,
                ros_message_type=type(ros2_msg).__name__,
                python_message_class=type(ros2_msg),
            )

    def service_call(
        self, message: RRIMessage, target: str, timeout_sec: float = 1.0
    ) -> RRIMessage:
        if not isinstance(message, ROS2RRIMessage):
            raise TypeError("Message must be of type ROS2RRIMessage")
        return self._service_call(message, target, timeout_sec)

    def start_action(
        self,
        action: RRIMessage,
        target: str,
        on_feedback: Callable[[Any], None],
        on_done: Callable[[Any], None],
        timeout_sec: float = 1.0,
    ) -> str:
        if not isinstance(action, ROS2RRIMessage):
            raise TypeError("Action must be of type ROS2RRIMessage")
        return self._start_action(action, target, on_feedback, on_done, timeout_sec)

    def terminate_action(self, action_handle: str) -> ROS2RRIMessage:
        return self._terminate_action(action_handle)


class ROS2HRIConnector(HRIConnector, TopicSubscriber, TopicPublisher):
    def __init__(self, node: Node, sources: Sequence[str], targets: Sequence[str]):
        self.node = node
        self.sources = sources
        self.targets = targets

    def send_message(self, message: LangchainBaseMessage | RAIMultimodalMessage):
        for target in self.targets:
            hri_message = HRIMessage.from_langchain(message)
            ros2rri_message = ROS2RRIMessage(
                payload=hri_message.text,  # TODO: Only string topics
                ros_message_type=type(hri_message).__name__,
                python_message_class=type(hri_message),
            )
            self._publish(ros2rri_message, target)

    def receive_message(self) -> LangchainBaseMessage | RAIMultimodalMessage:
        messages: List[Dict[str, str]] = []
        for source in self.sources:
            ros2_msg = self._receive_message(source)
            if not isinstance(ros2_msg, String):
                raise ValueError(
                    f"Received message from {source} is not a string. Only string topics are supported."
                )
            messages.append(ros2_msg)
        return HRIMessage(text=str(messages), type="human").to_langchain()
