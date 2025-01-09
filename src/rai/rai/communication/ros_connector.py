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

import atexit
import threading
from typing import Callable, Dict, Optional

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from rai.communication.base_connector import BaseConnector, BaseMessage
from rai.tools.ros.utils import import_message_from_str
from rai.tools.utils import wait_for_message


class ROS2Connector(BaseConnector):
    def __init__(
        self,
        node_name: str = "rai_ros2_connector",
        qos_profile: Optional[QoSProfile] = None,
    ):
        if not rclpy.ok():
            rclpy.init()

        self.node = Node(node_name=node_name)
        self.publishers: Dict[str, Publisher] = {}
        self.qos_profile = qos_profile or QoSProfile(
            depth=10,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.SYSTEM_DEFAULT,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(target=self.executor.spin)
        self.executor_thread.start()
        atexit.register(self.cleanup)

    def send_message(self, msg: BaseMessage, target: str) -> None:
        publisher = self.publishers.get(target)
        if publisher is None:
            self.publishers[target] = self.node.create_publisher(
                msg.msg_type, target, qos_profile=self.qos_profile
            )
        self.publishers[target].publish(msg.content)

    def _validate_and_get_msg_type(self, topic: str):
        """
        Validate that the topic exists and return the message type.
        """
        topic_names_and_types = self.node.get_topic_names_and_types()
        topic_names = [topic for topic, _ in topic_names_and_types]
        if topic not in topic_names:
            raise ValueError(
                f"Topic '{topic}' not found. Available topics: {topic_names}"
            )
        return topic_names_and_types[topic_names.index(topic)][1][0]

    def receive_message(self, source: str) -> BaseMessage:
        msg_type = self._validate_and_get_msg_type(source)
        status, msg = wait_for_message(
            import_message_from_str(msg_type),
            self.node,
            source,
            qos_profile=self.qos_profile,
        )

        if status:
            return BaseMessage(content=msg)
        else:
            raise ValueError(f"No message found for {source}")

    def start_action(
        self, target: str, on_feedback: Callable, on_finish: Callable = lambda _: None
    ) -> str:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not suport starting actions"
        )

    def terminate_action(self, action_handle: str):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not suport terminating actions"
        )

    def send_and_wait(self, target: str) -> BaseMessage:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not suport sending messages"
        )

    def cleanup(self):
        self.executor.shutdown()
        self.executor_thread.join()
        self.node.destroy_node()
