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

import threading
from typing import Any, Callable, Optional, TypedDict

from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile

from rai.communication.ari_connector import ARIConnector, ARIMessage
from rai.communication.ros2.api import ROS2ActionAPI, ROS2ServiceAPI, ROS2TopicAPI


class ROS2ARIMessageMetadata(TypedDict):
    msg_type: str
    qos_profile: Optional[QoSProfile]
    auto_qos_matching: bool


class ROS2ARIMessage(ARIMessage):
    # TODO: Resolve reportIncompatibleVariableOverride
    metadata: ROS2ARIMessageMetadata


class ROS2ARIConnector(ARIConnector[ROS2ARIMessage]):
    def __init__(self, node_name: str = "rai_ros2_ari_connector"):
        super().__init__()
        self._node = Node(node_name)
        self._topic_api = ROS2TopicAPI(self._node)
        self._service_api = ROS2ServiceAPI(self._node)
        self._actions_api = ROS2ActionAPI(self._node)

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin)
        self._thread.start()

    def send_message(self, message: ROS2ARIMessage, target: str):
        self._topic_api.publish(
            topic=target, msg_content=message.payload, **message.metadata
        )

    def receive_message(self, source: str, timeout_sec: float = 1.0) -> ROS2ARIMessage:
        msg = self._topic_api.receive(topic=source, timeout_sec=timeout_sec)
        return ROS2ARIMessage(
            payload=msg, metadata={"msg_type": str(type(msg)), "topic": source}
        )

    def service_call(
        self, message: ROS2ARIMessage, target: str, timeout_sec: float = 1.0
    ) -> ROS2ARIMessage:
        msg = self._service_api.call_service(
            service_name=target,
            service_type=message.metadata["msg_type"],
            request=message.payload,
            timeout_sec=timeout_sec,
        )
        return ROS2ARIMessage(
            payload=msg, metadata={"msg_type": str(type(msg)), "service": target}
        )

    def start_action(
        self,
        action_data: Optional[ROS2ARIMessage],
        target: str,
        on_feedback: Callable[[Any], None] = lambda _: None,
        on_done: Callable[[Any], None] = lambda _: None,
        timeout_sec: float = 1.0,
    ) -> str:
        if not isinstance(action_data, ROS2ARIMessage):
            raise ValueError("Action data must be of type ROS2ARIMessage")

        accepted, handle = self._actions_api.send_goal(
            action_name=target,
            action_type=action_data.metadata["msg_type"],
            goal=action_data.payload,
            timeout_sec=timeout_sec,
            feedback_callback=on_feedback,
            done_callback=on_done,
        )
        if not accepted:
            raise RuntimeError("Action goal was not accepted")
        return handle

    def terminate_action(self, action_handle: str):
        self._actions_api.terminate_goal(action_handle)

    def shutdown(self):
        self._cleanup()

    def _cleanup(self):
        self._executor.shutdown()
        self._thread.join()
        self._node.destroy_node()
