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
import uuid
from typing import Any, Callable, Dict, Optional, TypedDict

from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from rai.communication.ari_connector import ARIConnector, ARIMessage
from rai.communication.ros2.api import ROS2ActionAPI, ROS2ServiceAPI, ROS2TopicAPI


class ROS2ARIPayload(TypedDict):
    data: Any


class ROS2ARIMessage(ARIMessage):
    def __init__(
        self, payload: ROS2ARIPayload, metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(payload, metadata)


class ROS2ARIConnector(ARIConnector[ROS2ARIMessage]):
    def __init__(
        self, node_name: str = f"rai_ros2_ari_connector_{str(uuid.uuid4())[-12:]}"
    ):
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
        auto_qos_matching = message.metadata.get("auto_qos_matching", True)
        qos_profile = message.metadata.get("qos_profile", None)
        msg_type = message.metadata.get("msg_type", None)

        # TODO: allow msg_type to be None, add auto topic type detection
        if msg_type is None:
            raise ValueError("msg_type is required")

        self._topic_api.publish(
            topic=target,
            msg_content=message.payload,
            msg_type=msg_type,
            auto_qos_matching=auto_qos_matching,
            qos_profile=qos_profile,
        )

    def receive_message(
        self,
        source: str,
        timeout_sec: float = 1.0,
        msg_type: Optional[str] = None,
        auto_topic_type: bool = True,
    ) -> ROS2ARIMessage:
        msg = self._topic_api.receive(
            topic=source,
            timeout_sec=timeout_sec,
            msg_type=msg_type,
            auto_topic_type=auto_topic_type,
        )
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
        msg_type = action_data.metadata.get("msg_type", None)
        if msg_type is None:
            raise ValueError("msg_type is required")
        accepted, handle = self._actions_api.send_goal(
            action_name=target,
            action_type=msg_type,
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
        self._executor.shutdown()
        self._thread.join()
        self._actions_api.shutdown()
        self._topic_api.shutdown()
        self._node.destroy_node()
