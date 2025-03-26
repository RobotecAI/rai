# Copyright (C) 2025 Robotec.AI
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
import time
import uuid
from typing import Any, Callable, List, Optional, Tuple

import rclpy
import rclpy.executors
import rclpy.node
import rclpy.time
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tf2_ros import Buffer, LookupException, TransformListener, TransformStamped

from rai.communication import ARIConnector
from rai.communication.ros2.api import (
    ROS2ActionAPI,
    ROS2ServiceAPI,
    ROS2TopicAPI,
)
from rai.communication.ros2.messages import ROS2ARIMessage


class ROS2ARIConnector(ARIConnector[ROS2ARIMessage]):
    def __init__(
        self, node_name: str = f"rai_ros2_ari_connector_{str(uuid.uuid4())[-12:]}"
    ):
        super().__init__()
        self._node = Node(node_name)
        self._topic_api = ROS2TopicAPI(self._node)
        self._service_api = ROS2ServiceAPI(self._node)
        self._actions_api = ROS2ActionAPI(self._node)
        self._tf_buffer = Buffer(node=self._node)
        self.tf_listener = TransformListener(self._tf_buffer, self._node)

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin)
        self._thread.start()

    def get_topics_names_and_types(self) -> List[Tuple[str, List[str]]]:
        return self._topic_api.get_topic_names_and_types()

    def send_message(
        self,
        message: ROS2ARIMessage,
        target: str,
        *,
        msg_type: str,  # TODO: allow msg_type to be None, add auto topic type detection
        auto_qos_matching: bool = True,
        qos_profile: Optional[QoSProfile] = None,
        **kwargs: Any,
    ):
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
        *,
        msg_type: Optional[str] = None,
        auto_topic_type: bool = True,
        **kwargs: Any,
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
        self,
        message: ROS2ARIMessage,
        target: str,
        timeout_sec: float = 1.0,
        *,
        msg_type: str,
        **kwargs: Any,
    ) -> ROS2ARIMessage:
        msg = self._service_api.call_service(
            service_name=target,
            service_type=msg_type,
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
        *,
        msg_type: str,
        **kwargs: Any,
    ) -> str:
        if not isinstance(action_data, ROS2ARIMessage):
            raise ValueError("Action data must be of type ROS2ARIMessage")
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

    @staticmethod
    def wait_for_transform(
        tf_buffer: Buffer,
        target_frame: str,
        source_frame: str,
        timeout_sec: float = 1.0,
    ) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            if tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time()):
                return True
            time.sleep(0.1)
        return False

    def get_transform(
        self,
        target_frame: str,
        source_frame: str,
        timeout_sec: float = 5.0,
    ) -> TransformStamped:
        transform_available = self.wait_for_transform(
            self._tf_buffer, target_frame, source_frame, timeout_sec
        )
        if not transform_available:
            raise LookupException(
                f"Could not find transform from {source_frame} to {target_frame} in {timeout_sec} seconds"
            )
        transform: TransformStamped = self._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=int(timeout_sec)),
        )

        return transform

    def terminate_action(self, action_handle: str, **kwargs: Any):
        self._actions_api.terminate_goal(action_handle)

    @property
    def node(self) -> Node:
        return self._node

    def shutdown(self):
        self.tf_listener.unregister()
        self._node.destroy_node()
        self._actions_api.shutdown()
        self._topic_api.shutdown()
        self._executor.shutdown()
        self._thread.join()
