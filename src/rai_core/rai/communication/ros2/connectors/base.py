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
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import rclpy
import rclpy.executors
import rclpy.node
import rclpy.time
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from tf2_ros import Buffer, LookupException, TransformListener, TransformStamped

from rai.communication import BaseConnector
from rai.communication.ros2.api import (
    ROS2ActionAPI,
    ROS2ServiceAPI,
    ROS2TopicAPI,
)
from rai.communication.ros2.connectors.action_mixin import ROS2ActionMixin
from rai.communication.ros2.connectors.service_mixin import ROS2ServiceMixin
from rai.communication.ros2.messages import ROS2Message

T = TypeVar("T", bound=ROS2Message)


class ROS2BaseConnector(ROS2ActionMixin, ROS2ServiceMixin, BaseConnector[T]):
    """ROS2-specific implementation of the ARIConnector.

    This connector provides functionality for ROS2 communication through topics,
    services, and actions, as well as TF (Transform) operations.

    Parameters
    ----------
    node_name : str, optional
        Name of the ROS2 node. If not provided, generates a unique name with UUID.
    destroy_subscribers : bool, optional
        Whether to destroy subscribers after receiving a message, by default False.

    Methods
    -------
    get_topics_names_and_types()
        Get list of available topics and their message types.
    get_services_names_and_types()
        Get list of available services and their types.
    get_actions_names_and_types()
        Get list of available actions and their types.
    send_message(message, target, msg_type, auto_qos_matching=True, qos_profile=None, **kwargs)
        Send a message to a specified topic.
    receive_message(source, timeout_sec=1.0, msg_type=None, auto_topic_type=True, **kwargs)
        Receive a message from a specified topic.
    wait_for_transform(tf_buffer, target_frame, source_frame, timeout_sec=1.0)
        Wait for a transform to become available.
    get_transform(target_frame, source_frame, timeout_sec=5.0)
        Get the transform between two frames.
    create_service(service_name, on_request, on_done=None, service_type, **kwargs)
        Create a ROS2 service.
    create_action(action_name, generate_feedback_callback, action_type, **kwargs)
        Create a ROS2 action server.
    shutdown()
        Clean up resources and shut down the connector.

    Notes
    -----
    Threading Model:
        The connector creates a MultiThreadedExecutor that runs in a dedicated thread.
        This executor processes all ROS2 callbacks and operations asynchronously.

    Subscriber Lifecycle:
        The `destroy_subscribers` parameter controls subscriber cleanup behavior:
        - True: Subscribers are destroyed after receiving a message
            - Pros: Better resource utilization
            - Cons: Known stability issues (see: https://github.com/ros2/rclpy/issues/1142)
        - False (default): Subscribers remain active after message reception
            - Pros: More stable operation, avoids potential crashes
            - Cons: May lead to memory/performance overhead from inactive subscribers
    """

    def __init__(
        self,
        node_name: str = f"rai_ros2_connector_{str(uuid.uuid4())[-12:]}",
        destroy_subscribers: bool = False,
    ):
        super().__init__()
        self._node = Node(node_name)
        self._topic_api = ROS2TopicAPI(self._node, destroy_subscribers)
        self._service_api = ROS2ServiceAPI(self._node)
        self._actions_api = ROS2ActionAPI(self._node)
        self._tf_buffer = Buffer(node=self._node)
        self._tf_listener = TransformListener(self._tf_buffer, self._node)

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin)
        self._thread.start()

        # cache for last received messages
        self.last_msg: Dict[str, T] = {}

    def last_message_callback(self, source: str, msg: T):
        self.last_msg[source] = msg

    def get_topics_names_and_types(self) -> List[Tuple[str, List[str]]]:
        return self._topic_api.get_topic_names_and_types()

    def get_services_names_and_types(self) -> List[Tuple[str, List[str]]]:
        return self._service_api.get_service_names_and_types()

    def get_actions_names_and_types(self) -> List[Tuple[str, List[str]]]:
        return self._actions_api.get_action_names_and_types()

    def send_message(
        self,
        message: T,
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

    def general_callback_preprocessor(self, message: Any):
        return self.T_class(payload=message, metadata={"msg_type": str(type(message))})

    def register_callback(
        self,
        source: str,
        callback: Callable[[T | Any], None],
        raw: bool = False,
        *,
        msg_type: Optional[str] = None,
        qos_profile: Optional[QoSProfile] = None,
        auto_qos_matching: bool = True,
        **kwargs: Any,
    ) -> str:
        exists = self._topic_api.subscriber_exists(source)
        if not exists:
            self._topic_api.create_subscriber(
                topic=source,
                msg_type=msg_type,
                callback=partial(self.general_callback, source),
                qos_profile=qos_profile,
                auto_qos_matching=auto_qos_matching,
            )
        return super().register_callback(source, callback, raw=raw)

    def receive_message(
        self,
        source: str,
        timeout_sec: float = 1.0,
        *,
        msg_type: Optional[str] = None,
        qos_profile: Optional[QoSProfile] = None,
        auto_qos_matching: bool = True,
        **kwargs: Any,
    ) -> T:
        if self._topic_api.subscriber_exists(source):
            # trying to hit cache first
            if source in self.last_msg:
                if self.last_msg[source].timestamp > time.time() - timeout_sec:
                    return self.last_msg[source]
        else:
            self._topic_api.create_subscriber(
                topic=source,
                callback=partial(self.general_callback, source),
                msg_type=msg_type,
                qos_profile=qos_profile,
                auto_qos_matching=auto_qos_matching,
            )
            self.register_callback(source, partial(self.last_message_callback, source))

        start_time = time.time()
        # wait for the message to be received
        while time.time() - start_time < timeout_sec:
            if source in self.last_msg:
                return self.last_msg[source]
            time.sleep(0.1)
        else:
            raise TimeoutError(
                f"Message from {source} not received in {timeout_sec} seconds"
            )

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

    def create_service(
        self,
        service_name: str,
        on_request: Callable,
        on_done: Optional[Callable] = None,
        *,
        service_type: str,
        **kwargs: Any,
    ) -> str:
        return self._service_api.create_service(
            service_name=service_name,
            callback=on_request,
            service_type=service_type,
            **kwargs,
        )

    def create_action(
        self,
        action_name: str,
        generate_feedback_callback: Callable,
        *,
        action_type: str,
        **kwargs: Any,
    ) -> str:
        return self._actions_api.create_action_server(
            action_name=action_name,
            action_type=action_type,
            execute_callback=generate_feedback_callback,
            **kwargs,
        )

    @property
    def node(self) -> Node:
        return self._node

    def shutdown(self):
        self._tf_listener.unregister()
        self._node.destroy_node()
        self._actions_api.shutdown()
        self._topic_api.shutdown()
        self._executor.shutdown()
        self._thread.join()
