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

import logging
import time
import uuid
from functools import partial
from typing import Annotated, Any, Dict, List, Optional, Tuple, Type, TypedDict, cast

import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.qos
import rclpy.subscription
import rclpy.task
import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities
from action_msgs.srv import CancelGoal
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.publisher import Publisher
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    LivelinessPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.task import Future
from rclpy.topic_endpoint_info import TopicEndpointInfo

from rai.tools.ros.utils import import_message_from_str, wait_for_message


def adapt_requests_to_offers(publisher_info: List[TopicEndpointInfo]) -> QoSProfile:
    if not publisher_info:
        return QoSProfile(depth=1)

    num_endpoints = len(publisher_info)
    reliability_reliable_count = 0
    durability_transient_local_count = 0

    for endpoint in publisher_info:
        profile = endpoint.qos_profile
        if profile.reliability == ReliabilityPolicy.RELIABLE:
            reliability_reliable_count += 1
        if profile.durability == DurabilityPolicy.TRANSIENT_LOCAL:
            durability_transient_local_count += 1

    request_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        liveliness=LivelinessPolicy.AUTOMATIC,
    )

    # Set reliability based on publisher offers
    if reliability_reliable_count == num_endpoints:
        request_qos.reliability = ReliabilityPolicy.RELIABLE
    else:
        if reliability_reliable_count > 0:
            logging.warning(
                "Some, but not all, publishers are offering RELIABLE reliability. "
                "Falling back to BEST_EFFORT as it will connect to all publishers. "
                "Some messages from Reliable publishers could be dropped."
            )
        request_qos.reliability = ReliabilityPolicy.BEST_EFFORT

    # Set durability based on publisher offers
    if durability_transient_local_count == num_endpoints:
        request_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    else:
        if durability_transient_local_count > 0:
            logging.warning(
                "Some, but not all, publishers are offering TRANSIENT_LOCAL durability. "
                "Falling back to VOLATILE as it will connect to all publishers. "
                "Previously-published latched messages will not be retrieved."
            )
        request_qos.durability = DurabilityPolicy.VOLATILE

    return request_qos


def build_ros2_msg(msg_type: str, msg_args: Dict[str, Any]) -> object:
    """Build a ROS2 message instance from type string and content dictionary."""
    msg_cls = import_message_from_str(msg_type)
    msg = msg_cls()
    rosidl_runtime_py.set_message.set_message_fields(msg, msg_args)
    return msg


def build_ros2_service_request(
    service_type: str, service_request_args: Dict[str, Any]
) -> Tuple[object, Type[Any]]:
    msg_cls = import_message_from_str(service_type)
    msg = msg_cls.Request()
    rosidl_runtime_py.set_message.set_message_fields(msg, service_request_args)
    return msg, msg_cls  # type: ignore


class ROS2TopicAPI:
    """Handles ROS2 topic operations including publishing and subscribing to messages.

    This class provides a high-level interface for ROS2 topic operations with automatic
    QoS profile matching and proper resource management.

    Attributes:
        node: The ROS2 node instance
        logger: Logger instance for this class
        _publishers: Dictionary mapping topic names to their publisher instances
    """

    def __init__(self, node: rclpy.node.Node) -> None:
        """Initialize the ROS2 topic API.

        Args:
            node: ROS2 node instance to use for communication
        """
        self._node = node
        self._logger = node.get_logger()
        self._publishers: Dict[str, Publisher] = {}

    def get_topic_names_and_types(
        self, no_demangle: bool = False
    ) -> List[Tuple[str, List[str]]]:
        """Get list of available topics and their types.

        Returns:
            List of tuples containing (topic_name, list_of_types)
        """
        return self._node.get_topic_names_and_types(no_demangle=no_demangle)

    def publish(
        self,
        topic: str,
        msg_content: Dict[str, Any],
        msg_type: str,
        *,  # Force keyword arguments
        auto_qos_matching: bool = True,
        qos_profile: Optional[QoSProfile] = None,
    ) -> None:
        """Publish a message to a ROS2 topic.

        Args:
            topic: Name of the topic to publish to
            msg_content: Dictionary containing the message content
            msg_type: ROS2 message type as string (e.g. 'std_msgs/msg/String')
            auto_qos_matching: Whether to automatically match QoS with subscribers
            qos_profile: Optional custom QoS profile to use

        Raises:
            ValueError: If neither auto_qos_matching is True nor qos_profile is provided
        """
        qos_profile = self._resolve_qos_profile(
            topic, auto_qos_matching, qos_profile, for_publisher=True
        )

        msg = build_ros2_msg(msg_type, msg_content)
        publisher = self._get_or_create_publisher(topic, type(msg), qos_profile)
        publisher.publish(msg)

    def receive(
        self,
        topic: str,
        msg_type: str,
        *,  # Force keyword arguments
        timeout_sec: float = 1.0,
        auto_qos_matching: bool = True,
        qos_profile: Optional[QoSProfile] = None,
    ) -> Any:
        """Receive a single message from a ROS2 topic.

        Args:
            topic: Name of the topic to receive from
            msg_type: ROS2 message type as string
            timeout_sec: How long to wait for a message
            auto_qos_matching: Whether to automatically match QoS with publishers
            qos_profile: Optional custom QoS profile to use

        Returns:
            The received message

        Raises:
            ValueError: If no publisher exists or no message is received within timeout
        """
        self._verify_publisher_exists(topic)

        qos_profile = self._resolve_qos_profile(
            topic, auto_qos_matching, qos_profile, for_publisher=False
        )

        msg_cls = self._get_message_class(msg_type)
        success, msg = wait_for_message(
            msg_cls,
            self._node,
            topic,
            qos_profile=qos_profile,
            time_to_wait=int(timeout_sec),
        )

        if not success:
            raise ValueError(
                f"No message received from topic: {topic} within {timeout_sec} seconds"
            )
        return msg

    def _get_or_create_publisher(
        self, topic: str, msg_cls: Type[Any], qos_profile: QoSProfile
    ) -> Publisher:
        """Get existing publisher or create a new one if it doesn't exist."""
        if topic not in self._publishers:
            self._publishers[topic] = self._node.create_publisher(  # type: ignore
                msg_cls, topic, qos_profile=qos_profile
            )
        return self._publishers[topic]

    def _resolve_qos_profile(
        self,
        topic: str,
        auto_qos_matching: bool,
        qos_profile: Optional[QoSProfile],
        for_publisher: bool,
    ) -> QoSProfile:
        """Resolve which QoS profile to use based on settings and existing endpoints."""
        if auto_qos_matching and qos_profile is not None:
            self._logger.warning(  # type: ignore
                "Auto QoS matching is enabled, but qos_profile is provided. "
                "Using provided qos_profile."
            )
            return qos_profile

        if auto_qos_matching:
            endpoint_info = (
                self._node.get_subscriptions_info_by_topic(topic)
                if for_publisher
                else self._node.get_publishers_info_by_topic(topic)
            )
            return adapt_requests_to_offers(endpoint_info)

        if qos_profile is not None:
            return qos_profile

        raise ValueError(
            "Either auto_qos_matching must be True or qos_profile must be provided"
        )

    @staticmethod
    def _get_message_class(msg_type: str) -> Type[Any]:
        """Convert message type string to actual message class."""
        return import_message_from_str(msg_type)

    def _verify_publisher_exists(self, topic: str) -> None:
        """Verify that at least one publisher exists for the given topic.

        Raises:
            ValueError: If no publisher exists for the topic
        """
        if not self._node.get_publishers_info_by_topic(topic):
            raise ValueError(f"No publisher found for topic: {topic}")

    def __del__(self) -> None:
        """Cleanup publishers when object is destroyed."""
        for publisher in self._publishers.values():
            publisher.destroy()


class ROS2ServiceAPI:
    """Handles ROS2 service operations including calling services."""

    def __init__(self, node: rclpy.node.Node) -> None:
        self.node = node
        self._logger = node.get_logger()

    def call_service(
        self,
        service_name: str,
        service_type: str,
        request: Any,
        timeout_sec: float = 1.0,
    ) -> Any:
        """
        Call a ROS2 service.

        Args:
            service_name: Name of the service to call
            service_type: ROS2 service type as string
            request: Request message content

        Returns:
            The response message
        """
        srv_msg, srv_cls = build_ros2_service_request(service_type, request)
        service_client = self.node.create_client(srv_cls, service_name)  # type: ignore
        client_ready = service_client.wait_for_service(timeout_sec=timeout_sec)
        if not client_ready:
            raise ValueError(
                f"Service {service_name} not ready within {timeout_sec} seconds. "
                "Try increasing the timeout or check if the service is running."
            )
        return service_client.call(srv_msg)


class ROS2ActionData(TypedDict):
    action_client: Optional[ActionClient]
    goal_future: Optional[rclpy.task.Future]
    result_future: Optional[rclpy.task.Future]
    client_goal_handle: Optional[ClientGoalHandle]
    feedbacks: List[Any]


class ROS2ActionAPI:
    def __init__(self, node: rclpy.node.Node) -> None:
        self.node = node
        self._logger = node.get_logger()
        self.actions: Dict[str, ROS2ActionData] = {}

    def _generate_handle(self):
        return str(uuid.uuid4())

    def _generic_callback(self, handle: str, feedback_msg: Any) -> None:
        self.actions[handle]["feedbacks"].append(feedback_msg.feedback)

    def send_goal(
        self,
        action_name: str,
        action_type: str,
        goal: Dict[str, Any],
        timeout_sec: float = 1.0,
    ) -> Tuple[bool, Annotated[str, "action handle"]]:
        handle = self._generate_handle()
        self.actions[handle] = ROS2ActionData(
            action_client=None,
            goal_future=None,
            result_future=None,
            client_goal_handle=None,
            feedbacks=[],
        )

        action_cls = import_message_from_str(action_type)
        action_goal = action_cls.Goal()  # type: ignore
        rosidl_runtime_py.set_message.set_message_fields(action_goal, goal)

        action_client = ActionClient(self.node, action_cls, action_name)
        if not action_client.wait_for_server(timeout_sec=timeout_sec):  # type: ignore
            return False, ""

        send_goal_future: Future = action_client.send_goal_async(
            goal=action_goal, feedback_callback=partial(self._generic_callback, handle)
        )
        self.actions[handle]["action_client"] = action_client
        self.actions[handle]["goal_future"] = send_goal_future

        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            if send_goal_future.done():
                break
            time.sleep(0.01)

        goal_handle = cast(Optional[ClientGoalHandle], send_goal_future.result())
        if goal_handle is None:
            return False, ""

        get_result_future = cast(Future, goal_handle.get_result_async())  # type: ignore

        self.actions[handle]["result_future"] = get_result_future
        self.actions[handle]["client_goal_handle"] = goal_handle

        return goal_handle.accepted, handle  # type: ignore

    def terminate_goal(self, handle: str) -> CancelGoal.Response:
        if self.actions[handle]["client_goal_handle"] is None:
            raise ValueError(
                f"Cannot terminate goal {handle} as it was not accepted or has no goal handle."
            )
        return self.actions[handle]["client_goal_handle"].cancel_goal()

    def get_feedback(self, handle: str) -> List[Any]:
        return self.actions[handle]["feedbacks"]

    def is_goal_done(self, handle: str) -> bool:
        if handle not in self.actions:
            raise ValueError(f"Invalid action handle: {handle}")
        if self.actions[handle]["result_future"] is None:
            raise ValueError(
                f"Result future is None for handle: {handle}. " "Was the goal accepted?"
            )
        return self.actions[handle]["result_future"].done()

    def get_result(self, handle: str) -> Any:
        if not self.is_goal_done(handle):
            raise ValueError(f"Goal {handle} is not done")
        if self.actions[handle]["result_future"] is None:
            raise ValueError(f"No result available for goal {handle}")
        return self.actions[handle]["result_future"].result()
