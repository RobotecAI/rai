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
from typing import Any, Dict, List, Optional, Tuple, Type

import rclpy
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.qos
import rclpy.subscription
import rclpy.task
import rosidl_runtime_py.set_message
import rosidl_runtime_py.utilities
from rclpy.publisher import Publisher
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    LivelinessPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
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

    def list_topics(self) -> List[Tuple[str, List[str]]]:
        """Get list of available topics and their types.

        Returns:
            List of tuples containing (topic_name, list_of_types)
        """
        return self._node.get_topic_names_and_types()

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
