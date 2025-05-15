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

import time
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)

import rclpy
import rclpy.node
from rclpy.publisher import Publisher
from rclpy.qos import (
    QoSProfile,
)
from rclpy.subscription import Subscription
from rclpy.topic_endpoint_info import TopicEndpointInfo

from rai.communication.ros2.api.base import (
    BaseROS2API,
)
from rai.communication.ros2.api.conversion import import_message_from_str


class ROS2TopicAPI(BaseROS2API):
    """Handles ROS2 topic operations including publishing and subscribing to messages.

    This class provides a high-level interface for ROS2 topic operations with automatic
    QoS profile matching and proper resource management.

    Attributes:
        node: The ROS2 node instance
        logger: Logger instance for this class
        _publishers: Dictionary mapping topic names to their publisher instances
    """

    def __init__(
        self, node: rclpy.node.Node, destroy_subscribers: bool = False
    ) -> None:
        """Initialize the ROS2 topic API.

        Args:
            node: ROS2 node instance to use for communication
        """
        self._node = node
        self._logger = node.get_logger()
        self._publishers: Dict[str, Publisher] = {}

        # TODO: These fields are a workaround to prevent subscriber destruction,
        # which often fails as described in https://github.com/ros2/rclpy/issues/1142
        # By setting destroy_subscriptions to False, subscribers are not destroyed.
        # While this may lead to memory/performance issues, it's preferable to
        # preventing node crashes.
        self._last_msg: Dict[str, Tuple[float, Any]] = {}
        self._subscriptions: Dict[str, rclpy.node.Subscription] = {}
        self._destroy_subscribers: bool = destroy_subscribers
        self.node = node
        self.subscriptions: Dict[str, Subscription] = {}
        self.publishers: Dict[str, Publisher] = {}

    def subscriber_exists(self, topic: str) -> bool:
        return topic in self.subscriptions

    def publisher_exists(self, topic: str) -> bool:
        return topic in self.publishers

    def create_subscriber(
        self,
        topic: str,
        callback: Callable[[Any], None],
        msg_type: Optional[str] = None,
        qos_profile: Optional[QoSProfile] = None,
        auto_qos_matching: bool = True,
    ):
        if msg_type is None:
            msg_type = self.get_topic_type(topic)
        msg_cls = self.import_message_from_str(msg_type)
        if auto_qos_matching:
            qos = self.adapt_requests_to_offers(
                self.node.get_publishers_info_by_topic(topic)
            )
        elif qos_profile is not None:
            qos = qos_profile
        else:
            raise ValueError("Either qos_profile or auto_qos_matching must be provided")
        subscription = self.node.create_subscription(
            topic=topic, msg_type=msg_cls, callback=callback, qos_profile=qos
        )
        self.subscriptions[topic] = subscription
        return subscription

    def create_publisher(
        self,
        topic: str,
        msg_type: str,
        qos_profile: Optional[QoSProfile] = None,
        auto_qos_matching: bool = True,
    ):
        msg_cls = self.import_message_from_str(msg_type)
        if auto_qos_matching:
            qos = self.adapt_requests_to_offers(
                self.node.get_subscriptions_info_by_topic(topic)
            )
        elif qos_profile is not None:
            qos = qos_profile
        else:
            raise ValueError("Either qos_profile or auto_qos_matching must be provided")
        publisher = self.node.create_publisher(
            topic=topic, msg_type=msg_cls, qos_profile=qos
        )
        self.publishers[topic] = publisher
        return publisher

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
        *,
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

        msg = self.build_ros2_msg(msg_type, msg_content)
        publisher = self._get_or_create_publisher(topic, type(msg), qos_profile)
        publisher.publish(msg)

    def _verify_receive_args(
        self, topic: str, auto_topic_type: bool, msg_type: Optional[str]
    ) -> None:
        if auto_topic_type and msg_type is not None:
            raise ValueError("Cannot provide both auto_topic_type and msg_type")
        if not auto_topic_type and msg_type is None:
            raise ValueError("msg_type must be provided if auto_topic_type is False")

    def _generic_callback(self, topic: str, msg: Any) -> None:
        self._last_msg[topic] = (time.time(), msg)

    def _is_topic_available(self, topic: str, timeout_sec: float) -> bool:
        ts = time.time()
        topic = topic if topic.startswith("/") else f"/{topic}"
        while time.time() - ts < timeout_sec:
            for topic_name, _ in self.get_topic_names_and_types():
                if topic == topic_name:
                    return True
            time.sleep(0.1)
        return False

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
            return self.adapt_requests_to_offers(endpoint_info)

        if qos_profile is not None:
            return qos_profile

        raise ValueError(
            "Either auto_qos_matching must be True or qos_profile must be provided"
        )

    @staticmethod
    def _get_message_class(msg_type: str) -> Type[Any]:
        """Convert message type string to actual message class."""
        return import_message_from_str(msg_type)

    def _verify_publisher_exists(self, topic: str) -> List[TopicEndpointInfo]:
        """Verify that at least one publisher exists for the given topic.

        Raises:
            ValueError: If no publisher exists for the topic
        """
        topic_endpoints = self._node.get_publishers_info_by_topic(topic)
        if not topic_endpoints:
            raise ValueError(f"No publisher found for topic: {topic}")
        return topic_endpoints

    def shutdown(self) -> None:
        """Cleanup publishers when object is destroyed."""
        for publisher in self._publishers.values():
            publisher.destroy()
