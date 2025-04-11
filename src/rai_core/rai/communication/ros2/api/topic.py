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
from dataclasses import dataclass
from functools import partial
from queue import Queue
from threading import Lock
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
)

import rclpy
import rclpy.action
import rclpy.callback_groups
import rclpy.executors
import rclpy.node
import rclpy.qos
import rclpy.subscription
import rclpy.task
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.publisher import Publisher
from rclpy.qos import (
    QoSProfile,
)
from rclpy.topic_endpoint_info import TopicEndpointInfo

from rai.communication.ros2.api.base import (
    BaseROS2API,
    IROS2Message,
)
from rai.tools.ros2.utils import import_message_from_str


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

    def receive(
        self,
        topic: str,
        *,
        auto_topic_type: bool = True,
        msg_type: Optional[str] = None,
        timeout_sec: float = 1.0,
        auto_qos_matching: bool = True,
        qos_profile: Optional[QoSProfile] = None,
        retry_count: int = 3,
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
            ValueError: If auto_topic_type is False and msg_type is not provided
            ValueError: If auto_topic_type is True and msg_type is provided
        """
        self._verify_receive_args(topic, auto_topic_type, msg_type)
        topic_endpoints = self._verify_publisher_exists(topic)

        # TODO: Verify publishers topic type consistency
        if auto_topic_type:
            msg_type = topic_endpoints[0].topic_type
        else:
            if msg_type is None:
                raise ValueError(
                    "msg_type must be provided if auto_topic_type is False"
                )

        qos_profile = self._resolve_qos_profile(
            topic, auto_qos_matching, qos_profile, for_publisher=False
        )

        msg_cls = self._get_message_class(msg_type)
        if not self._is_topic_available(topic, timeout_sec):
            raise ValueError(
                f"Topic {topic} is not available within {timeout_sec} seconds. Check if the topic exists."
            )

        for _ in range(retry_count):
            success, msg = self._wait_for_message(
                msg_cls,
                self._node,
                topic,
                qos_profile=qos_profile,
                timeout_sec=timeout_sec / retry_count,
            )
            if success:
                return msg
        else:
            raise ValueError(
                f"No message received from topic: {topic} within {timeout_sec} seconds"
            )

    def _generic_callback(self, topic: str, msg: Any) -> None:
        self._last_msg[topic] = (time.time(), msg)

    def _wait_for_message_once(
        self,
        msg_cls: Type[Any],
        node: rclpy.node.Node,
        topic: str,
        qos_profile: QoSProfile,
        timeout_sec: float,
    ) -> Tuple[bool, Any]:
        ts = time.time()
        success = False
        msg = None

        def callback(received_msg: Any):
            nonlocal success, msg
            success = True
            msg = received_msg

        sub = node.create_subscription(
            msg_cls,
            topic,
            callback,
            qos_profile=qos_profile,
        )
        while not success and time.time() - ts < timeout_sec:
            time.sleep(0.01)
        node.destroy_subscription(sub)
        return success, msg

    def _wait_for_message_persistent(
        self,
        msg_cls: Type[Any],
        node: rclpy.node.Node,
        topic: str,
        qos_profile: QoSProfile,
        timeout_sec: float,
    ) -> Tuple[bool, Any]:
        if topic not in self._subscriptions:
            self._subscriptions[topic] = node.create_subscription(
                msg_cls,
                topic,
                partial(self._generic_callback, topic),
                qos_profile=qos_profile,
            )
        ts = time.time()
        while time.time() - ts < timeout_sec:
            if topic in self._last_msg:
                if self._last_msg[topic][0] + timeout_sec > time.time():
                    return True, self._last_msg[topic][1]
            time.sleep(0.01)
        return False, None

    def _wait_for_message(
        self,
        msg_cls: Type[Any],
        node: rclpy.node.Node,
        topic: str,
        qos_profile: QoSProfile,
        timeout_sec: float,
    ) -> Tuple[bool, Any]:
        if self._destroy_subscribers:
            return self._wait_for_message_once(
                msg_cls, node, topic, qos_profile, timeout_sec
            )
        return self._wait_for_message_persistent(
            msg_cls, node, topic, qos_profile, timeout_sec
        )

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


@dataclass
class TopicConfig:
    msg_type: str = "rai_interfaces/msg/HRIMessage"
    auto_qos_matching: bool = True
    qos_profile: Optional[QoSProfile] = None
    is_subscriber: bool = False
    # if queue_maxsize is not set, the queue will be unbounded
    # which may lead to memory issues for high-bandwidth topics
    queue_maxsize: Optional[int] = None
    # if queue_maxsize is provided, the overflow policy must be set
    overflow_policy: Optional[Literal["drop_oldest", "drop_newest"]] = None
    subscriber_callback: Optional[Callable[[IROS2Message], None]] = None
    source_author: Literal["human", "ai"] = "ai"

    def __post_init__(self):
        if not self.auto_qos_matching and self.qos_profile is None:
            raise ValueError(
                "Either 'auto_qos_matching' must be True or 'qos_profile' must be set."
            )


class ConfigurableROS2TopicAPI(ROS2TopicAPI):
    def __init__(self, node: rclpy.node.Node):
        super().__init__(node)
        self._subscribtions: dict[str, rclpy.node.Subscription] = {}
        self.callback_group = ReentrantCallbackGroup()
        self.topic_msg_queue: Dict[str, Queue[Any]] = {}
        self.topic_queue_locks: Dict[str, Lock] = {}
        self.topic_config: Dict[str, TopicConfig] = {}

    def _generic_callback(self, topic: str, msg: Any) -> None:
        """Handle incoming messages for a topic based on queue configuration.

        Args:
            topic: The topic name receiving the message
            msg: The received message

        Raises:
            ValueError: If an invalid overflow policy is configured
        """
        with self.topic_queue_locks[topic]:
            self._put_msg_in_queue(topic, msg)

    def _put_msg_in_queue(self, topic: str, msg: Any):
        queue = self.topic_msg_queue[topic]
        config = self.topic_config[topic]

        # Fast path for unbounded queues
        if config.queue_maxsize is None:
            queue.put(msg)
            return

        # Handle bounded queues with overflow policies
        if queue.full():
            if config.overflow_policy == "drop_oldest":
                queue.get()  # Remove oldest message
                queue.put(msg)
            elif config.overflow_policy == "drop_newest":
                return  # Silently drop the new message
            else:
                raise ValueError(
                    f"Invalid overflow policy for topic {topic}: {config.overflow_policy}"
                )
        else:
            queue.put(msg)

    def configure_publisher(self, topic: str, config: TopicConfig):
        if config.is_subscriber:
            raise ValueError(
                "Can't reconfigure publisher with subscriber config! Set config.is_subscriber to False"
            )
        qos_profile = self._resolve_qos_profile(
            topic, config.auto_qos_matching, config.qos_profile, for_publisher=True
        )
        if topic in self._publishers:
            flag = self._node.destroy_publisher(self._publishers[topic].handle)
            if not flag:
                raise ValueError(f"Failed to reconfigure existing publisher to {topic}")

        self._publishers[topic] = self._node.create_publisher(
            import_message_from_str(config.msg_type),
            topic=topic,
            qos_profile=qos_profile,
        )
        self.topic_config[topic] = config

    def configure_subscriber(
        self,
        topic: str,
        config: TopicConfig,
    ):
        if not config.is_subscriber:
            raise ValueError(
                "Can't reconfigure subscriber with publisher config! Set config.is_subscriber to True"
            )
        qos_profile = self._resolve_qos_profile(
            topic, config.auto_qos_matching, config.qos_profile, for_publisher=False
        )
        if topic in self._subscribtions:
            flag = self._node.destroy_subscription(self._subscribtions[topic])
            if not flag:
                raise ValueError(
                    f"Failed to reconfigure existing subscriber to {topic}"
                )

        msg_type = import_message_from_str(config.msg_type)
        self._subscribtions[topic] = self._node.create_subscription(
            msg_type=msg_type,
            topic=topic,
            callback=config.subscriber_callback
            or partial(self._generic_callback, topic),
            qos_profile=qos_profile,
            callback_group=self.callback_group,
        )
        if config.queue_maxsize is not None:
            self.topic_msg_queue[topic] = Queue(maxsize=config.queue_maxsize)
            if config.overflow_policy is None:
                raise ValueError(
                    "Overflow policy must be set if queue_maxsize is provided"
                )
        else:
            self.topic_msg_queue[topic] = Queue()
        self.topic_config[topic] = config
        self.topic_queue_locks[topic] = Lock()

    def publish_configured(self, topic: str, msg_content: dict[str, Any]) -> None:
        """Publish a message to a ROS2 topic.

        Args:
            topic: Name of the topic to publish to
            msg_content: Dictionary containing the message content

        Raises:
            ValueError: If topic has not been configured for publishing
        """
        try:
            publisher = self._publishers[topic]
        except Exception as e:
            raise ValueError(f"{topic} has not been configured for publishing") from e
        msg_type = publisher.msg_type
        msg = self.build_ros2_msg(msg_type, msg_content)  # type: ignore
        publisher.publish(msg)

    def receive(
        self,
        topic: str,
        *,
        auto_topic_type: bool = True,
        msg_type: Optional[str] = None,
        timeout_sec: float = 1.0,
        auto_qos_matching: bool = True,
        qos_profile: Optional[QoSProfile] = None,
        retry_count: int = 3,
    ) -> Any:
        """Receive a single message from a ROS2 topic's queue or by waiting for a new message.

        For topics with configured subscribers, retrieves the next message from the topic's queue.
        For unconfigured topics, falls back to the parent class behavior of waiting for a new message.

        Args:
            topic: Name of the topic to receive from
            auto_topic_type: If True, automatically detect message type from publishers (ignored for configured topics)
            msg_type: ROS2 message type as string (ignored for configured topics)
            timeout_sec: Maximum time to wait for a message in seconds
            auto_qos_matching: Whether to automatically match QoS with publishers (ignored for configured topics)
            qos_profile: Optional custom QoS profile to use (ignored for configured topics)
            retry_count: Number of attempts to receive a message (ignored for configured topics)

        Returns:
            The received message

        Raises:
            ValueError: If no message is available in the queue for configured topics
            ValueError: For unconfigured topics, inherits parent class exceptions
        """
        if topic not in self.topic_msg_queue:
            super().receive(
                topic,
                auto_topic_type=auto_topic_type,
                msg_type=msg_type,
                timeout_sec=timeout_sec,
                auto_qos_matching=auto_qos_matching,
                qos_profile=qos_profile,
                retry_count=retry_count,
            )
        else:
            ts = time.time()
            while time.time() - ts < timeout_sec:
                with self.topic_queue_locks[topic]:
                    if not self.topic_msg_queue[topic].empty():
                        msg = self.topic_msg_queue[topic].get()
                        return msg
                time.sleep(0.01)
            raise ValueError(
                f"No message received from topic: {topic} within {timeout_sec} seconds"
            )
