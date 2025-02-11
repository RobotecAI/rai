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
import time
import uuid
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import rclpy
import rclpy.executors
import rclpy.node
import rclpy.time
import rosidl_runtime_py.convert
from cv_bridge import CvBridge
from PIL import Image
from pydub import AudioSegment
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image as ROS2Image
from tf2_ros import Buffer, LookupException, TransformListener, TransformStamped

from rai.communication import (
    ARIConnector,
    ARIMessage,
    HRIConnector,
    HRIMessage,
    HRIPayload,
)
from rai.communication.ros2.api import (
    ConfigurableROS2TopicAPI,
    ROS2ActionAPI,
    ROS2ServiceAPI,
    ROS2TopicAPI,
    TopicConfig,
)
from rai_interfaces.msg import HRIMessage as ROS2HRIMessage_
from rai_interfaces.msg._audio_message import (
    AudioMessage as ROS2HRIMessage__Audio,
)


class ROS2ARIMessage(ARIMessage):
    def __init__(self, payload: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a ROS2ARIMessage instance.
        
        This constructor initializes a ROS2ARIMessage by passing the given payload and optional metadata to the base ARIMessage class.
        
        Parameters:
            payload (Any): The content of the message.
            metadata (Optional[Dict[str, Any]]): Supplementary information for the message (default is None).
        """
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
        tf_buffer = Buffer(node=self._node)
        tf_listener = TransformListener(tf_buffer, self._node)
        transform_available = self.wait_for_transform(
            tf_buffer, target_frame, source_frame, timeout_sec
        )
        if not transform_available:
            raise LookupException(
                f"Could not find transform from {source_frame} to {target_frame} in {timeout_sec} seconds"
            )
        transform: TransformStamped = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rclpy.time.Time(),
            timeout=Duration(seconds=timeout_sec),
        )
        tf_listener.unregister()
        return transform

    def terminate_action(self, action_handle: str, **kwargs: Any):
        self._actions_api.terminate_goal(action_handle)

    def shutdown(self):
        self._executor.shutdown()
        self._thread.join()
        self._actions_api.shutdown()
        self._topic_api.shutdown()
        self._node.destroy_node()


class ROS2HRIMessage(HRIMessage):
    def __init__(self, payload: HRIPayload, message_author: Literal["ai", "human"]):
        """
        Initialize a ROS2HRIMessage instance with the specified payload and message author.
        
        Parameters:
            payload (HRIPayload): The message payload containing HRI data.
            message_author (Literal["ai", "human"]): Specifies the origin of the message, either "ai" or "human".
        """
        super().__init__(payload, message_author)

    @classmethod
    def from_ros2(cls, msg: Any, message_author: Literal["ai", "human"]):
        """
        Create a ROS2HRIMessage instance from a ROS2 HRI message.
        
        This class method converts a ROS2 message into a ROS2HRIMessage object. The conversion process includes:
          - Converting image messages from `msg.images` to OpenCV images using CvBridge and then to PIL Image objects.
          - Converting audio messages from `msg.audios` into AudioSegment objects with a fixed sample width.
          - Constructing a payload that bundles the text, converted images, and audio segments.
        
        Parameters:
            msg (Any): A ROS2 HRI message instance from `rai_interfaces.msg.HRIMessage` containing the attributes:
                       - text: A string message.
                       - images: A list of ROS2 image messages.
                       - audios: A list of ROS2 audio messages.
            message_author (Literal["ai", "human"]): Indicates the author of the message. Must be either "ai" or "human".
        
        Returns:
            ROS2HRIMessage: A new instance of ROS2HRIMessage with the converted payload and specified message_author.
        
        Raises:
            ValueError: If the provided message is not an instance of `rai_interfaces.msg.HRIMessage`.
        
        Example:
            >>> from rai_interfaces.msg import HRIMessage
            >>> ros2_msg = HRIMessage(text="Hello", images=[...], audios=[...])
            >>> hri_message = ROS2HRIMessage.from_ros2(ros2_msg, message_author="human")
        """
        from rai_interfaces.msg import HRIMessage as ROS2HRIMessage_

        if not isinstance(msg, ROS2HRIMessage_):
            raise ValueError(
                "ROS2HRIMessage can only be created from rai_interfaces/msg/HRIMessage"
            )

        cv_bridge = CvBridge()
        images = [
            cv_bridge.imgmsg_to_cv2(img_msg, "rgb8")
            for img_msg in cast(List[ROS2Image], msg.images)
        ]
        pil_images = [Image.fromarray(img) for img in images]
        audio_segments = [
            AudioSegment(
                data=audio_msg.audio,
                frame_rate=audio_msg.sample_rate,
                sample_width=2,  # bytes, int16
                channels=audio_msg.channels,
            )
            for audio_msg in msg.audios
        ]
        return ROS2HRIMessage(
            payload=HRIPayload(text=msg.text, images=pil_images, audios=audio_segments),
            message_author=message_author,
        )

    def to_ros2_dict(self):
        """
        Convert the HRI payload to a ROS2-compatible ordered dictionary.
        
        This method converts the raw HRIPayload associated with the instance into a dictionary format that
        can be used with ROS2 messaging. The conversion process includes the following steps:
        - Images in the payload are converted to ROS2 image messages using CvBridge. Each image is first
          transformed into a NumPy array and then mapped to an image message with an "rgb8" encoding.
        - Audio segments in the payload are converted into ROS2 audio messages using the ROS2HRIMessage__Audio
          helper, where attributes such as raw data, sample rate, and channel count are transferred.
        - A new ROS2HRIMessage_ instance is created with the text, converted images, and converted audio messages.
        - Finally, the ROS2HRIMessage_ instance is converted to an OrderedDict using the rosidl_runtime_py
          conversion utility.
        
        Raises:
            AssertionError: If the payload is not an instance of HRIPayload.
        
        Returns:
            OrderedDict[str, Any]: An ordered dictionary representing the ROS2-compatible HRI message.
        
        Example:
            >>> hri_message = ROS2HRIMessage(payload)
            >>> ros2_message_dict = hri_message.to_ros2_dict()
        """
        cv_bridge = CvBridge()
        assert isinstance(self.payload, HRIPayload)
        img_msgs = [
            cv_bridge.cv2_to_imgmsg(np.array(img), "rgb8")
            for img in self.payload.images
        ]
        audio_msgs = [
            ROS2HRIMessage__Audio(
                audio=audio.raw_data,
                sample_rate=audio.frame_rate,
                channels=audio.channels,
            )
            for audio in self.payload.audios
        ]

        return cast(
            OrderedDict[str, Any],
            rosidl_runtime_py.convert.message_to_ordereddict(
                ROS2HRIMessage_(
                    text=self.payload.text,
                    images=img_msgs,
                    audios=audio_msgs,
                )
            ),
        )


class ROS2HRIConnector(HRIConnector[ROS2HRIMessage]):
    def __init__(
        self,
        node_name: str = f"rai_ros2_hri_connector_{str(uuid.uuid4())[-12:]}",
        targets: List[Union[str, Tuple[str, TopicConfig]]] = [],
        sources: List[Union[str, Tuple[str, TopicConfig]]] = [],
    ):
        """
        Initialize a ROS2 HRI Connector instance.
        
        This constructor sets up the ROS2 node and configures the topic APIs for
        publishing and subscribing HRI messages. It accepts flexible configurations
        for targets and sources, allowing them to be specified either as a simple
        string (the topic name) or as a tuple containing the topic name and a
        TopicConfig instance. If targets or sources are provided as strings, default
        TopicConfig values will be applied (with `is_subscriber` set to False for
        targets and True for sources).
        
        Parameters:
            node_name (str): Name of the ROS2 node. Defaults to a generated string
                in the format "rai_ros2_hri_connector_<UUID_suffix>".
            targets (List[Union[str, Tuple[str, TopicConfig]]]): List of topics for
                publishing messages. Each element can be either a topic name (str) or
                a tuple of the form (topic name, TopicConfig). When specified as a
                string, a default TopicConfig (with `is_subscriber=False`) is used.
            sources (List[Union[str, Tuple[str, TopicConfig]]]): List of topics for
                subscribing to messages. Each element can be either a topic name (str)
                or a tuple of the form (topic name, TopicConfig). When specified as a
                string, a default TopicConfig (with `is_subscriber=True`) is used.
        
        Side Effects:
            - Instantiates a new ROS2 Node.
            - Initializes configurable APIs for topics, services, and actions.
            - Configures publishers and subscribers based on given targets and sources.
            - Passes the processed topic configurations to the superclass initializer.
            - Creates a MultiThreadedExecutor, adds the node to it, and starts a new
              thread to handle asynchronous callback spinning.
        
        Raises:
            Exception: Propagates any exceptions encountered during the initialization
                of the node, APIs, or threading.
        """
        configured_targets = [
            target[0] if isinstance(target, tuple) else target for target in targets
        ]
        configured_sources = [
            source[0] if isinstance(source, tuple) else source for source in sources
        ]

        _targets = [
            target
            if isinstance(target, tuple)
            else (target, TopicConfig(is_subscriber=False))
            for target in targets
        ]
        _sources = [
            source
            if isinstance(source, tuple)
            else (source, TopicConfig(is_subscriber=True))
            for source in sources
        ]

        self._node = Node(node_name)
        self._topic_api = ConfigurableROS2TopicAPI(self._node)
        self._service_api = ROS2ServiceAPI(self._node)
        self._actions_api = ROS2ActionAPI(self._node)

        self._configure_publishers(_targets)
        self._configure_subscribers(_sources)

        super().__init__(configured_targets, configured_sources)

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin)
        self._thread.start()

    def _configure_publishers(self, targets: List[Tuple[str, TopicConfig]]):
        for target in targets:
            self._topic_api.configure_publisher(target[0], target[1])

    def _configure_subscribers(self, sources: List[Tuple[str, TopicConfig]]):
        for source in sources:
            self._topic_api.configure_subscriber(source[0], source[1])

    def send_message(self, message: ROS2HRIMessage, target: str, **kwargs):
        """
        Publish a ROS2 HRI message to the specified target topic.
        
        This method converts the provided ROS2HRIMessage instance into a ROS2-compatible dictionary
        using its to_ros2_dict() method and publishes it via the configured topic API.
        
        Parameters:
            message (ROS2HRIMessage): The HRI message to be published.
            target (str): The identifier of the target topic where the message will be sent.
            **kwargs: Additional keyword arguments (currently not used).
        
        Returns:
            None
        """
        self._topic_api.publish_configured(
            topic=target,
            msg_content=message.to_ros2_dict(),
        )

    def receive_message(
        self,
        source: str,
        timeout_sec: float = 1.0,
        *,
        message_author: Literal["human", "ai"],
        msg_type: Optional[str] = None,
        auto_topic_type: bool = True,
        **kwargs: Any,
    ) -> ROS2HRIMessage:
        """
        Receives a message from the specified source and converts it into a ROS2HRIMessage.
        
        This method retrieves a message using the underlying topic API with the given source and timeout settings.
        The received message is then converted into a ROS2HRIMessage instance using the provided message_author.
        
        Parameters:
            source (str): The name of the topic or source from which to receive the message.
            timeout_sec (float, optional): Maximum time (in seconds) to wait for a message. Defaults to 1.0.
            message_author (Literal["human", "ai"]): The designated author of the message; must be either "human" or "ai".
            msg_type (Optional[str], optional): An optional message type specifier (currently not used in processing).
            auto_topic_type (bool, optional): If True, the topic API will automatically detect the message type. Defaults to True.
            **kwargs (Any): Additional keyword arguments for future extension (currently unused).
        
        Returns:
            ROS2HRIMessage: An instance of ROS2HRIMessage created from the received ROS2 message.
        
        Raises:
            Exception: Propagates any exception raised by the underlying topic API's receive method or during message conversion.
        """
        msg = self._topic_api.receive(
            topic=source,
            timeout_sec=timeout_sec,
            auto_topic_type=auto_topic_type,
        )
        return ROS2HRIMessage.from_ros2(msg, message_author)

    def service_call(
        self, message: ROS2HRIMessage, target: str, timeout_sec: float, **kwargs: Any
    ) -> ROS2HRIMessage:
        """
        Attempt to perform a service call with a ROS2HRIMessage.
        
        This method is not implemented since the ROS2HRIConnector does not support service calls.
        
        Parameters:
            message (ROS2HRIMessage): The ROS2 HRI message to be used in the service call.
            target (str): The identifier of the service target.
            timeout_sec (float): The maximum time in seconds to wait for a response.
            **kwargs: Additional keyword arguments that might be used for the service call.
        
        Raises:
            NotImplementedError: Always raised to indicate that service calls are not supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support service calls"
        )

    def start_action(
        self,
        action_data: Optional[ROS2HRIMessage],
        target: str,
        on_feedback: Callable,
        on_done: Callable,
        timeout_sec: float,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support action calls"
        )

    def terminate_action(self, action_handle: str, **kwargs: Any):
        """
        Terminate an ongoing action (if supported).
        
        This method is not implemented for the current connector as action calls are not supported. Calling this method will always raise a NotImplementedError.
        
        Parameters:
            action_handle (str): Unique identifier of the action to be terminated.
            **kwargs: Additional keyword arguments specific to the connector (unused).
        
        Raises:
            NotImplementedError: Always raised to indicate that action calls are not supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support action calls"
        )

    def shutdown(self):
        """
        Cleanly shuts down all ROS2 connector resources.
        
        This method performs a sequence of cleanup operations to gracefully terminate the connector:
        1. Shuts down the multi-threaded executor.
        2. Waits for the executor thread to finish.
        3. Shuts down both the actions and topic APIs.
        4. Destroys the associated ROS2 node.
        
        This ensures that all background processes and resources are properly released.
        """
        self._executor.shutdown()
        self._thread.join()
        self._actions_api.shutdown()
        self._topic_api.shutdown()
        self._node.destroy_node()
