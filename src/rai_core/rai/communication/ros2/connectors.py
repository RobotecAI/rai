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

import rai_interfaces.msg
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
        super().__init__(payload, message_author)

    @classmethod
    def from_ros2(
        cls, msg: rai_interfaces.msg.HRIMessage, message_author: Literal["ai", "human"]
    ):
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

    def to_ros2_dict(self) -> OrderedDict[str, Any]:
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
        msg = self._topic_api.receive(
            topic=source,
            timeout_sec=timeout_sec,
            auto_topic_type=auto_topic_type,
        )
        return ROS2HRIMessage.from_ros2(msg, message_author)

    def service_call(
        self, message: ROS2HRIMessage, target: str, timeout_sec: float, **kwargs: Any
    ) -> ROS2HRIMessage:
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
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support action calls"
        )

    def shutdown(self):
        self._executor.shutdown()
        self._thread.join()
        self._actions_api.shutdown()
        self._topic_api.shutdown()
        self._node.destroy_node()
