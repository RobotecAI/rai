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

from typing import Any, Callable, Optional

from rclpy.qos import QoSProfile

from rai.communication import HRIConnector
from rai.communication.ros2.connectors.ros2_connector import QoS, ROS2Base
from rai.communication.ros2.messages import ROS2HRIMessage
from rai.communication.ros2.runtime import ExecutorKind, ROS2Runtime

HRI_TYPE = "rai_interfaces/msg/HRIMessage"


class ROS2HRIConnector(ROS2Base, HRIConnector[ROS2HRIMessage]):
    """Sends and receives multimodal ``rai_interfaces/msg/HRIMessage``s (text,
    images, audio) on the shared ROS2Runtime.

    HRI QoS is reliable and its callback queues never drop: a message from a
    human is always delivered, in order. Received messages are decoded off the
    executor thread.
    """

    def __init__(
        self,
        node_name: Optional[str] = None,
        *,
        runtime: Optional[ROS2Runtime] = None,
        executor: Optional[ExecutorKind] = None,
        use_sim_time: bool = False,
        executor_type: Optional[str] = None,  # R5: deprecated, ignored
    ) -> None:
        HRIConnector.__init__(self)
        self._attach(node_name, runtime, executor, use_sim_time, executor_type)

    def close(self) -> None:
        super().close()
        self.callback_executor.shutdown(wait=False)

    def send_message(
        self,
        message: ROS2HRIMessage,
        target: str,
        *,
        qos_profile: Optional[QoSProfile] = None,
        **kwargs: Any,
    ) -> None:
        self._check()
        self._runtime.publish(target, message.to_ros2_msg(), qos_profile or QoS.HRI)

    def register_callback(
        self,
        source: str,
        callback: Callable[[Any], Any],
        raw: bool = False,
        *,
        msg_type: Optional[str] = None,
        qos_profile: Optional[QoSProfile] = None,
        **kwargs: Any,
    ) -> str:
        """Call callback(ROS2HRIMessage) for every message (the ROS message if
        raw). Returns an id for unregister_callback."""

        def decoded(msg: Any) -> Any:
            return callback(ROS2HRIMessage.from_ros2(msg, message_author="human"))

        return self._register(
            source,
            callback if raw else decoded,
            msg_type or HRI_TYPE,
            qos_profile or QoS.HRI,
            None,
        )

    def receive_message(
        self,
        source: str,
        timeout_sec: float = 1.0,
        *,
        msg_type: Optional[str] = None,
        qos_profile: Optional[QoSProfile] = None,
        max_age: Optional[float] = None,
        **kwargs: Any,
    ) -> ROS2HRIMessage:
        msg = self._runtime.run(
            self.receive(
                source,
                msg_type=msg_type or HRI_TYPE,
                timeout=timeout_sec,
                max_age=max_age,
                qos=qos_profile or QoS.HRI,
            )
        )
        return ROS2HRIMessage.from_ros2(msg, message_author="human")
