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

import importlib.util
import logging
import uuid
from typing import Any, Optional

from rclpy.qos import QoSProfile

from rai.communication import HRIConnector
from rai.communication.ros2.connectors.base import ROS2BaseConnector
from rai.communication.ros2.messages import ROS2HRIMessage

if importlib.util.find_spec("rai_interfaces.msg") is None:
    logging.warning(
        "This feature is based on rai_interfaces.msg. Make sure rai_interfaces is installed."
    )


class ROS2HRIConnector(ROS2BaseConnector[ROS2HRIMessage], HRIConnector[ROS2HRIMessage]):
    def __init__(
        self,
        node_name: str = f"rai_ros2_hri_connector_{str(uuid.uuid4())[-12:]}",
    ):
        super().__init__(node_name=node_name)

    def send_message(
        self,
        message: ROS2HRIMessage,
        target: str,
        *,
        qos_profile: Optional[QoSProfile] = None,
        auto_qos_matching: bool = True,
        **kwargs,
    ):
        self._topic_api.publish(
            topic=target,
            msg_content=message.to_ros2_dict(),
            msg_type="rai_interfaces/msg/HRIMessage",
            auto_qos_matching=auto_qos_matching,
            qos_profile=qos_profile,
        )

    def general_callback_preprocessor(self, message: Any) -> ROS2HRIMessage:
        return ROS2HRIMessage.from_ros2(message, message_author="human")
