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

import re
from abc import abstractmethod
from queue import Queue

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from rai_hmi.base import BaseHMINode


def split_message(message: str):
    sentences = re.split(r"(?<=\.)\s|[:!]", message)
    for sentence in sentences:
        if sentence:
            yield sentence


class GenericVoiceNode(BaseHMINode):
    def __init__(self, node_name: str, queue: Queue, robot_description_package: str):
        super().__init__(node_name, queue, robot_description_package)

        self.callback_group = ReentrantCallbackGroup()
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_ALL,
        )
        self.hmi_subscription = self.create_subscription(
            String,
            "/from_human",
            self.handle_human_message,
            qos_profile=reliable_qos,
        )
        self.hmi_publisher = self.create_publisher(
            String,
            "/to_human",
            qos_profile=reliable_qos,
            callback_group=self.callback_group,
        )

    @abstractmethod
    def _handle_human_message(self, msg: String):
        pass

    def handle_human_message(self, msg: String):
        self.processing = True
        self._handle_human_message(msg)
        self.processing = False
