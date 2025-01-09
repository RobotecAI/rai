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

import os
import threading
import uuid
from typing import List

import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles, QoSProfile
from sensor_msgs.msg import Image
from std_msgs.msg import String

from rai.node import RaiBaseNode


def get_qos_profiles() -> List[str]:
    ros_distro = os.environ.get("ROS_DISTRO")
    match ros_distro:
        case "humble":
            # TODO: Humble fails in CI, while it works locally on a clean ubuntu 22.04.
            return [
                # QoSPresetProfiles.SYSTEM_DEFAULT.name,
                # QoSPresetProfiles.SENSOR_DATA.name,
                # QoSPresetProfiles.SERVICES_DEFAULT.name,
                # QoSPresetProfiles.PARAMETERS.name,
                # QoSPresetProfiles.PARAMETER_EVENTS.name,
                # QoSPresetProfiles.ACTION_STATUS_DEFAULT.name,
            ]
        case "jazzy":
            return [
                QoSPresetProfiles.DEFAULT.name,
                QoSPresetProfiles.SYSTEM_DEFAULT.name,
                QoSPresetProfiles.SENSOR_DATA.name,
                QoSPresetProfiles.SERVICES_DEFAULT.name,
                QoSPresetProfiles.PARAMETERS.name,
                QoSPresetProfiles.PARAMETER_EVENTS.name,
                QoSPresetProfiles.ACTION_STATUS_DEFAULT.name,
                QoSPresetProfiles.BEST_AVAILABLE.name,
            ]
        case _:
            raise ValueError(f"Invalid ROS_DISTRO: {ros_distro}")


class TestPublisher(Node):
    def __init__(self, qos_profile: QoSProfile):
        super().__init__("test_publisher_" + str(uuid.uuid4()).replace("-", ""))

        self.image_publisher = self.create_publisher(Image, "image", qos_profile)
        self.text_publisher = self.create_publisher(String, "text", qos_profile)

        self.image_timer = self.create_timer(0.1, self.image_callback)
        self.text_timer = self.create_timer(0.1, self.text_callback)

    def image_callback(self):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = 540
        msg.width = 960
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = np.random.randint(0, 255, size=msg.height * msg.width * 3).tobytes()
        self.image_publisher.publish(msg)

    def text_callback(self):
        msg = String()
        msg.data = "Hello, world!"
        self.text_publisher.publish(msg)


@pytest.mark.parametrize(
    "qos_profile",
    get_qos_profiles(),
)
def test_transport(qos_profile: str):
    if not rclpy.ok():
        rclpy.init()
    publisher = TestPublisher(QoSPresetProfiles.get_from_short_key(qos_profile))
    executor = SingleThreadedExecutor()
    executor.add_node(publisher)
    thread = threading.Thread(target=executor.spin)
    thread.start()

    rai_base_node = RaiBaseNode(
        node_name="test_transport_" + str(uuid.uuid4()).replace("-", "")
    )
    topics = ["/image", "/text"]
    try:
        for topic in topics:
            output = rai_base_node.get_raw_message_from_topic(topic, timeout_sec=5.0)
            assert not isinstance(output, str), "No message received"
    finally:
        executor.shutdown()
        publisher.destroy_node()
        rclpy.shutdown()
        thread.join(timeout=1.0)
