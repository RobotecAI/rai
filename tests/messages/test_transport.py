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

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from rai.node import RaiBaseNode


class TestPublisher(Node):
    def __init__(self):
        super().__init__("test_publisher")
        self.image_publisher = self.create_publisher(
            Image, "image", qos_profile_sensor_data
        )
        self.text_publisher = self.create_publisher(String, "text", 10)

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


def test_transport():
    rclpy.init()
    publisher = TestPublisher()
    thread = threading.Thread(target=rclpy.spin, args=(publisher,))
    thread.start()

    rai_base_node = RaiBaseNode(node_name="test_transport")
    topics = ["/image", "/text"]
    for topic in topics:
        output = rai_base_node.get_raw_message_from_topic(topic)
        assert output is not None

    publisher.destroy_node()
    rclpy.shutdown()
    thread.join()
