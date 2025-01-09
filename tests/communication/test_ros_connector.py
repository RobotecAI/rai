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
from typing import List, Optional

import pytest
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from rai.communication.base_connector import BaseMessage
from rai.communication.ros_connector import ROS2Connector


class ReceiverNode(Node):
    def __init__(self):
        super().__init__("receiver")
        self.received_images: List[Image] = []
        self.received_strings: List[String] = []
        self.group = ReentrantCallbackGroup()
        self.string_sub = self.create_subscription(  # type: ignore
            String, "/text", self.on_message, 10, callback_group=self.group
        )
        self.image_sub = self.create_subscription(  # type: ignore
            Image, "/image", self.on_message, 10, callback_group=self.group
        )

    def on_message(self, msg: Optional[String | Image]):
        if isinstance(msg, String):
            self.received_strings.append(msg)
        elif isinstance(msg, Image):
            self.received_images.append(msg)
        else:
            raise ValueError(f"Unknown message type: {type(msg)}")


class SenderNode(Node):
    def __init__(self):
        super().__init__("sender")
        self.text_pub = self.create_publisher(  # type: ignore
            String,
            "/text",
            10,
        )
        self.image_pub = self.create_publisher(  # type: ignore
            Image,
            "/image",
            10,
        )
        self.timer = self.create_timer(0.1, self.on_timer)  # type: ignore

    def on_timer(self):
        self.text_pub.publish(String(data="Hello, world!"))
        self.image_pub.publish(Image())


def test_send_message():
    rclpy.init()
    receiver = ReceiverNode()
    receiver_executor = MultiThreadedExecutor()
    receiver_executor.add_node(receiver)
    receiver_thread = threading.Thread(target=receiver_executor.spin)
    receiver_thread.start()

    connector = ROS2Connector()
    connector.send_message(BaseMessage(String(data="Hello, world!")), "/text")
    connector.send_message(BaseMessage(Image()), "/image")

    time.sleep(1.0)
    assert len(receiver.received_strings) == 1
    assert len(receiver.received_images) == 1

    rclpy.shutdown()
    receiver_thread.join()


def test_receive_message():
    rclpy.init()
    sender = SenderNode()
    sender_executor = MultiThreadedExecutor()
    sender_executor.add_node(sender)
    sender_thread = threading.Thread(target=sender_executor.spin)
    sender_thread.start()

    connector = ROS2Connector()
    message = connector.receive_message("/text")
    assert isinstance(message, BaseMessage)
    assert isinstance(message.content, String)

    message = connector.receive_message("/image")
    assert isinstance(message, BaseMessage)
    assert isinstance(message.content, Image)

    rclpy.shutdown()
    sender_thread.join()


@pytest.mark.skip(reason="Not implemented")
def test_start_action():
    pass


@pytest.mark.skip(reason="Not implemented")
def test_terminate_action():
    pass


@pytest.mark.skip(reason="Not implemented")
def test_send_and_wait():
    pass


def test_connector_cleanup():
    rclpy.init()
    connector = ROS2Connector()

    connector.send_message(BaseMessage(String(data="Test")), "/text")
    connector.send_message(BaseMessage(Image()), "/image")

    initial_publisher_count = len(connector.publishers)
    assert initial_publisher_count == 2

    connector.send_message(BaseMessage(String(data="Test")), "/text")
    assert (
        len(connector.publishers) == initial_publisher_count
    )  # reuses existing publishers

    connector.cleanup()
    assert not connector.executor_thread.is_alive()

    rclpy.shutdown()
    assert not connector.node.context.ok()


def test_invalid_topic():
    rclpy.init()
    connector = ROS2Connector()

    with pytest.raises(ValueError, match="Topic '/nonexistent' not found"):
        connector.receive_message("/nonexistent")

    rclpy.shutdown()
