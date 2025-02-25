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

import pytest
import rclpy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from rai.agents.state_based import create_state_based_agent
from rai.tools.ros.native import Ros2PubMessageTool


class Subscriber(Node):
    def __init__(self) -> None:
        super().__init__("subscriber")

        self.sub = self.create_subscription(
            String,
            "/rai/tests/ros2_topic",
            self.callback,
            10,
        )

        self.get_logger().info("Subscriber created")
        self.callback_called = False
        self.received_message = ""

    def callback(self, msg):
        self.get_logger().info(f"Received message: {msg.data}")
        self.received_message = msg.data
        self.callback_called = True


class RosNode(Node):
    def __init__(self):
        super().__init__("ros_node")


@pytest.mark.billable
@pytest.mark.timeout(10)
@pytest.mark.parametrize(
    "test_message, topic",
    [
        ("test", "/rai/tests/ros2_topic"),
    ],
)
def test_ros2_pub_message_tool_llm(
    chat_openai_text: ChatOpenAI, test_message: str, topic: str
):
    rclpy.init()
    ros_node = RosNode()
    pub_ros2_message_tool = Ros2PubMessageTool(node=ros_node)
    tools = [pub_ros2_message_tool]
    subscriber = Subscriber()

    llm_with_tools = create_state_based_agent(
        llm=chat_openai_text,
        tools=tools,
        state_retriever=lambda: {},
        logger=ros_node.get_logger(),
    )

    executor = MultiThreadedExecutor()
    executor.add_node(ros_node)
    executor.add_node(subscriber)
    t = threading.Thread(target=executor.spin)

    system = SystemMessage(
        "You are a ros2 agent that can run tools: {render_text_description_and_args(tools)}"
    )
    query = HumanMessage(
        f"Publish a std_msgs/msg/String '{test_message}' to the topic '{topic}'"
    )

    messages = [system, query]
    try:
        t.start()
        llm_with_tools.invoke(dict(messages=messages))

        while not subscriber.callback_called:
            pass
        assert subscriber.received_message == "test"
    finally:
        executor.shutdown()
        t.join()
