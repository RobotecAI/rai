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
# See the License for the specific language goveself.rning permissions and
# limitations under the License.

import argparse

import rclpy
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from rai import get_llm_model
from rai.agents.langchain.core import (
    ConversationalAgentState,
    create_conversational_agent,
)
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2 import ROS2ServicesToolkit
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool
from rai.tools.time import WaitForSecondsTool
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from rai_whoami.models import EmbodimentInfo


class MockBehaviorTreeNode(Node):
    def __init__(
        self,
        tractor_number: int,
        agent: Runnable[ConversationalAgentState, ConversationalAgentState],
    ):
        super().__init__(f"mock_behavior_tree_node_{tractor_number}")
        self.tractor_number = tractor_number
        self.agent = agent
        self.working = False
        # Create a callback group for concurrent execution
        self.callback_group = ReentrantCallbackGroup()

        # Create service client
        self.current_state_client = self.create_client(
            Trigger,
            f"/tractor{tractor_number}/current_state",
            callback_group=self.callback_group,
        )

        # Create timer for periodic checks
        self.create_timer(
            5.0, self.check_tractor_state, callback_group=self.callback_group
        )

        self.get_logger().info(
            f"Mock Behavior Tree Node for Tractor {tractor_number} initialized"
        )

    async def check_tractor_state(self):
        # Call the current_state service
        response = await self.current_state_client.call_async(Trigger.Request())

        self.get_logger().info(f"Current state: {response.message}")

        if "STOPPED" in response.message and not self.working:
            self.get_logger().info(
                "The tractor has stopped. Calling RAI Agent to decide what to do."
            )

            self.working = True
            self.agent.invoke(
                ConversationalAgentState(
                    messages=[
                        HumanMessage(
                            content="Anomaly has been detected. The tractor has stopped. Please decide what to do."
                        )
                    ]
                )
            )
            self.working = False


def main():
    parser = argparse.ArgumentParser(description="Autonomous Tractor Demo")
    parser.add_argument(
        "--tractor_number",
        type=int,
        choices=[1, 2],
        help="Tractor number (1 or 2)",
        default=1,
    )
    args = parser.parse_args()

    tractor_number = args.tractor_number

    rclpy.init()

    system_prompt = EmbodimentInfo.from_file(
        "examples/embodiments/agriculture_embodiment.json"
    ).to_langchain()
    connector = ROS2Connector()
    agent = create_conversational_agent(
        llm=get_llm_model("complex_model"),
        system_prompt=system_prompt,
        tools=[
            GetROS2ImageConfiguredTool(
                connector=connector,
                topic=f"/tractor{args.tractor_number}/camera_image_color",
            ),
            *ROS2ServicesToolkit(
                connector=connector,
                writable=[
                    f"/tractor{args.tractor_number}/continue",
                    f"/tractor{args.tractor_number}/current_state",
                    f"/tractor{args.tractor_number}/flash",
                    f"/tractor{args.tractor_number}/replan",
                    f"/tractor{args.tractor_number}/stop",
                ],
            ).get_tools(),
            WaitForSecondsTool(),
        ],
    )

    mock_node = MockBehaviorTreeNode(tractor_number, agent)

    # Use a MultiThreadedExecutor to allow for concurrent execution
    executor = MultiThreadedExecutor()
    executor.add_node(mock_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        connector.shutdown()
        mock_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
