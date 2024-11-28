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
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from rai.node import RaiStateBasedLlmNode, describe_ros_image
from rai.tools.ros.native import (
    GetCameraImage,
    GetMsgFromTopic,
    Ros2GenericServiceCaller,
    Ros2GetRobotInterfaces,
    Ros2ShowMsgInterfaceTool,
)
from rai.tools.time import WaitForSecondsTool
from rai_interfaces.action import Task


class MockBehaviorTreeNode(Node):
    def __init__(self, tractor_number: int):
        super().__init__(f"mock_behavior_tree_node_{tractor_number}")
        self.tractor_number = tractor_number

        # Create a callback group for concurrent execution
        self.callback_group = ReentrantCallbackGroup()

        # Create service client
        self.current_state_client = self.create_client(
            Trigger,
            f"/tractor{tractor_number}/current_state",
            callback_group=self.callback_group,
        )

        # Create action client
        self.perform_task_client = ActionClient(
            self, Task, "/perform_task", callback_group=self.callback_group
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

        if "STOPPED" in response.message:
            self.get_logger().info(
                "The tractor has stopped. Calling RaiNode to decide what to do."
            )

            # Send goal to perform_task action server
            goal_msg = Task.Goal()
            goal_msg.priority = 10
            goal_msg.description = ""
            goal_msg.task = "Anomaly detected. Please decide what to do."

            self.perform_task_client.wait_for_server()

            future = self.perform_task_client.send_goal_async(goal_msg)
            await future

            goal_handle = future.result()
            if goal_handle.accepted:
                self.get_logger().info("Goal accepted by perform_task action server")
                result = await goal_handle.get_result_async()
                self.get_logger().info(f"Result: {result.result}")
            else:
                self.get_logger().warn("Goal rejected by perform_task action server")


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
    tractor_prefix = f"/tractor{tractor_number}"

    rclpy.init()

    observe_topics = [
        f"{tractor_prefix}/camera_image_color",
    ]

    observe_postprocessors = {
        f"{tractor_prefix}/camera_image_color": describe_ros_image
    }

    topics_allowlist = [
        "/rosout",
        f"{tractor_prefix}/camera_image_color",
        # Services
        f"{tractor_prefix}/continue",
        f"{tractor_prefix}/current_state",
        f"{tractor_prefix}/flash",
        f"{tractor_prefix}/replan",
    ]

    actions_allowlist = []

    SYSTEM_PROMPT = f"""
    You are autonomous tractor {tractor_number} operating in an agricultural field. You are activated whenever the tractor stops due to an unexpected situation. Your task is to call a service based on your assessment of the situation.

    The system is not perfect, so it may stop you unnecessarily at times.

    You are to call one of the following services based on the situation:
    1. continue - If you believe the stop was unnecessary.
    2. current_state - If you want to check the current state of the tractor.
    3. flash - If you want to flash the lights to signal possible animals to move out of the way.
    4. replan - If you want to replan the path due to an obstacle in front of you.

    Important: You must call only one service. The tractor can only handle one service call.
    """

    rai_node = RaiStateBasedLlmNode(
        observe_topics=observe_topics,
        observe_postprocessors=observe_postprocessors,
        allowlist=topics_allowlist + actions_allowlist,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            Ros2ShowMsgInterfaceTool,
            WaitForSecondsTool,
            GetMsgFromTopic,
            GetCameraImage,
            Ros2GetRobotInterfaces,
            Ros2GenericServiceCaller,
        ],
    )

    mock_node = MockBehaviorTreeNode(tractor_number)

    # Use a MultiThreadedExecutor to allow for concurrent execution
    executor = MultiThreadedExecutor()
    executor.add_node(rai_node)
    executor.add_node(mock_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rai_node.destroy_node()
        mock_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
