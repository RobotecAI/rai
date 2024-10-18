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
#


import argparse
import subprocess
import threading
import time

import rclpy

from rai.node import RaiStateBasedLlmNode, describe_ros_image
from rai.tools.ros.native import (
    GetCameraImage,
    GetMsgFromTopic,
    Ros2GenericServiceCaller,
    Ros2GetRobotInterfaces,
    Ros2ShowMsgInterfaceTool,
)
from rai.tools.time import WaitForSecondsTool


def mock_behavior_tree(tractor_number: int):
    """
    This is a mock of behavior tree that simulates scenario where the tractor stops
    due to an unexpected situation and calls the RaiNode to decide what to do.
    """

    while True:
        output = subprocess.check_output(
            [
                "ros2",
                "service",
                "call",
                f"/tractor{tractor_number}/current_state",
                "std_srvs/srv/Trigger",
                "{}",
            ]
        )
        print(output.decode("utf-8"))
        if "STOPPED" in output.decode("utf-8"):
            print("The tractor has stopped. Calling RaiNode to decide what to do.")
            action_output = subprocess.check_output(
                [
                    "ros2",
                    "action",
                    "send_goal",
                    "-f",
                    "/perform_task",
                    "rai_interfaces/action/Task",
                    "{priority: 10, description: '', task: 'Anomaly detected. Please decide what to do.'}",
                ]
            )
            print(action_output.decode("utf-8"))
        time.sleep(5)


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

    topics_whitelist = [
        "/rosout",
        f"{tractor_prefix}/camera_image_color",
        # Services
        f"{tractor_prefix}/continue",
        f"{tractor_prefix}/current_state",
        f"{tractor_prefix}/flash",
        f"{tractor_prefix}/replan",
    ]

    actions_whitelist = []

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

    node = RaiStateBasedLlmNode(
        observe_topics=observe_topics,
        observe_postprocessors=observe_postprocessors,
        whitelist=topics_whitelist + actions_whitelist,
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

    thread = threading.Thread(target=mock_behavior_tree, args=(tractor_number,))
    thread.start()

    node.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
