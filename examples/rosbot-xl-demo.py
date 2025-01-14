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
import argparse
from pathlib import Path
from typing import Optional

import rclpy
import rclpy.executors
import rclpy.logging
from rai_open_set_vision.tools import GetDetectionTool, GetDistanceToObjectsTool

from rai.node import RaiStateBasedLlmNode
from rai.tools.ros.native import (
    GetMsgFromTopic,
    Ros2GetRobotInterfaces,
    Ros2PubMessageTool,
    Ros2ShowMsgInterfaceTool,
)
from rai.tools.ros.native_actions import (
    GetTransformTool,
    Ros2CancelAction,
    Ros2GetActionResult,
    Ros2GetLastActionFeedback,
    Ros2RunActionAsync,
)
from rai.tools.time import WaitForSecondsTool

p = argparse.ArgumentParser()
p.add_argument("--allowlist", type=Path, required=False, default=None)


def main(allowlist: Optional[Path] = None):
    rclpy.init()
    # observe_topics = [
    #     "/camera/camera/color/image_raw",
    # ]
    #
    # observe_postprocessors = {"/camera/camera/color/image_raw": describe_ros_image}
    ros2_allowlist = []
    if allowlist is not None:
        try:
            content = allowlist.read_text().strip()
            if content:
                ros2_allowlist = content.splitlines()
            else:
                rclpy.logging.get_logger("rosbot_xl_demo").warning(
                    "Allowlist file is empty"
                )
        except Exception as e:
            rclpy.logging.get_logger("rosbot_xl_demo").error(
                f"Failed to read allowlist: {e}"
            )
    else:
        ros2_allowlist = None

    SYSTEM_PROMPT = """You are an autonomous robot connected to ros2 environment. Your main goal is to fulfill the user's requests.
    Do not make assumptions about the environment you are currently in.
    You can use ros2 topics, services and actions to operate.

    <rule> As a first step check transforms by getting 1 message from /tf topic </rule>
    <rule> use /cmd_vel topic very carefully. Obstacle detection works only with nav2 stack, so be careful when it is not used. </rule>>
    <rule> be patient with running ros2 actions. usually the take some time to run. </rule>
    <rule> Always check your transform before and after you perform ros2 actions, so that you can verify if it worked. </rule>

    Navigation tips:
    - it's good to start finding objects by rotating, then navigating to some diverse location with occasional rotations. Remember to frequency detect objects.
    - for driving forward/backward or to some coordinates, ros2 actions are better.
    - for driving for some specific time or in specific manner (like shaper or turns) it good to use /cmd_vel topic
    - you are currently unable to read map or point-cloud, so please avoid subscribing to such topics.
    - if you are asked to drive towards some object, it's good to:
        1. check the camera image and verify if objects can be seen
        2. if only driving forward is required, do it
        3. if obstacle avoidance might be required, use ros2 actions navigate_*, but first check your current position, then very accurately estimate the goal pose.
    - it is good to verify using given information if the robot is not stuck
    - navigation actions sometimes fail. Their output can be read from rosout. You can also tell if they partially worked by checking the robot position and rotation.
    - before using any ros2 interfaces, always make sure to check you are using the right interface
    - processing camera image takes 5-10s. Take it into account that if the robot is moving, the information can be outdated. Handle it by good planning of your movements.
    - you are encouraged to use wait tool in between checking the status of actions
    - to find some object navigate around and check the surrounding area
    - when the goal is accomplished please make sure to cancel running actions
    - when you reach the navigation goal - double check if you reached it by checking the current position
    - if you detect collision, please stop operation

    - you will be given your camera image description. Based on this information you can reason about positions of objects.
    - be careful and aboid obstacles

    Here are the corners of your environment:
    (-2.76,9.04, 0.0),
    (4.62, 9.07, 0.0),
    (-2.79, -3.83, 0.0),
    (4.59, -3.81, 0.0)

    This is location of places:
    Kitchen:
    (2.06, -0.23, 0.0),
    (2.07, -1.43, 0.0),
    (-2.44, -0.38, 0.0),
    (-2.56, -1.47, 0.0)

    # Living room:
    (-2.49, 1.87, 0.0),
    (-2.50, 5.49, 0.0),
    (0.79, 5.73, 0.0),
    (0.92, 1.01, 0.0)
    """

    node = RaiStateBasedLlmNode(
        observe_topics=None,
        observe_postprocessors=None,
        allowlist=ros2_allowlist,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            Ros2GetRobotInterfaces,
            Ros2PubMessageTool,
            Ros2RunActionAsync,
            Ros2CancelAction,
            Ros2GetActionResult,
            Ros2GetLastActionFeedback,
            Ros2ShowMsgInterfaceTool,
            GetTransformTool,
            WaitForSecondsTool,
            GetMsgFromTopic,
            GetDetectionTool,
            GetDistanceToObjectsTool,
        ],
    )
    node.declare_parameter("conversion_ratio", 1.0)

    node.spin()
    rclpy.shutdown()


if __name__ == "__main__":
    args = p.parse_args()
    main(**vars(args))
